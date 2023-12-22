from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[384, 1]"; primals_2: "f32[384, 1]"; primals_3: "f32[384, 1]"; primals_4: "f32[384, 1]"; primals_5: "f32[384, 1]"; primals_6: "f32[384, 1]"; primals_7: "f32[384, 1]"; primals_8: "f32[384, 1]"; primals_9: "f32[384, 1]"; primals_10: "f32[384, 1]"; primals_11: "f32[384, 1]"; primals_12: "f32[384, 1]"; primals_13: "f32[30522, 768]"; primals_14: "f32[512, 768]"; primals_15: "f32[2, 768]"; primals_16: "f32[768]"; primals_17: "f32[768]"; primals_18: "f32[384, 768]"; primals_19: "f32[384]"; primals_20: "f32[384, 768]"; primals_21: "f32[384]"; primals_22: "f32[384, 768]"; primals_23: "f32[384]"; primals_24: "f32[768, 1, 9]"; primals_25: "f32[384, 768, 1]"; primals_26: "f32[54, 384]"; primals_27: "f32[54]"; primals_28: "f32[384, 768]"; primals_29: "f32[384]"; primals_30: "f32[768, 768]"; primals_31: "f32[768]"; primals_32: "f32[768]"; primals_33: "f32[768]"; primals_34: "f32[3072, 768]"; primals_35: "f32[3072]"; primals_36: "f32[768, 3072]"; primals_37: "f32[768]"; primals_38: "f32[768]"; primals_39: "f32[768]"; primals_40: "f32[384, 768]"; primals_41: "f32[384]"; primals_42: "f32[384, 768]"; primals_43: "f32[384]"; primals_44: "f32[384, 768]"; primals_45: "f32[384]"; primals_46: "f32[768, 1, 9]"; primals_47: "f32[384, 768, 1]"; primals_48: "f32[54, 384]"; primals_49: "f32[54]"; primals_50: "f32[384, 768]"; primals_51: "f32[384]"; primals_52: "f32[768, 768]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[768]"; primals_56: "f32[3072, 768]"; primals_57: "f32[3072]"; primals_58: "f32[768, 3072]"; primals_59: "f32[768]"; primals_60: "f32[768]"; primals_61: "f32[768]"; primals_62: "f32[384, 768]"; primals_63: "f32[384]"; primals_64: "f32[384, 768]"; primals_65: "f32[384]"; primals_66: "f32[384, 768]"; primals_67: "f32[384]"; primals_68: "f32[768, 1, 9]"; primals_69: "f32[384, 768, 1]"; primals_70: "f32[54, 384]"; primals_71: "f32[54]"; primals_72: "f32[384, 768]"; primals_73: "f32[384]"; primals_74: "f32[768, 768]"; primals_75: "f32[768]"; primals_76: "f32[768]"; primals_77: "f32[768]"; primals_78: "f32[3072, 768]"; primals_79: "f32[3072]"; primals_80: "f32[768, 3072]"; primals_81: "f32[768]"; primals_82: "f32[768]"; primals_83: "f32[768]"; primals_84: "f32[384, 768]"; primals_85: "f32[384]"; primals_86: "f32[384, 768]"; primals_87: "f32[384]"; primals_88: "f32[384, 768]"; primals_89: "f32[384]"; primals_90: "f32[768, 1, 9]"; primals_91: "f32[384, 768, 1]"; primals_92: "f32[54, 384]"; primals_93: "f32[54]"; primals_94: "f32[384, 768]"; primals_95: "f32[384]"; primals_96: "f32[768, 768]"; primals_97: "f32[768]"; primals_98: "f32[768]"; primals_99: "f32[768]"; primals_100: "f32[3072, 768]"; primals_101: "f32[3072]"; primals_102: "f32[768, 3072]"; primals_103: "f32[768]"; primals_104: "f32[768]"; primals_105: "f32[768]"; primals_106: "f32[384, 768]"; primals_107: "f32[384]"; primals_108: "f32[384, 768]"; primals_109: "f32[384]"; primals_110: "f32[384, 768]"; primals_111: "f32[384]"; primals_112: "f32[768, 1, 9]"; primals_113: "f32[384, 768, 1]"; primals_114: "f32[54, 384]"; primals_115: "f32[54]"; primals_116: "f32[384, 768]"; primals_117: "f32[384]"; primals_118: "f32[768, 768]"; primals_119: "f32[768]"; primals_120: "f32[768]"; primals_121: "f32[768]"; primals_122: "f32[3072, 768]"; primals_123: "f32[3072]"; primals_124: "f32[768, 3072]"; primals_125: "f32[768]"; primals_126: "f32[768]"; primals_127: "f32[768]"; primals_128: "f32[384, 768]"; primals_129: "f32[384]"; primals_130: "f32[384, 768]"; primals_131: "f32[384]"; primals_132: "f32[384, 768]"; primals_133: "f32[384]"; primals_134: "f32[768, 1, 9]"; primals_135: "f32[384, 768, 1]"; primals_136: "f32[54, 384]"; primals_137: "f32[54]"; primals_138: "f32[384, 768]"; primals_139: "f32[384]"; primals_140: "f32[768, 768]"; primals_141: "f32[768]"; primals_142: "f32[768]"; primals_143: "f32[768]"; primals_144: "f32[3072, 768]"; primals_145: "f32[3072]"; primals_146: "f32[768, 3072]"; primals_147: "f32[768]"; primals_148: "f32[768]"; primals_149: "f32[768]"; primals_150: "f32[384, 768]"; primals_151: "f32[384]"; primals_152: "f32[384, 768]"; primals_153: "f32[384]"; primals_154: "f32[384, 768]"; primals_155: "f32[384]"; primals_156: "f32[768, 1, 9]"; primals_157: "f32[384, 768, 1]"; primals_158: "f32[54, 384]"; primals_159: "f32[54]"; primals_160: "f32[384, 768]"; primals_161: "f32[384]"; primals_162: "f32[768, 768]"; primals_163: "f32[768]"; primals_164: "f32[768]"; primals_165: "f32[768]"; primals_166: "f32[3072, 768]"; primals_167: "f32[3072]"; primals_168: "f32[768, 3072]"; primals_169: "f32[768]"; primals_170: "f32[768]"; primals_171: "f32[768]"; primals_172: "f32[384, 768]"; primals_173: "f32[384]"; primals_174: "f32[384, 768]"; primals_175: "f32[384]"; primals_176: "f32[384, 768]"; primals_177: "f32[384]"; primals_178: "f32[768, 1, 9]"; primals_179: "f32[384, 768, 1]"; primals_180: "f32[54, 384]"; primals_181: "f32[54]"; primals_182: "f32[384, 768]"; primals_183: "f32[384]"; primals_184: "f32[768, 768]"; primals_185: "f32[768]"; primals_186: "f32[768]"; primals_187: "f32[768]"; primals_188: "f32[3072, 768]"; primals_189: "f32[3072]"; primals_190: "f32[768, 3072]"; primals_191: "f32[768]"; primals_192: "f32[768]"; primals_193: "f32[768]"; primals_194: "f32[384, 768]"; primals_195: "f32[384]"; primals_196: "f32[384, 768]"; primals_197: "f32[384]"; primals_198: "f32[384, 768]"; primals_199: "f32[384]"; primals_200: "f32[768, 1, 9]"; primals_201: "f32[384, 768, 1]"; primals_202: "f32[54, 384]"; primals_203: "f32[54]"; primals_204: "f32[384, 768]"; primals_205: "f32[384]"; primals_206: "f32[768, 768]"; primals_207: "f32[768]"; primals_208: "f32[768]"; primals_209: "f32[768]"; primals_210: "f32[3072, 768]"; primals_211: "f32[3072]"; primals_212: "f32[768, 3072]"; primals_213: "f32[768]"; primals_214: "f32[768]"; primals_215: "f32[768]"; primals_216: "f32[384, 768]"; primals_217: "f32[384]"; primals_218: "f32[384, 768]"; primals_219: "f32[384]"; primals_220: "f32[384, 768]"; primals_221: "f32[384]"; primals_222: "f32[768, 1, 9]"; primals_223: "f32[384, 768, 1]"; primals_224: "f32[54, 384]"; primals_225: "f32[54]"; primals_226: "f32[384, 768]"; primals_227: "f32[384]"; primals_228: "f32[768, 768]"; primals_229: "f32[768]"; primals_230: "f32[768]"; primals_231: "f32[768]"; primals_232: "f32[3072, 768]"; primals_233: "f32[3072]"; primals_234: "f32[768, 3072]"; primals_235: "f32[768]"; primals_236: "f32[768]"; primals_237: "f32[768]"; primals_238: "f32[384, 768]"; primals_239: "f32[384]"; primals_240: "f32[384, 768]"; primals_241: "f32[384]"; primals_242: "f32[384, 768]"; primals_243: "f32[384]"; primals_244: "f32[768, 1, 9]"; primals_245: "f32[384, 768, 1]"; primals_246: "f32[54, 384]"; primals_247: "f32[54]"; primals_248: "f32[384, 768]"; primals_249: "f32[384]"; primals_250: "f32[768, 768]"; primals_251: "f32[768]"; primals_252: "f32[768]"; primals_253: "f32[768]"; primals_254: "f32[3072, 768]"; primals_255: "f32[3072]"; primals_256: "f32[768, 3072]"; primals_257: "f32[768]"; primals_258: "f32[768]"; primals_259: "f32[768]"; primals_260: "f32[384, 768]"; primals_261: "f32[384]"; primals_262: "f32[384, 768]"; primals_263: "f32[384]"; primals_264: "f32[384, 768]"; primals_265: "f32[384]"; primals_266: "f32[768, 1, 9]"; primals_267: "f32[384, 768, 1]"; primals_268: "f32[54, 384]"; primals_269: "f32[54]"; primals_270: "f32[384, 768]"; primals_271: "f32[384]"; primals_272: "f32[768, 768]"; primals_273: "f32[768]"; primals_274: "f32[768]"; primals_275: "f32[768]"; primals_276: "f32[3072, 768]"; primals_277: "f32[3072]"; primals_278: "f32[768, 3072]"; primals_279: "f32[768]"; primals_280: "f32[768]"; primals_281: "f32[768]"; primals_282: "f32[768, 768]"; primals_283: "f32[768]"; primals_284: "f32[768]"; primals_285: "f32[768]"; primals_286: "f32[30522, 768]"; primals_287: "f32[30522]"; primals_288: "i64[1, 512]"; primals_289: "i64[1, 512]"; primals_290: "i64[1, 512]"; primals_291: "i64[1, 512]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512, 30522]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:832, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:835, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_288, 0, 0, 9223372036854775807);  primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:836, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_2: "f32[1, 512]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(slice_2, 1);  slice_2 = None
    unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_3: "f32[1, 1, 1, 512]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, slice_3);  slice_3 = None
    mul: "f32[1, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:216, code: position_ids = self.position_ids[:, :seq_length]
    slice_4: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_289, 0, 0, 9223372036854775807);  primals_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:230, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_13, primals_290, 0);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:231, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_14, slice_4);  primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:232, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_15, expand);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:234, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    add_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:235, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1)
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_16);  mul_1 = None
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_17);  mul_2 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:236, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_3, 0.1, True);  add_3 = None
    getitem_2: "f32[1, 512, 768]" = native_dropout[0]
    getitem_3: "b8[1, 512, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute: "f32[768, 384]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_19, view, permute);  primals_19 = None
    view_1: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm, [1, 512, 384]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_2: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute_1: "f32[768, 384]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_1: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_21, view_2, permute_1);  primals_21 = None
    view_3: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_1, [1, 512, 384]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_4: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute_2: "f32[768, 384]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_2: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_23, view_4, permute_2);  primals_23 = None
    view_5: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_2, [1, 512, 384]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_3: "f32[1, 768, 512]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_3, primals_24, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_1: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution, primals_25, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_4: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_1, primals_1);  convolution_1 = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_6: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_1, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_7: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 6, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_8: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_5, [1, 512, 6, 64]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_7: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_8: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_4, [0, 2, 1]);  add_4 = None
    mul_3: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_8, view_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_9: "f32[384, 54]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    view_9: "f32[512, 384]" = torch.ops.aten.view.default(mul_3, [512, 384]);  mul_3 = None
    mm: "f32[512, 54]" = torch.ops.aten.mm.default(view_9, permute_9)
    view_10: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm, [1, 512, 54]);  mm = None
    add_5: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_10, primals_27);  view_10 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_11: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_5, [-1, 9, 1]);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_11, [1], True)
    sub_2: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_11, amax);  view_11 = amax = None
    exp: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
    div: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_12: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute_10: "f32[768, 384]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_3: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_29, view_12, permute_10);  primals_29 = None
    view_13: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_3, [1, 512, 384]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_14: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_13, [1, -1, 384]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_11: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    clone: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    unsqueeze_2: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone, -1);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_3: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    iota_1: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_4: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    add_6: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
    iota_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
    iota_3: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_6: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
    add_7: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None
    constant_pad_nd: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_2, [0, 0, 4, 4], 0.0);  unsqueeze_2 = None
    unsqueeze_7: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_6, -1);  add_6 = None
    unsqueeze_8: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, -1);  unsqueeze_7 = None
    slice_5: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 0, 0, 9223372036854775807);  constant_pad_nd = None
    slice_6: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    index: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_6, [None, None, unsqueeze_8, add_7]);  slice_6 = unsqueeze_8 = add_7 = None
    permute_12: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
    view_15: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_12, [1, 3456, 512]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_13: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    view_16: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_13, [1, 512, 384, 9]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_1: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_16, memory_format = torch.contiguous_format);  view_16 = None
    view_17: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_1, [3072, 64, 9]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_1: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_17, [3072, 64, 9]);  view_17 = None
    view_18: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_1, [3072, 64, 9]);  expand_1 = None
    expand_2: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div, [3072, 9, 1]);  div = None
    view_19: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_2, [3072, 9, 1]);  expand_2 = None
    bmm: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_18, view_19)
    view_20: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm, [3072, 64, 1]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_21: "f32[512, 384]" = torch.ops.aten.view.default(view_20, [-1, 384]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_14: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2]);  permute_6 = None
    expand_3: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 6, 512, 64]);  permute_5 = None
    view_22: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_3, [6, 512, 64]);  expand_3 = None
    expand_4: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_14, [1, 6, 64, 512]);  permute_14 = None
    view_23: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_4, [6, 64, 512]);  expand_4 = None
    bmm_1: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_22, view_23)
    view_24: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_1, [1, 6, 512, 512]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_1: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_24, 8.0);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_8: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_1, mul);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_8, [-1], True)
    sub_3: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_8, amax_1);  add_8 = amax_1 = None
    exp_1: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_2: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_2, 0.1, True);  div_2 = None
    getitem_4: "f32[1, 6, 512, 512]" = native_dropout_1[0]
    getitem_5: "b8[1, 6, 512, 512]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_5: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_4, [1, 6, 512, 512]);  getitem_4 = None
    view_25: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_5, [6, 512, 512]);  expand_5 = None
    expand_6: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_7, [1, 6, 512, 64]);  permute_7 = None
    view_26: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_6, [6, 512, 64]);  expand_6 = None
    bmm_2: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_25, view_26)
    view_27: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_2, [1, 6, 512, 64]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_15: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    clone_2: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_28: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_21, [1, -1, 6, 64]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_2, view_28], 2);  clone_2 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_29: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat, [1, 512, 768]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_30: "f32[512, 768]" = torch.ops.aten.view.default(view_29, [512, 768]);  view_29 = None
    permute_16: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    addmm_4: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_31, view_30, permute_16);  primals_31 = None
    view_31: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_4, [1, 512, 768]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_31, 0.1, True);  view_31 = None
    getitem_6: "f32[1, 512, 768]" = native_dropout_2[0]
    getitem_7: "b8[1, 512, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_6, getitem_2);  getitem_6 = getitem_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_9)
    mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = None
    mul_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, primals_32);  mul_4 = None
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, primals_33);  mul_5 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[512, 768]" = torch.ops.aten.view.default(add_11, [512, 768])
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_5: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_35, view_32, permute_17);  primals_35 = None
    view_33: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_5, [1, 512, 3072]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_12: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_12);  mul_6 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_34: "f32[512, 3072]" = torch.ops.aten.view.default(mul_8, [512, 3072]);  mul_8 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_37, view_34, permute_18);  primals_37 = None
    view_35: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_35, 0.1, True);  view_35 = None
    getitem_10: "f32[1, 512, 768]" = native_dropout_3[0]
    getitem_11: "b8[1, 512, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_10, add_11);  getitem_10 = add_11 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_13)
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_38);  mul_9 = None
    add_15: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_39);  mul_10 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_36: "f32[512, 768]" = torch.ops.aten.view.default(add_15, [512, 768])
    permute_19: "f32[768, 384]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_7: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_41, view_36, permute_19);  primals_41 = None
    view_37: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_7, [1, 512, 384]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.view.default(add_15, [512, 768])
    permute_20: "f32[768, 384]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_8: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_43, view_38, permute_20);  primals_43 = None
    view_39: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_8, [1, 512, 384]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_40: "f32[512, 768]" = torch.ops.aten.view.default(add_15, [512, 768])
    permute_21: "f32[768, 384]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_9: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_45, view_40, permute_21);  primals_45 = None
    view_41: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_9, [1, 512, 384]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_22: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_15, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_2: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_22, primals_46, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_3: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_2, primals_47, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_16: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_3, primals_2);  convolution_3 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_42: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_37, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_43: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_39, [1, 512, 6, 64]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_25: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_44: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_41, [1, 512, 6, 64]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_27: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_16, [0, 2, 1]);  add_16 = None
    mul_11: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_27, view_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_28: "f32[384, 54]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    view_45: "f32[512, 384]" = torch.ops.aten.view.default(mul_11, [512, 384]);  mul_11 = None
    mm_1: "f32[512, 54]" = torch.ops.aten.mm.default(view_45, permute_28)
    view_46: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_1, [1, 512, 54]);  mm_1 = None
    add_17: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_46, primals_49);  view_46 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_47: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_17, [-1, 9, 1]);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_2: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_47, [1], True)
    sub_6: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_47, amax_2);  view_47 = amax_2 = None
    exp_2: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_3: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
    div_3: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_48: "f32[512, 768]" = torch.ops.aten.view.default(add_15, [512, 768])
    permute_29: "f32[768, 384]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_10: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_51, view_48, permute_29);  primals_51 = None
    view_49: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_10, [1, 512, 384]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_50: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_49, [1, -1, 384]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_30: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    clone_3: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    unsqueeze_9: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_3, -1);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_4: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_10: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
    iota_5: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_11: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
    add_18: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_10, unsqueeze_11);  unsqueeze_10 = unsqueeze_11 = None
    iota_6: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_12: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_6, 0);  iota_6 = None
    iota_7: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_13: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
    add_19: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_12, unsqueeze_13);  unsqueeze_12 = unsqueeze_13 = None
    constant_pad_nd_1: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_9, [0, 0, 4, 4], 0.0);  unsqueeze_9 = None
    unsqueeze_14: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_18, -1);  add_18 = None
    unsqueeze_15: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    slice_7: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_1, 0, 0, 9223372036854775807);  constant_pad_nd_1 = None
    slice_8: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    index_1: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_8, [None, None, unsqueeze_15, add_19]);  slice_8 = unsqueeze_15 = add_19 = None
    permute_31: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
    view_51: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_31, [1, 3456, 512]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_32: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    view_52: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_32, [1, 512, 384, 9]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_4: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_52, memory_format = torch.contiguous_format);  view_52 = None
    view_53: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_4, [3072, 64, 9]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_7: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_53, [3072, 64, 9]);  view_53 = None
    view_54: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_7, [3072, 64, 9]);  expand_7 = None
    expand_8: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_3, [3072, 9, 1]);  div_3 = None
    view_55: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_8, [3072, 9, 1]);  expand_8 = None
    bmm_3: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_54, view_55)
    view_56: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_3, [3072, 64, 1]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_57: "f32[512, 384]" = torch.ops.aten.view.default(view_56, [-1, 384]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_33: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
    expand_9: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_24, [1, 6, 512, 64]);  permute_24 = None
    view_58: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_9, [6, 512, 64]);  expand_9 = None
    expand_10: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_33, [1, 6, 64, 512]);  permute_33 = None
    view_59: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_10, [6, 64, 512]);  expand_10 = None
    bmm_4: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_58, view_59)
    view_60: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 6, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_4: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_60, 8.0);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_20: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_20, [-1], True)
    sub_7: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_20, amax_3);  add_20 = amax_3 = None
    exp_3: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_4: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_5: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_4 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_14: "f32[1, 6, 512, 512]" = native_dropout_4[0]
    getitem_15: "b8[1, 6, 512, 512]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_14, [1, 6, 512, 512]);  getitem_14 = None
    view_61: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_11, [6, 512, 512]);  expand_11 = None
    expand_12: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_26, [1, 6, 512, 64]);  permute_26 = None
    view_62: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_12, [6, 512, 64]);  expand_12 = None
    bmm_5: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_61, view_62)
    view_63: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 6, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_34: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
    clone_5: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_64: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_57, [1, -1, 6, 64]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_1: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_5, view_64], 2);  clone_5 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_65: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_1, [1, 512, 768]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.view.default(view_65, [512, 768]);  view_65 = None
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_53, view_66, permute_35);  primals_53 = None
    view_67: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_67, 0.1, True);  view_67 = None
    getitem_16: "f32[1, 512, 768]" = native_dropout_5[0]
    getitem_17: "b8[1, 512, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_16, add_15);  getitem_16 = add_15 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_19)
    mul_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = None
    mul_13: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, primals_54);  mul_12 = None
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_13, primals_55);  mul_13 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_68: "f32[512, 768]" = torch.ops.aten.view.default(add_23, [512, 768])
    permute_36: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_12: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_57, view_68, permute_36);  primals_57 = None
    view_69: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_12, [1, 512, 3072]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_24);  mul_14 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_70: "f32[512, 3072]" = torch.ops.aten.view.default(mul_16, [512, 3072]);  mul_16 = None
    permute_37: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_59, view_70, permute_37);  primals_59 = None
    view_71: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_13, [1, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_71, 0.1, True);  view_71 = None
    getitem_20: "f32[1, 512, 768]" = native_dropout_6[0]
    getitem_21: "b8[1, 512, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_20, add_23);  getitem_20 = add_23 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_23)
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_60);  mul_17 = None
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_61);  mul_18 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_72: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768])
    permute_38: "f32[768, 384]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    addmm_14: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_63, view_72, permute_38);  primals_63 = None
    view_73: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_14, [1, 512, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_74: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768])
    permute_39: "f32[768, 384]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_15: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_65, view_74, permute_39);  primals_65 = None
    view_75: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_15, [1, 512, 384]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_76: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768])
    permute_40: "f32[768, 384]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_16: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_67, view_76, permute_40);  primals_67 = None
    view_77: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_16, [1, 512, 384]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_41: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_27, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_4: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_41, primals_68, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_5: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_4, primals_69, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_28: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_5, primals_3);  convolution_5 = primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_78: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_73, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_43: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_79: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_75, [1, 512, 6, 64]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_44: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_80: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_77, [1, 512, 6, 64]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_45: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_46: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_28, [0, 2, 1]);  add_28 = None
    mul_19: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_46, view_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_47: "f32[384, 54]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_81: "f32[512, 384]" = torch.ops.aten.view.default(mul_19, [512, 384]);  mul_19 = None
    mm_2: "f32[512, 54]" = torch.ops.aten.mm.default(view_81, permute_47)
    view_82: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_2, [1, 512, 54]);  mm_2 = None
    add_29: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_82, primals_71);  view_82 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_83: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_29, [-1, 9, 1]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_4: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_83, [1], True)
    sub_10: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_83, amax_4);  view_83 = amax_4 = None
    exp_4: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_5: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [1], True)
    div_6: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_84: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768])
    permute_48: "f32[768, 384]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_17: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_73, view_84, permute_48);  primals_73 = None
    view_85: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_17, [1, 512, 384]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_86: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_85, [1, -1, 384]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_49: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None
    clone_6: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    unsqueeze_16: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_6, -1);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_8: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_17: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_8, 0);  iota_8 = None
    iota_9: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_18: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
    add_30: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_17, unsqueeze_18);  unsqueeze_17 = unsqueeze_18 = None
    iota_10: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_19: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_10, 0);  iota_10 = None
    iota_11: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_20: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
    add_31: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_19, unsqueeze_20);  unsqueeze_19 = unsqueeze_20 = None
    constant_pad_nd_2: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_16, [0, 0, 4, 4], 0.0);  unsqueeze_16 = None
    unsqueeze_21: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_30, -1);  add_30 = None
    unsqueeze_22: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, -1);  unsqueeze_21 = None
    slice_9: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_2, 0, 0, 9223372036854775807);  constant_pad_nd_2 = None
    slice_10: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    index_2: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_10, [None, None, unsqueeze_22, add_31]);  slice_10 = unsqueeze_22 = add_31 = None
    permute_50: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_2, [0, 1, 2, 4, 3, 5]);  index_2 = None
    view_87: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_50, [1, 3456, 512]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_51: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
    view_88: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_51, [1, 512, 384, 9]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_7: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_88, memory_format = torch.contiguous_format);  view_88 = None
    view_89: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_7, [3072, 64, 9]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_13: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_89, [3072, 64, 9]);  view_89 = None
    view_90: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_13, [3072, 64, 9]);  expand_13 = None
    expand_14: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_6, [3072, 9, 1]);  div_6 = None
    view_91: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_14, [3072, 9, 1]);  expand_14 = None
    bmm_6: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_90, view_91)
    view_92: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_6, [3072, 64, 1]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_93: "f32[512, 384]" = torch.ops.aten.view.default(view_92, [-1, 384]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_52: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_44, [0, 1, 3, 2]);  permute_44 = None
    expand_15: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_43, [1, 6, 512, 64]);  permute_43 = None
    view_94: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_15, [6, 512, 64]);  expand_15 = None
    expand_16: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_52, [1, 6, 64, 512]);  permute_52 = None
    view_95: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_16, [6, 64, 512]);  expand_16 = None
    bmm_7: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_94, view_95)
    view_96: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_7, [1, 6, 512, 512]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_7: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_96, 8.0);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_32: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_7, mul);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_32, [-1], True)
    sub_11: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_32, amax_5);  add_32 = amax_5 = None
    exp_5: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_6: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_8: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_7 = torch.ops.aten.native_dropout.default(div_8, 0.1, True);  div_8 = None
    getitem_24: "f32[1, 6, 512, 512]" = native_dropout_7[0]
    getitem_25: "b8[1, 6, 512, 512]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_17: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_24, [1, 6, 512, 512]);  getitem_24 = None
    view_97: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_17, [6, 512, 512]);  expand_17 = None
    expand_18: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_45, [1, 6, 512, 64]);  permute_45 = None
    view_98: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_18, [6, 512, 64]);  expand_18 = None
    bmm_8: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_8, [1, 6, 512, 64]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_53: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    clone_8: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_100: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_93, [1, -1, 6, 64]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_2: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_8, view_100], 2);  clone_8 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_101: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_2, [1, 512, 768]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_102: "f32[512, 768]" = torch.ops.aten.view.default(view_101, [512, 768]);  view_101 = None
    permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_75, view_102, permute_54);  primals_75 = None
    view_103: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_103, 0.1, True);  view_103 = None
    getitem_26: "f32[1, 512, 768]" = native_dropout_8[0]
    getitem_27: "b8[1, 512, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_26, add_27);  getitem_26 = add_27 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_29)
    mul_20: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_5);  sub_12 = None
    mul_21: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, primals_76);  mul_20 = None
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_77);  mul_21 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.view.default(add_35, [512, 768])
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    addmm_19: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_79, view_104, permute_55);  primals_79 = None
    view_105: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_19, [1, 512, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.5)
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476)
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_36: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_36);  mul_22 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 3072]" = torch.ops.aten.view.default(mul_24, [512, 3072]);  mul_24 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_81, view_106, permute_56);  primals_81 = None
    view_107: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_107, 0.1, True);  view_107 = None
    getitem_30: "f32[1, 512, 768]" = native_dropout_9[0]
    getitem_31: "b8[1, 512, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_30, add_35);  getitem_30 = add_35 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_33)
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_6);  sub_13 = None
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_82);  mul_25 = None
    add_39: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_83);  mul_26 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_108: "f32[512, 768]" = torch.ops.aten.view.default(add_39, [512, 768])
    permute_57: "f32[768, 384]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    addmm_21: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_85, view_108, permute_57);  primals_85 = None
    view_109: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_21, [1, 512, 384]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_110: "f32[512, 768]" = torch.ops.aten.view.default(add_39, [512, 768])
    permute_58: "f32[768, 384]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_22: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_87, view_110, permute_58);  primals_87 = None
    view_111: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_22, [1, 512, 384]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_112: "f32[512, 768]" = torch.ops.aten.view.default(add_39, [512, 768])
    permute_59: "f32[768, 384]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_23: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_89, view_112, permute_59);  primals_89 = None
    view_113: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_23, [1, 512, 384]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_60: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_39, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_6: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_60, primals_90, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_7: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_6, primals_91, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_40: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_7, primals_4);  convolution_7 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_114: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_109, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_62: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_115: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_111, [1, 512, 6, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_63: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_116: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 6, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_64: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_65: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    mul_27: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_65, view_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_66: "f32[384, 54]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    view_117: "f32[512, 384]" = torch.ops.aten.view.default(mul_27, [512, 384]);  mul_27 = None
    mm_3: "f32[512, 54]" = torch.ops.aten.mm.default(view_117, permute_66)
    view_118: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_3, [1, 512, 54]);  mm_3 = None
    add_41: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_118, primals_93);  view_118 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_119: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_41, [-1, 9, 1]);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_6: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_119, [1], True)
    sub_14: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_119, amax_6);  view_119 = amax_6 = None
    exp_6: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_7: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True)
    div_9: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_120: "f32[512, 768]" = torch.ops.aten.view.default(add_39, [512, 768])
    permute_67: "f32[768, 384]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    addmm_24: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_95, view_120, permute_67);  primals_95 = None
    view_121: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_24, [1, 512, 384]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_122: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_121, [1, -1, 384]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_68: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    clone_9: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    unsqueeze_23: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_9, -1);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_12: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_24: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_12, 0);  iota_12 = None
    iota_13: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_25: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
    add_42: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_24, unsqueeze_25);  unsqueeze_24 = unsqueeze_25 = None
    iota_14: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_26: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_14, 0);  iota_14 = None
    iota_15: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_27: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
    add_43: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_26, unsqueeze_27);  unsqueeze_26 = unsqueeze_27 = None
    constant_pad_nd_3: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_23, [0, 0, 4, 4], 0.0);  unsqueeze_23 = None
    unsqueeze_28: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_42, -1);  add_42 = None
    unsqueeze_29: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    slice_11: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_3, 0, 0, 9223372036854775807);  constant_pad_nd_3 = None
    slice_12: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_11, 1, 0, 9223372036854775807);  slice_11 = None
    index_3: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_12, [None, None, unsqueeze_29, add_43]);  slice_12 = unsqueeze_29 = add_43 = None
    permute_69: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_3, [0, 1, 2, 4, 3, 5]);  index_3 = None
    view_123: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_69, [1, 3456, 512]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_70: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    view_124: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_70, [1, 512, 384, 9]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_10: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_124, memory_format = torch.contiguous_format);  view_124 = None
    view_125: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_10, [3072, 64, 9]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_19: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_125, [3072, 64, 9]);  view_125 = None
    view_126: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_19, [3072, 64, 9]);  expand_19 = None
    expand_20: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_9, [3072, 9, 1]);  div_9 = None
    view_127: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_20, [3072, 9, 1]);  expand_20 = None
    bmm_9: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_126, view_127)
    view_128: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_9, [3072, 64, 1]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_129: "f32[512, 384]" = torch.ops.aten.view.default(view_128, [-1, 384]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_71: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_63, [0, 1, 3, 2]);  permute_63 = None
    expand_21: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_62, [1, 6, 512, 64]);  permute_62 = None
    view_130: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_21, [6, 512, 64]);  expand_21 = None
    expand_22: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_71, [1, 6, 64, 512]);  permute_71 = None
    view_131: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_22, [6, 64, 512]);  expand_22 = None
    bmm_10: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_130, view_131)
    view_132: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 6, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_10: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_132, 8.0);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_44: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_44, [-1], True)
    sub_15: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_44, amax_7);  add_44 = amax_7 = None
    exp_7: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_8: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_11: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_10 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_34: "f32[1, 6, 512, 512]" = native_dropout_10[0]
    getitem_35: "b8[1, 6, 512, 512]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_34, [1, 6, 512, 512]);  getitem_34 = None
    view_133: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_23, [6, 512, 512]);  expand_23 = None
    expand_24: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_64, [1, 6, 512, 64]);  permute_64 = None
    view_134: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_24, [6, 512, 64]);  expand_24 = None
    bmm_11: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_133, view_134)
    view_135: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 6, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_72: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
    clone_11: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_136: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_129, [1, -1, 6, 64]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_3: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_11, view_136], 2);  clone_11 = view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_137: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_3, [1, 512, 768]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_138: "f32[512, 768]" = torch.ops.aten.view.default(view_137, [512, 768]);  view_137 = None
    permute_73: "f32[768, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_25: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_97, view_138, permute_73);  primals_97 = None
    view_139: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_25, [1, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_139, 0.1, True);  view_139 = None
    getitem_36: "f32[1, 512, 768]" = native_dropout_11[0]
    getitem_37: "b8[1, 512, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_36, add_39);  getitem_36 = add_39 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_39)
    mul_28: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_7);  sub_16 = None
    mul_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_98);  mul_28 = None
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_99);  mul_29 = primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_140: "f32[512, 768]" = torch.ops.aten.view.default(add_47, [512, 768])
    permute_74: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_26: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_101, view_140, permute_74);  primals_101 = None
    view_141: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_26, [1, 512, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_30, add_48);  mul_30 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_142: "f32[512, 3072]" = torch.ops.aten.view.default(mul_32, [512, 3072]);  mul_32 = None
    permute_75: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_103, view_142, permute_75);  primals_103 = None
    view_143: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_143, 0.1, True);  view_143 = None
    getitem_40: "f32[1, 512, 768]" = native_dropout_12[0]
    getitem_41: "b8[1, 512, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_40, add_47);  getitem_40 = add_47 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_43)
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_8);  sub_17 = None
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_104);  mul_33 = None
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_105);  mul_34 = primals_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_144: "f32[512, 768]" = torch.ops.aten.view.default(add_51, [512, 768])
    permute_76: "f32[768, 384]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_28: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_107, view_144, permute_76);  primals_107 = None
    view_145: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_28, [1, 512, 384]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_146: "f32[512, 768]" = torch.ops.aten.view.default(add_51, [512, 768])
    permute_77: "f32[768, 384]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_29: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_109, view_146, permute_77);  primals_109 = None
    view_147: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_29, [1, 512, 384]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_148: "f32[512, 768]" = torch.ops.aten.view.default(add_51, [512, 768])
    permute_78: "f32[768, 384]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_30: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_111, view_148, permute_78);  primals_111 = None
    view_149: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_30, [1, 512, 384]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_79: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_51, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_8: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_79, primals_112, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_9: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_8, primals_113, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_52: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_9, primals_5);  convolution_9 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_150: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_145, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_151: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_147, [1, 512, 6, 64]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_152: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_149, [1, 512, 6, 64]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_84: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_52, [0, 2, 1]);  add_52 = None
    mul_35: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_84, view_145)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_85: "f32[384, 54]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_153: "f32[512, 384]" = torch.ops.aten.view.default(mul_35, [512, 384]);  mul_35 = None
    mm_4: "f32[512, 54]" = torch.ops.aten.mm.default(view_153, permute_85)
    view_154: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_4, [1, 512, 54]);  mm_4 = None
    add_53: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_154, primals_115);  view_154 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_155: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_53, [-1, 9, 1]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_8: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_155, [1], True)
    sub_18: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_155, amax_8);  view_155 = amax_8 = None
    exp_8: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_9: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True)
    div_12: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_156: "f32[512, 768]" = torch.ops.aten.view.default(add_51, [512, 768])
    permute_86: "f32[768, 384]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    addmm_31: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_117, view_156, permute_86);  primals_117 = None
    view_157: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_31, [1, 512, 384]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_158: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_157, [1, -1, 384]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_87: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    clone_12: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    unsqueeze_30: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_12, -1);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_16: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_31: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_16, 0);  iota_16 = None
    iota_17: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_32: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
    add_54: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_31, unsqueeze_32);  unsqueeze_31 = unsqueeze_32 = None
    iota_18: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_33: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_18, 0);  iota_18 = None
    iota_19: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_34: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
    add_55: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_33, unsqueeze_34);  unsqueeze_33 = unsqueeze_34 = None
    constant_pad_nd_4: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_30, [0, 0, 4, 4], 0.0);  unsqueeze_30 = None
    unsqueeze_35: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_54, -1);  add_54 = None
    unsqueeze_36: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, -1);  unsqueeze_35 = None
    slice_13: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_4, 0, 0, 9223372036854775807);  constant_pad_nd_4 = None
    slice_14: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    index_4: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_14, [None, None, unsqueeze_36, add_55]);  slice_14 = unsqueeze_36 = add_55 = None
    permute_88: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
    view_159: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_88, [1, 3456, 512]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_89: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
    view_160: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_89, [1, 512, 384, 9]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_13: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_160, memory_format = torch.contiguous_format);  view_160 = None
    view_161: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_13, [3072, 64, 9]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_25: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_161, [3072, 64, 9]);  view_161 = None
    view_162: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_25, [3072, 64, 9]);  expand_25 = None
    expand_26: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_12, [3072, 9, 1]);  div_12 = None
    view_163: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_26, [3072, 9, 1]);  expand_26 = None
    bmm_12: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_162, view_163)
    view_164: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_12, [3072, 64, 1]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_165: "f32[512, 384]" = torch.ops.aten.view.default(view_164, [-1, 384]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_90: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_82, [0, 1, 3, 2]);  permute_82 = None
    expand_27: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_81, [1, 6, 512, 64]);  permute_81 = None
    view_166: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_27, [6, 512, 64]);  expand_27 = None
    expand_28: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_90, [1, 6, 64, 512]);  permute_90 = None
    view_167: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_28, [6, 64, 512]);  expand_28 = None
    bmm_13: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_13, [1, 6, 512, 512]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_13: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_168, 8.0);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_56: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_13, mul);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_56, [-1], True)
    sub_19: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_56, amax_9);  add_56 = amax_9 = None
    exp_9: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_10: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_14: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_14, 0.1, True);  div_14 = None
    getitem_44: "f32[1, 6, 512, 512]" = native_dropout_13[0]
    getitem_45: "b8[1, 6, 512, 512]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_29: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_44, [1, 6, 512, 512]);  getitem_44 = None
    view_169: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_29, [6, 512, 512]);  expand_29 = None
    expand_30: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_83, [1, 6, 512, 64]);  permute_83 = None
    view_170: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_30, [6, 512, 64]);  expand_30 = None
    bmm_14: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_169, view_170)
    view_171: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_14, [1, 6, 512, 64]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_91: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    clone_14: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_172: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_165, [1, -1, 6, 64]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_4: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_14, view_172], 2);  clone_14 = view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_173: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_4, [1, 512, 768]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 768]" = torch.ops.aten.view.default(view_173, [512, 768]);  view_173 = None
    permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_119, view_174, permute_92);  primals_119 = None
    view_175: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_32, [1, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_175, 0.1, True);  view_175 = None
    getitem_46: "f32[1, 512, 768]" = native_dropout_14[0]
    getitem_47: "b8[1, 512, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_46, add_51);  getitem_46 = add_51 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_49)
    mul_36: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_9);  sub_20 = None
    mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, primals_120);  mul_36 = None
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, primals_121);  mul_37 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.view.default(add_59, [512, 768])
    permute_93: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_33: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_123, view_176, permute_93);  primals_123 = None
    view_177: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_33, [1, 512, 3072]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.5)
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476)
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_60: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_60);  mul_38 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_178: "f32[512, 3072]" = torch.ops.aten.view.default(mul_40, [512, 3072]);  mul_40 = None
    permute_94: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_34: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_125, view_178, permute_94);  primals_125 = None
    view_179: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_34, [1, 512, 768]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_179, 0.1, True);  view_179 = None
    getitem_50: "f32[1, 512, 768]" = native_dropout_15[0]
    getitem_51: "b8[1, 512, 768]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_50, add_59);  getitem_50 = add_59 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_61, getitem_53)
    mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_10);  sub_21 = None
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, primals_126);  mul_41 = None
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, primals_127);  mul_42 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_180: "f32[512, 768]" = torch.ops.aten.view.default(add_63, [512, 768])
    permute_95: "f32[768, 384]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_35: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_129, view_180, permute_95);  primals_129 = None
    view_181: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_35, [1, 512, 384]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_182: "f32[512, 768]" = torch.ops.aten.view.default(add_63, [512, 768])
    permute_96: "f32[768, 384]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_36: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_131, view_182, permute_96);  primals_131 = None
    view_183: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_36, [1, 512, 384]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_184: "f32[512, 768]" = torch.ops.aten.view.default(add_63, [512, 768])
    permute_97: "f32[768, 384]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_37: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_133, view_184, permute_97);  primals_133 = None
    view_185: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_37, [1, 512, 384]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_98: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_63, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_10: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_98, primals_134, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_11: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_10, primals_135, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_64: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_11, primals_6);  convolution_11 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_186: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_181, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_100: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_187: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_183, [1, 512, 6, 64]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_188: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_185, [1, 512, 6, 64]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_102: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_103: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None
    mul_43: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_103, view_181)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_104: "f32[384, 54]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    view_189: "f32[512, 384]" = torch.ops.aten.view.default(mul_43, [512, 384]);  mul_43 = None
    mm_5: "f32[512, 54]" = torch.ops.aten.mm.default(view_189, permute_104)
    view_190: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_5, [1, 512, 54]);  mm_5 = None
    add_65: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_190, primals_137);  view_190 = primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_191: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_65, [-1, 9, 1]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_10: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_191, [1], True)
    sub_22: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_191, amax_10);  view_191 = amax_10 = None
    exp_10: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_11: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [1], True)
    div_15: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_192: "f32[512, 768]" = torch.ops.aten.view.default(add_63, [512, 768])
    permute_105: "f32[768, 384]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_38: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_139, view_192, permute_105);  primals_139 = None
    view_193: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_38, [1, 512, 384]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_194: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_193, [1, -1, 384]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_106: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_194, [0, 2, 1]);  view_194 = None
    clone_15: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    unsqueeze_37: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_15, -1);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_20: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_38: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_20, 0);  iota_20 = None
    iota_21: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_39: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
    add_66: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_38, unsqueeze_39);  unsqueeze_38 = unsqueeze_39 = None
    iota_22: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_40: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_22, 0);  iota_22 = None
    iota_23: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_41: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
    add_67: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_40, unsqueeze_41);  unsqueeze_40 = unsqueeze_41 = None
    constant_pad_nd_5: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_37, [0, 0, 4, 4], 0.0);  unsqueeze_37 = None
    unsqueeze_42: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_66, -1);  add_66 = None
    unsqueeze_43: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    slice_15: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_5, 0, 0, 9223372036854775807);  constant_pad_nd_5 = None
    slice_16: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_15, 1, 0, 9223372036854775807);  slice_15 = None
    index_5: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_16, [None, None, unsqueeze_43, add_67]);  slice_16 = unsqueeze_43 = add_67 = None
    permute_107: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
    view_195: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_107, [1, 3456, 512]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_108: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
    view_196: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_108, [1, 512, 384, 9]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_16: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_196, memory_format = torch.contiguous_format);  view_196 = None
    view_197: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_16, [3072, 64, 9]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_31: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_197, [3072, 64, 9]);  view_197 = None
    view_198: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_31, [3072, 64, 9]);  expand_31 = None
    expand_32: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_15, [3072, 9, 1]);  div_15 = None
    view_199: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_32, [3072, 9, 1]);  expand_32 = None
    bmm_15: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_198, view_199)
    view_200: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_15, [3072, 64, 1]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_201: "f32[512, 384]" = torch.ops.aten.view.default(view_200, [-1, 384]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_109: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
    expand_33: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_100, [1, 6, 512, 64]);  permute_100 = None
    view_202: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_33, [6, 512, 64]);  expand_33 = None
    expand_34: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_109, [1, 6, 64, 512]);  permute_109 = None
    view_203: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_34, [6, 64, 512]);  expand_34 = None
    bmm_16: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_202, view_203)
    view_204: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_16, [1, 6, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_16: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_204, 8.0);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_68: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_68, [-1], True)
    sub_23: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_68, amax_11);  add_68 = amax_11 = None
    exp_11: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_12: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_17: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_16 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_54: "f32[1, 6, 512, 512]" = native_dropout_16[0]
    getitem_55: "b8[1, 6, 512, 512]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_54, [1, 6, 512, 512]);  getitem_54 = None
    view_205: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_35, [6, 512, 512]);  expand_35 = None
    expand_36: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_102, [1, 6, 512, 64]);  permute_102 = None
    view_206: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_36, [6, 512, 64]);  expand_36 = None
    bmm_17: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_205, view_206)
    view_207: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_17, [1, 6, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_110: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_17: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_208: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_201, [1, -1, 6, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_5: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_17, view_208], 2);  clone_17 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_209: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_5, [1, 512, 768]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_210: "f32[512, 768]" = torch.ops.aten.view.default(view_209, [512, 768]);  view_209 = None
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_141, view_210, permute_111);  primals_141 = None
    view_211: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_39, [1, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_211, 0.1, True);  view_211 = None
    getitem_56: "f32[1, 512, 768]" = native_dropout_17[0]
    getitem_57: "b8[1, 512, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_56, add_63);  getitem_56 = add_63 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_59)
    mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_11);  sub_24 = None
    mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_142);  mul_44 = None
    add_71: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_143);  mul_45 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_212: "f32[512, 768]" = torch.ops.aten.view.default(add_71, [512, 768])
    permute_112: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_40: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_145, view_212, permute_112);  primals_145 = None
    view_213: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 512, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_72);  mul_46 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 3072]" = torch.ops.aten.view.default(mul_48, [512, 3072]);  mul_48 = None
    permute_113: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_147, view_214, permute_113);  primals_147 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_41, [1, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_215, 0.1, True);  view_215 = None
    getitem_60: "f32[1, 512, 768]" = native_dropout_18[0]
    getitem_61: "b8[1, 512, 768]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_60, add_71);  getitem_60 = add_71 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_63)
    mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_12);  sub_25 = None
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_148);  mul_49 = None
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_149);  mul_50 = primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_216: "f32[512, 768]" = torch.ops.aten.view.default(add_75, [512, 768])
    permute_114: "f32[768, 384]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_42: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_151, view_216, permute_114);  primals_151 = None
    view_217: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_42, [1, 512, 384]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_218: "f32[512, 768]" = torch.ops.aten.view.default(add_75, [512, 768])
    permute_115: "f32[768, 384]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_43: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_153, view_218, permute_115);  primals_153 = None
    view_219: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_43, [1, 512, 384]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_220: "f32[512, 768]" = torch.ops.aten.view.default(add_75, [512, 768])
    permute_116: "f32[768, 384]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_44: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_155, view_220, permute_116);  primals_155 = None
    view_221: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_44, [1, 512, 384]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_117: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_75, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_12: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_117, primals_156, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_13: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_12, primals_157, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_76: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_13, primals_7);  convolution_13 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_222: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_217, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_119: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_223: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_219, [1, 512, 6, 64]);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_120: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_224: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_221, [1, 512, 6, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_121: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_122: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_76, [0, 2, 1]);  add_76 = None
    mul_51: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_122, view_217)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_123: "f32[384, 54]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    view_225: "f32[512, 384]" = torch.ops.aten.view.default(mul_51, [512, 384]);  mul_51 = None
    mm_6: "f32[512, 54]" = torch.ops.aten.mm.default(view_225, permute_123)
    view_226: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_6, [1, 512, 54]);  mm_6 = None
    add_77: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_226, primals_159);  view_226 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_227: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_77, [-1, 9, 1]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_12: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_227, [1], True)
    sub_26: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_227, amax_12);  view_227 = amax_12 = None
    exp_12: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_13: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True)
    div_18: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_228: "f32[512, 768]" = torch.ops.aten.view.default(add_75, [512, 768])
    permute_124: "f32[768, 384]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_45: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_161, view_228, permute_124);  primals_161 = None
    view_229: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_45, [1, 512, 384]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_230: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_229, [1, -1, 384]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_125: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    clone_18: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    unsqueeze_44: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_18, -1);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_24: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_45: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_24, 0);  iota_24 = None
    iota_25: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_46: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
    add_78: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_45, unsqueeze_46);  unsqueeze_45 = unsqueeze_46 = None
    iota_26: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_47: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_26, 0);  iota_26 = None
    iota_27: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_48: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
    add_79: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_47, unsqueeze_48);  unsqueeze_47 = unsqueeze_48 = None
    constant_pad_nd_6: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_44, [0, 0, 4, 4], 0.0);  unsqueeze_44 = None
    unsqueeze_49: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_78, -1);  add_78 = None
    unsqueeze_50: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, -1);  unsqueeze_49 = None
    slice_17: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_6, 0, 0, 9223372036854775807);  constant_pad_nd_6 = None
    slice_18: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    index_6: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_18, [None, None, unsqueeze_50, add_79]);  slice_18 = unsqueeze_50 = add_79 = None
    permute_126: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
    view_231: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_126, [1, 3456, 512]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_127: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    view_232: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_127, [1, 512, 384, 9]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_19: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_232, memory_format = torch.contiguous_format);  view_232 = None
    view_233: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_19, [3072, 64, 9]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_37: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_233, [3072, 64, 9]);  view_233 = None
    view_234: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_37, [3072, 64, 9]);  expand_37 = None
    expand_38: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_18, [3072, 9, 1]);  div_18 = None
    view_235: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_38, [3072, 9, 1]);  expand_38 = None
    bmm_18: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_234, view_235)
    view_236: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_18, [3072, 64, 1]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_237: "f32[512, 384]" = torch.ops.aten.view.default(view_236, [-1, 384]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_128: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2]);  permute_120 = None
    expand_39: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_119, [1, 6, 512, 64]);  permute_119 = None
    view_238: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_39, [6, 512, 64]);  expand_39 = None
    expand_40: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_128, [1, 6, 64, 512]);  permute_128 = None
    view_239: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_40, [6, 64, 512]);  expand_40 = None
    bmm_19: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_238, view_239)
    view_240: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_19, [1, 6, 512, 512]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_19: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_240, 8.0);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_80: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_19, mul);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_13: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_80, [-1], True)
    sub_27: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_80, amax_13);  add_80 = amax_13 = None
    exp_13: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_14: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_20: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_19 = torch.ops.aten.native_dropout.default(div_20, 0.1, True);  div_20 = None
    getitem_64: "f32[1, 6, 512, 512]" = native_dropout_19[0]
    getitem_65: "b8[1, 6, 512, 512]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_41: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_64, [1, 6, 512, 512]);  getitem_64 = None
    view_241: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_41, [6, 512, 512]);  expand_41 = None
    expand_42: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_121, [1, 6, 512, 64]);  permute_121 = None
    view_242: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_42, [6, 512, 64]);  expand_42 = None
    bmm_20: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_241, view_242)
    view_243: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_20, [1, 6, 512, 64]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_129: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
    clone_20: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_244: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_237, [1, -1, 6, 64]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_6: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_20, view_244], 2);  clone_20 = view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_245: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_6, [1, 512, 768]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_246: "f32[512, 768]" = torch.ops.aten.view.default(view_245, [512, 768]);  view_245 = None
    permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_46: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_163, view_246, permute_130);  primals_163 = None
    view_247: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_46, [1, 512, 768]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_247, 0.1, True);  view_247 = None
    getitem_66: "f32[1, 512, 768]" = native_dropout_20[0]
    getitem_67: "b8[1, 512, 768]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_66, add_75);  getitem_66 = add_75 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_28: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_69)
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_13);  sub_28 = None
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, primals_164);  mul_52 = None
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, primals_165);  mul_53 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_248: "f32[512, 768]" = torch.ops.aten.view.default(add_83, [512, 768])
    permute_131: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_47: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_167, view_248, permute_131);  primals_167 = None
    view_249: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_47, [1, 512, 3072]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, 0.5)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476)
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_84: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_56: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_84);  mul_54 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_250: "f32[512, 3072]" = torch.ops.aten.view.default(mul_56, [512, 3072]);  mul_56 = None
    permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_48: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_169, view_250, permute_132);  primals_169 = None
    view_251: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_48, [1, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_251, 0.1, True);  view_251 = None
    getitem_70: "f32[1, 512, 768]" = native_dropout_21[0]
    getitem_71: "b8[1, 512, 768]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_70, add_83);  getitem_70 = add_83 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_86: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_73)
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_14);  sub_29 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_170);  mul_57 = None
    add_87: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_171);  mul_58 = primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_252: "f32[512, 768]" = torch.ops.aten.view.default(add_87, [512, 768])
    permute_133: "f32[768, 384]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_49: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_173, view_252, permute_133);  primals_173 = None
    view_253: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_49, [1, 512, 384]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_254: "f32[512, 768]" = torch.ops.aten.view.default(add_87, [512, 768])
    permute_134: "f32[768, 384]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_50: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_175, view_254, permute_134);  primals_175 = None
    view_255: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_50, [1, 512, 384]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_256: "f32[512, 768]" = torch.ops.aten.view.default(add_87, [512, 768])
    permute_135: "f32[768, 384]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_51: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_177, view_256, permute_135);  primals_177 = None
    view_257: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_51, [1, 512, 384]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_136: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_87, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_14: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_136, primals_178, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_15: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_14, primals_179, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_88: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_15, primals_8);  convolution_15 = primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_258: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_253, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_138: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_259: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_255, [1, 512, 6, 64]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_139: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_260: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_257, [1, 512, 6, 64]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_140: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_141: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_88, [0, 2, 1]);  add_88 = None
    mul_59: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_141, view_253)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_142: "f32[384, 54]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    view_261: "f32[512, 384]" = torch.ops.aten.view.default(mul_59, [512, 384]);  mul_59 = None
    mm_7: "f32[512, 54]" = torch.ops.aten.mm.default(view_261, permute_142)
    view_262: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_7, [1, 512, 54]);  mm_7 = None
    add_89: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_262, primals_181);  view_262 = primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_263: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_89, [-1, 9, 1]);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_14: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_263, [1], True)
    sub_30: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_263, amax_14);  view_263 = amax_14 = None
    exp_14: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_15: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [1], True)
    div_21: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_14: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_264: "f32[512, 768]" = torch.ops.aten.view.default(add_87, [512, 768])
    permute_143: "f32[768, 384]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_52: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_183, view_264, permute_143);  primals_183 = None
    view_265: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_52, [1, 512, 384]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_266: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_265, [1, -1, 384]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_144: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
    clone_21: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    unsqueeze_51: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_21, -1);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_28: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_52: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_28, 0);  iota_28 = None
    iota_29: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_53: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
    add_90: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_52, unsqueeze_53);  unsqueeze_52 = unsqueeze_53 = None
    iota_30: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_54: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_30, 0);  iota_30 = None
    iota_31: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_55: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
    add_91: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_54, unsqueeze_55);  unsqueeze_54 = unsqueeze_55 = None
    constant_pad_nd_7: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_51, [0, 0, 4, 4], 0.0);  unsqueeze_51 = None
    unsqueeze_56: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_90, -1);  add_90 = None
    unsqueeze_57: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    slice_19: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_7, 0, 0, 9223372036854775807);  constant_pad_nd_7 = None
    slice_20: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_19, 1, 0, 9223372036854775807);  slice_19 = None
    index_7: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_20, [None, None, unsqueeze_57, add_91]);  slice_20 = unsqueeze_57 = add_91 = None
    permute_145: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
    view_267: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_145, [1, 3456, 512]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_146: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
    view_268: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_146, [1, 512, 384, 9]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_22: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_268, memory_format = torch.contiguous_format);  view_268 = None
    view_269: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_22, [3072, 64, 9]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_43: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_269, [3072, 64, 9]);  view_269 = None
    view_270: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_43, [3072, 64, 9]);  expand_43 = None
    expand_44: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_21, [3072, 9, 1]);  div_21 = None
    view_271: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_44, [3072, 9, 1]);  expand_44 = None
    bmm_21: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_270, view_271)
    view_272: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_21, [3072, 64, 1]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_273: "f32[512, 384]" = torch.ops.aten.view.default(view_272, [-1, 384]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_147: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_139, [0, 1, 3, 2]);  permute_139 = None
    expand_45: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_138, [1, 6, 512, 64]);  permute_138 = None
    view_274: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_45, [6, 512, 64]);  expand_45 = None
    expand_46: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_147, [1, 6, 64, 512]);  permute_147 = None
    view_275: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_46, [6, 64, 512]);  expand_46 = None
    bmm_22: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_274, view_275)
    view_276: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_22, [1, 6, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_22: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_276, 8.0);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_92: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_15: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_92, [-1], True)
    sub_31: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_92, amax_15);  add_92 = amax_15 = None
    exp_15: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_16: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_23: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_15: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_22 = torch.ops.aten.native_dropout.default(div_23, 0.1, True);  div_23 = None
    getitem_74: "f32[1, 6, 512, 512]" = native_dropout_22[0]
    getitem_75: "b8[1, 6, 512, 512]" = native_dropout_22[1];  native_dropout_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_74, [1, 6, 512, 512]);  getitem_74 = None
    view_277: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_47, [6, 512, 512]);  expand_47 = None
    expand_48: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_140, [1, 6, 512, 64]);  permute_140 = None
    view_278: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_48, [6, 512, 64]);  expand_48 = None
    bmm_23: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_277, view_278)
    view_279: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 6, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_148: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
    clone_23: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_280: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_273, [1, -1, 6, 64]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_7: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_23, view_280], 2);  clone_23 = view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_281: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_7, [1, 512, 768]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_282: "f32[512, 768]" = torch.ops.aten.view.default(view_281, [512, 768]);  view_281 = None
    permute_149: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_185, view_282, permute_149);  primals_185 = None
    view_283: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_53, [1, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_283, 0.1, True);  view_283 = None
    getitem_76: "f32[1, 512, 768]" = native_dropout_23[0]
    getitem_77: "b8[1, 512, 768]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_93: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_76, add_87);  getitem_76 = add_87 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_93, getitem_79)
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_15);  sub_32 = None
    mul_61: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_186);  mul_60 = None
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_187);  mul_61 = primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_284: "f32[512, 768]" = torch.ops.aten.view.default(add_95, [512, 768])
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_54: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_189, view_284, permute_150);  primals_189 = None
    view_285: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_54, [1, 512, 3072]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.5)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476)
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_64: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_96);  mul_62 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_286: "f32[512, 3072]" = torch.ops.aten.view.default(mul_64, [512, 3072]);  mul_64 = None
    permute_151: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_55: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_191, view_286, permute_151);  primals_191 = None
    view_287: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_55, [1, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_287, 0.1, True);  view_287 = None
    getitem_80: "f32[1, 512, 768]" = native_dropout_24[0]
    getitem_81: "b8[1, 512, 768]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_80, add_95);  getitem_80 = add_95 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_83)
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_16);  sub_33 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_192);  mul_65 = None
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_193);  mul_66 = primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_288: "f32[512, 768]" = torch.ops.aten.view.default(add_99, [512, 768])
    permute_152: "f32[768, 384]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_56: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_195, view_288, permute_152);  primals_195 = None
    view_289: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_56, [1, 512, 384]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_290: "f32[512, 768]" = torch.ops.aten.view.default(add_99, [512, 768])
    permute_153: "f32[768, 384]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_57: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_197, view_290, permute_153);  primals_197 = None
    view_291: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_57, [1, 512, 384]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_292: "f32[512, 768]" = torch.ops.aten.view.default(add_99, [512, 768])
    permute_154: "f32[768, 384]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_58: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_199, view_292, permute_154);  primals_199 = None
    view_293: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_58, [1, 512, 384]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_155: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_99, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_16: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_155, primals_200, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_17: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_16, primals_201, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_100: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_17, primals_9);  convolution_17 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_294: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_289, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_157: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_295: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_291, [1, 512, 6, 64]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_158: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_296: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_293, [1, 512, 6, 64]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_159: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_160: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None
    mul_67: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_160, view_289)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_161: "f32[384, 54]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    view_297: "f32[512, 384]" = torch.ops.aten.view.default(mul_67, [512, 384]);  mul_67 = None
    mm_8: "f32[512, 54]" = torch.ops.aten.mm.default(view_297, permute_161)
    view_298: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_8, [1, 512, 54]);  mm_8 = None
    add_101: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_298, primals_203);  view_298 = primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_299: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_101, [-1, 9, 1]);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_16: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_299, [1], True)
    sub_34: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_299, amax_16);  view_299 = amax_16 = None
    exp_16: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_17: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [1], True)
    div_24: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_16: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_300: "f32[512, 768]" = torch.ops.aten.view.default(add_99, [512, 768])
    permute_162: "f32[768, 384]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    addmm_59: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_205, view_300, permute_162);  primals_205 = None
    view_301: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_59, [1, 512, 384]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_302: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_301, [1, -1, 384]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_163: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
    clone_24: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    unsqueeze_58: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_24, -1);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_32: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_59: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_32, 0);  iota_32 = None
    iota_33: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_60: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
    add_102: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_59, unsqueeze_60);  unsqueeze_59 = unsqueeze_60 = None
    iota_34: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_61: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_34, 0);  iota_34 = None
    iota_35: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_62: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
    add_103: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_61, unsqueeze_62);  unsqueeze_61 = unsqueeze_62 = None
    constant_pad_nd_8: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_58, [0, 0, 4, 4], 0.0);  unsqueeze_58 = None
    unsqueeze_63: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_102, -1);  add_102 = None
    unsqueeze_64: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, -1);  unsqueeze_63 = None
    slice_21: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_8, 0, 0, 9223372036854775807);  constant_pad_nd_8 = None
    slice_22: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    index_8: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_22, [None, None, unsqueeze_64, add_103]);  slice_22 = unsqueeze_64 = add_103 = None
    permute_164: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_8, [0, 1, 2, 4, 3, 5]);  index_8 = None
    view_303: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_164, [1, 3456, 512]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_165: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    view_304: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_165, [1, 512, 384, 9]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_25: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_304, memory_format = torch.contiguous_format);  view_304 = None
    view_305: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_25, [3072, 64, 9]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_49: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_305, [3072, 64, 9]);  view_305 = None
    view_306: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_49, [3072, 64, 9]);  expand_49 = None
    expand_50: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_24, [3072, 9, 1]);  div_24 = None
    view_307: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_50, [3072, 9, 1]);  expand_50 = None
    bmm_24: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_306, view_307)
    view_308: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_24, [3072, 64, 1]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_309: "f32[512, 384]" = torch.ops.aten.view.default(view_308, [-1, 384]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_166: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_158, [0, 1, 3, 2]);  permute_158 = None
    expand_51: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_157, [1, 6, 512, 64]);  permute_157 = None
    view_310: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_51, [6, 512, 64]);  expand_51 = None
    expand_52: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_166, [1, 6, 64, 512]);  permute_166 = None
    view_311: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_52, [6, 64, 512]);  expand_52 = None
    bmm_25: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_310, view_311)
    view_312: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 6, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_25: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_312, 8.0);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_104: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_25, mul);  div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_17: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_104, [-1], True)
    sub_35: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_104, amax_17);  add_104 = amax_17 = None
    exp_17: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_18: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_26: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_17: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_25 = torch.ops.aten.native_dropout.default(div_26, 0.1, True);  div_26 = None
    getitem_84: "f32[1, 6, 512, 512]" = native_dropout_25[0]
    getitem_85: "b8[1, 6, 512, 512]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_53: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_84, [1, 6, 512, 512]);  getitem_84 = None
    view_313: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_53, [6, 512, 512]);  expand_53 = None
    expand_54: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_159, [1, 6, 512, 64]);  permute_159 = None
    view_314: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_54, [6, 512, 64]);  expand_54 = None
    bmm_26: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_313, view_314)
    view_315: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_26, [1, 6, 512, 64]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_167: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
    clone_26: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_316: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_309, [1, -1, 6, 64]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_8: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_26, view_316], 2);  clone_26 = view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_317: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_8, [1, 512, 768]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_318: "f32[512, 768]" = torch.ops.aten.view.default(view_317, [512, 768]);  view_317 = None
    permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
    addmm_60: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_207, view_318, permute_168);  primals_207 = None
    view_319: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_60, [1, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_319, 0.1, True);  view_319 = None
    getitem_86: "f32[1, 512, 768]" = native_dropout_26[0]
    getitem_87: "b8[1, 512, 768]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_105: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_86, add_99);  getitem_86 = add_99 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_106: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_105, getitem_89)
    mul_68: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_17);  sub_36 = None
    mul_69: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_68, primals_208);  mul_68 = None
    add_107: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_69, primals_209);  mul_69 = primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_320: "f32[512, 768]" = torch.ops.aten.view.default(add_107, [512, 768])
    permute_169: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
    addmm_61: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_211, view_320, permute_169);  primals_211 = None
    view_321: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_61, [1, 512, 3072]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, 0.5)
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, 0.7071067811865476)
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
    add_108: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_72: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_70, add_108);  mul_70 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_322: "f32[512, 3072]" = torch.ops.aten.view.default(mul_72, [512, 3072]);  mul_72 = None
    permute_170: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
    addmm_62: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_213, view_322, permute_170);  primals_213 = None
    view_323: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_62, [1, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_323, 0.1, True);  view_323 = None
    getitem_90: "f32[1, 512, 768]" = native_dropout_27[0]
    getitem_91: "b8[1, 512, 768]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_109: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_90, add_107);  getitem_90 = add_107 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_110: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_109, getitem_93)
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_18);  sub_37 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_214);  mul_73 = None
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_215);  mul_74 = primals_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_324: "f32[512, 768]" = torch.ops.aten.view.default(add_111, [512, 768])
    permute_171: "f32[768, 384]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    addmm_63: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_217, view_324, permute_171);  primals_217 = None
    view_325: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_63, [1, 512, 384]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_326: "f32[512, 768]" = torch.ops.aten.view.default(add_111, [512, 768])
    permute_172: "f32[768, 384]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    addmm_64: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_219, view_326, permute_172);  primals_219 = None
    view_327: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_64, [1, 512, 384]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_328: "f32[512, 768]" = torch.ops.aten.view.default(add_111, [512, 768])
    permute_173: "f32[768, 384]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_65: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_221, view_328, permute_173);  primals_221 = None
    view_329: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_65, [1, 512, 384]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_174: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_111, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_18: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_174, primals_222, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_19: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_18, primals_223, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_112: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_19, primals_10);  convolution_19 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_330: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_325, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_176: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_331: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_327, [1, 512, 6, 64]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_177: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_332: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_329, [1, 512, 6, 64]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_178: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_179: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_112, [0, 2, 1]);  add_112 = None
    mul_75: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_179, view_325)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_180: "f32[384, 54]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
    view_333: "f32[512, 384]" = torch.ops.aten.view.default(mul_75, [512, 384]);  mul_75 = None
    mm_9: "f32[512, 54]" = torch.ops.aten.mm.default(view_333, permute_180)
    view_334: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_9, [1, 512, 54]);  mm_9 = None
    add_113: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_334, primals_225);  view_334 = primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_335: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_113, [-1, 9, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_18: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_335, [1], True)
    sub_38: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_335, amax_18);  view_335 = amax_18 = None
    exp_18: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_19: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True)
    div_27: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_18: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_336: "f32[512, 768]" = torch.ops.aten.view.default(add_111, [512, 768])
    permute_181: "f32[768, 384]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    addmm_66: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_227, view_336, permute_181);  primals_227 = None
    view_337: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_66, [1, 512, 384]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_338: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_337, [1, -1, 384]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_182: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
    clone_27: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    unsqueeze_65: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_27, -1);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_36: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_66: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_36, 0);  iota_36 = None
    iota_37: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_67: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
    add_114: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_66, unsqueeze_67);  unsqueeze_66 = unsqueeze_67 = None
    iota_38: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_68: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_38, 0);  iota_38 = None
    iota_39: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_69: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
    add_115: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_68, unsqueeze_69);  unsqueeze_68 = unsqueeze_69 = None
    constant_pad_nd_9: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_65, [0, 0, 4, 4], 0.0);  unsqueeze_65 = None
    unsqueeze_70: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_114, -1);  add_114 = None
    unsqueeze_71: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    slice_23: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_9, 0, 0, 9223372036854775807);  constant_pad_nd_9 = None
    slice_24: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_23, 1, 0, 9223372036854775807);  slice_23 = None
    index_9: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_24, [None, None, unsqueeze_71, add_115]);  slice_24 = unsqueeze_71 = add_115 = None
    permute_183: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_9, [0, 1, 2, 4, 3, 5]);  index_9 = None
    view_339: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_183, [1, 3456, 512]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_184: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
    view_340: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_184, [1, 512, 384, 9]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_28: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_340, memory_format = torch.contiguous_format);  view_340 = None
    view_341: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_28, [3072, 64, 9]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_55: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_341, [3072, 64, 9]);  view_341 = None
    view_342: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_55, [3072, 64, 9]);  expand_55 = None
    expand_56: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_27, [3072, 9, 1]);  div_27 = None
    view_343: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_56, [3072, 9, 1]);  expand_56 = None
    bmm_27: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_342, view_343)
    view_344: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_27, [3072, 64, 1]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_345: "f32[512, 384]" = torch.ops.aten.view.default(view_344, [-1, 384]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_185: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_177, [0, 1, 3, 2]);  permute_177 = None
    expand_57: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_176, [1, 6, 512, 64]);  permute_176 = None
    view_346: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_57, [6, 512, 64]);  expand_57 = None
    expand_58: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_185, [1, 6, 64, 512]);  permute_185 = None
    view_347: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_58, [6, 64, 512]);  expand_58 = None
    bmm_28: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_346, view_347)
    view_348: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_28, [1, 6, 512, 512]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_28: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_348, 8.0);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_116: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_28, mul);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_19: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_116, [-1], True)
    sub_39: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_116, amax_19);  add_116 = amax_19 = None
    exp_19: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_20: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_29: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_19: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_28 = torch.ops.aten.native_dropout.default(div_29, 0.1, True);  div_29 = None
    getitem_94: "f32[1, 6, 512, 512]" = native_dropout_28[0]
    getitem_95: "b8[1, 6, 512, 512]" = native_dropout_28[1];  native_dropout_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_59: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_94, [1, 6, 512, 512]);  getitem_94 = None
    view_349: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_59, [6, 512, 512]);  expand_59 = None
    expand_60: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_178, [1, 6, 512, 64]);  permute_178 = None
    view_350: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_60, [6, 512, 64]);  expand_60 = None
    bmm_29: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_349, view_350)
    view_351: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_29, [1, 6, 512, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_186: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
    clone_29: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_352: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_345, [1, -1, 6, 64]);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_9: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_29, view_352], 2);  clone_29 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_353: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_9, [1, 512, 768]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_354: "f32[512, 768]" = torch.ops.aten.view.default(view_353, [512, 768]);  view_353 = None
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
    addmm_67: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_229, view_354, permute_187);  primals_229 = None
    view_355: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_67, [1, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_355, 0.1, True);  view_355 = None
    getitem_96: "f32[1, 512, 768]" = native_dropout_29[0]
    getitem_97: "b8[1, 512, 768]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_96, add_111);  getitem_96 = add_111 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_118: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_117, getitem_99)
    mul_76: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = None
    mul_77: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_76, primals_230);  mul_76 = None
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_77, primals_231);  mul_77 = primals_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[512, 768]" = torch.ops.aten.view.default(add_119, [512, 768])
    permute_188: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    addmm_68: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_233, view_356, permute_188);  primals_233 = None
    view_357: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_68, [1, 512, 3072]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_78: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.5)
    mul_79: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476)
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_120: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_80: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_78, add_120);  mul_78 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_358: "f32[512, 3072]" = torch.ops.aten.view.default(mul_80, [512, 3072]);  mul_80 = None
    permute_189: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_235, view_358, permute_189);  primals_235 = None
    view_359: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_69, [1, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_359, 0.1, True);  view_359 = None
    getitem_100: "f32[1, 512, 768]" = native_dropout_30[0]
    getitem_101: "b8[1, 512, 768]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_121: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_100, add_119);  getitem_100 = add_119 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_103: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_122: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-12);  getitem_102 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_41: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_121, getitem_103)
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_20);  sub_41 = None
    mul_82: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, primals_236);  mul_81 = None
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, primals_237);  mul_82 = primals_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_360: "f32[512, 768]" = torch.ops.aten.view.default(add_123, [512, 768])
    permute_190: "f32[768, 384]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
    addmm_70: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_239, view_360, permute_190);  primals_239 = None
    view_361: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_70, [1, 512, 384]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_362: "f32[512, 768]" = torch.ops.aten.view.default(add_123, [512, 768])
    permute_191: "f32[768, 384]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    addmm_71: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_241, view_362, permute_191);  primals_241 = None
    view_363: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_71, [1, 512, 384]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_364: "f32[512, 768]" = torch.ops.aten.view.default(add_123, [512, 768])
    permute_192: "f32[768, 384]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    addmm_72: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_243, view_364, permute_192);  primals_243 = None
    view_365: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_72, [1, 512, 384]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_193: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_123, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_20: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_193, primals_244, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_21: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_20, primals_245, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_124: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_21, primals_11);  convolution_21 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_366: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_361, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_367: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_363, [1, 512, 6, 64]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_196: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_368: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_365, [1, 512, 6, 64]);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_197: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_198: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None
    mul_83: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_198, view_361)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_199: "f32[384, 54]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    view_369: "f32[512, 384]" = torch.ops.aten.view.default(mul_83, [512, 384]);  mul_83 = None
    mm_10: "f32[512, 54]" = torch.ops.aten.mm.default(view_369, permute_199)
    view_370: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_10, [1, 512, 54]);  mm_10 = None
    add_125: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_370, primals_247);  view_370 = primals_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_371: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_125, [-1, 9, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_20: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_371, [1], True)
    sub_42: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_371, amax_20);  view_371 = amax_20 = None
    exp_20: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_21: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [1], True)
    div_30: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_20: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_372: "f32[512, 768]" = torch.ops.aten.view.default(add_123, [512, 768])
    permute_200: "f32[768, 384]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    addmm_73: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_249, view_372, permute_200);  primals_249 = None
    view_373: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_73, [1, 512, 384]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_374: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_373, [1, -1, 384]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_201: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_374, [0, 2, 1]);  view_374 = None
    clone_30: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
    unsqueeze_72: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_30, -1);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_40: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_73: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_40, 0);  iota_40 = None
    iota_41: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_74: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
    add_126: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_73, unsqueeze_74);  unsqueeze_73 = unsqueeze_74 = None
    iota_42: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_75: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_42, 0);  iota_42 = None
    iota_43: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_76: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
    add_127: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_75, unsqueeze_76);  unsqueeze_75 = unsqueeze_76 = None
    constant_pad_nd_10: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_72, [0, 0, 4, 4], 0.0);  unsqueeze_72 = None
    unsqueeze_77: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_126, -1);  add_126 = None
    unsqueeze_78: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, -1);  unsqueeze_77 = None
    slice_25: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_10, 0, 0, 9223372036854775807);  constant_pad_nd_10 = None
    slice_26: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 9223372036854775807);  slice_25 = None
    index_10: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_26, [None, None, unsqueeze_78, add_127]);  slice_26 = unsqueeze_78 = add_127 = None
    permute_202: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_10, [0, 1, 2, 4, 3, 5]);  index_10 = None
    view_375: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_202, [1, 3456, 512]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_203: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
    view_376: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_203, [1, 512, 384, 9]);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_31: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_376, memory_format = torch.contiguous_format);  view_376 = None
    view_377: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_31, [3072, 64, 9]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_61: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_377, [3072, 64, 9]);  view_377 = None
    view_378: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_61, [3072, 64, 9]);  expand_61 = None
    expand_62: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_30, [3072, 9, 1]);  div_30 = None
    view_379: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_62, [3072, 9, 1]);  expand_62 = None
    bmm_30: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_378, view_379)
    view_380: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_30, [3072, 64, 1]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_381: "f32[512, 384]" = torch.ops.aten.view.default(view_380, [-1, 384]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_204: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_196, [0, 1, 3, 2]);  permute_196 = None
    expand_63: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_195, [1, 6, 512, 64]);  permute_195 = None
    view_382: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_63, [6, 512, 64]);  expand_63 = None
    expand_64: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_204, [1, 6, 64, 512]);  permute_204 = None
    view_383: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_64, [6, 64, 512]);  expand_64 = None
    bmm_31: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_382, view_383)
    view_384: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_31, [1, 6, 512, 512]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_31: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_384, 8.0);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_128: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_31, mul);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_21: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_128, [-1], True)
    sub_43: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_128, amax_21);  add_128 = amax_21 = None
    exp_21: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_22: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_32: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_21: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_31 = torch.ops.aten.native_dropout.default(div_32, 0.1, True);  div_32 = None
    getitem_104: "f32[1, 6, 512, 512]" = native_dropout_31[0]
    getitem_105: "b8[1, 6, 512, 512]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_65: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_104, [1, 6, 512, 512]);  getitem_104 = None
    view_385: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_65, [6, 512, 512]);  expand_65 = None
    expand_66: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_197, [1, 6, 512, 64]);  permute_197 = None
    view_386: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_66, [6, 512, 64]);  expand_66 = None
    bmm_32: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_385, view_386)
    view_387: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 6, 512, 64]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
    clone_32: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_388: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_381, [1, -1, 6, 64]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_10: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_32, view_388], 2);  clone_32 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_389: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_10, [1, 512, 768]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_390: "f32[512, 768]" = torch.ops.aten.view.default(view_389, [512, 768]);  view_389 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    addmm_74: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_251, view_390, permute_206);  primals_251 = None
    view_391: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_74, [1, 512, 768]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_391, 0.1, True);  view_391 = None
    getitem_106: "f32[1, 512, 768]" = native_dropout_32[0]
    getitem_107: "b8[1, 512, 768]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_106, add_123);  getitem_106 = add_123 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_130: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-12);  getitem_108 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_129, getitem_109)
    mul_84: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_21);  sub_44 = None
    mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_252);  mul_84 = None
    add_131: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_85, primals_253);  mul_85 = primals_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[512, 768]" = torch.ops.aten.view.default(add_131, [512, 768])
    permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
    addmm_75: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_255, view_392, permute_207);  primals_255 = None
    view_393: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_75, [1, 512, 3072]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_86: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
    mul_87: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_132: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_88: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_132);  mul_86 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_394: "f32[512, 3072]" = torch.ops.aten.view.default(mul_88, [512, 3072]);  mul_88 = None
    permute_208: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
    addmm_76: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_257, view_394, permute_208);  primals_257 = None
    view_395: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_76, [1, 512, 768]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_395, 0.1, True);  view_395 = None
    getitem_110: "f32[1, 512, 768]" = native_dropout_33[0]
    getitem_111: "b8[1, 512, 768]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_110, add_131);  getitem_110 = add_131 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_113: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_134: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-12);  getitem_112 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_133, getitem_113)
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_22);  sub_45 = None
    mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, primals_258);  mul_89 = None
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_90, primals_259);  mul_90 = primals_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_396: "f32[512, 768]" = torch.ops.aten.view.default(add_135, [512, 768])
    permute_209: "f32[768, 384]" = torch.ops.aten.permute.default(primals_260, [1, 0]);  primals_260 = None
    addmm_77: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_261, view_396, permute_209);  primals_261 = None
    view_397: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_77, [1, 512, 384]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_398: "f32[512, 768]" = torch.ops.aten.view.default(add_135, [512, 768])
    permute_210: "f32[768, 384]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
    addmm_78: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_263, view_398, permute_210);  primals_263 = None
    view_399: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_78, [1, 512, 384]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_400: "f32[512, 768]" = torch.ops.aten.view.default(add_135, [512, 768])
    permute_211: "f32[768, 384]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None
    addmm_79: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_265, view_400, permute_211);  primals_265 = None
    view_401: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_79, [1, 512, 384]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_212: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_135, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_22: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_212, primals_266, None, [1], [4], [1], False, [0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_23: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_22, primals_267, None, [1], [0], [1], False, [0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_136: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_23, primals_12);  convolution_23 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_402: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_397, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_214: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_403: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_399, [1, 512, 6, 64]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_215: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_404: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_401, [1, 512, 6, 64]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_216: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_217: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_136, [0, 2, 1]);  add_136 = None
    mul_91: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_217, view_397)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_218: "f32[384, 54]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    view_405: "f32[512, 384]" = torch.ops.aten.view.default(mul_91, [512, 384]);  mul_91 = None
    mm_11: "f32[512, 54]" = torch.ops.aten.mm.default(view_405, permute_218)
    view_406: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_11, [1, 512, 54]);  mm_11 = None
    add_137: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_406, primals_269);  view_406 = primals_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_407: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_137, [-1, 9, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_22: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_407, [1], True)
    sub_46: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_407, amax_22);  view_407 = amax_22 = None
    exp_22: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_23: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [1], True)
    div_33: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_22: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(div_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_408: "f32[512, 768]" = torch.ops.aten.view.default(add_135, [512, 768])
    permute_219: "f32[768, 384]" = torch.ops.aten.permute.default(primals_270, [1, 0]);  primals_270 = None
    addmm_80: "f32[512, 384]" = torch.ops.aten.addmm.default(primals_271, view_408, permute_219);  primals_271 = None
    view_409: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_80, [1, 512, 384]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_410: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_409, [1, -1, 384]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_220: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
    clone_33: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
    unsqueeze_79: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_33, -1);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_44: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_80: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_44, 0);  iota_44 = None
    iota_45: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_81: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
    add_138: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_80, unsqueeze_81);  unsqueeze_80 = unsqueeze_81 = None
    iota_46: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_82: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_46, 0);  iota_46 = None
    iota_47: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_83: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
    add_139: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_82, unsqueeze_83);  unsqueeze_82 = unsqueeze_83 = None
    constant_pad_nd_11: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_79, [0, 0, 4, 4], 0.0);  unsqueeze_79 = None
    unsqueeze_84: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_138, -1);  add_138 = None
    unsqueeze_85: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    slice_27: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_11, 0, 0, 9223372036854775807);  constant_pad_nd_11 = None
    slice_28: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_27, 1, 0, 9223372036854775807);  slice_27 = None
    index_11: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_28, [None, None, unsqueeze_85, add_139]);  slice_28 = unsqueeze_85 = add_139 = None
    permute_221: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_11, [0, 1, 2, 4, 3, 5]);  index_11 = None
    view_411: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_221, [1, 3456, 512]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_222: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
    view_412: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_222, [1, 512, 384, 9]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_34: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_412, memory_format = torch.contiguous_format);  view_412 = None
    view_413: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_34, [3072, 64, 9]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_67: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_413, [3072, 64, 9]);  view_413 = None
    view_414: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_67, [3072, 64, 9]);  expand_67 = None
    expand_68: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_33, [3072, 9, 1]);  div_33 = None
    view_415: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_68, [3072, 9, 1]);  expand_68 = None
    bmm_33: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_414, view_415)
    view_416: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_33, [3072, 64, 1]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_417: "f32[512, 384]" = torch.ops.aten.view.default(view_416, [-1, 384]);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_223: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2]);  permute_215 = None
    expand_69: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_214, [1, 6, 512, 64]);  permute_214 = None
    view_418: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_69, [6, 512, 64]);  expand_69 = None
    expand_70: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_223, [1, 6, 64, 512]);  permute_223 = None
    view_419: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_70, [6, 64, 512]);  expand_70 = None
    bmm_34: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_418, view_419)
    view_420: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_34, [1, 6, 512, 512]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_34: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_420, 8.0);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_140: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_34, mul);  div_34 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_23: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_140, [-1], True)
    sub_47: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_140, amax_23);  add_140 = amax_23 = None
    exp_23: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
    sum_24: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_35: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_23: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(div_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    native_dropout_34 = torch.ops.aten.native_dropout.default(div_35, 0.1, True);  div_35 = None
    getitem_114: "f32[1, 6, 512, 512]" = native_dropout_34[0]
    getitem_115: "b8[1, 6, 512, 512]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_71: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(getitem_114, [1, 6, 512, 512]);  getitem_114 = None
    view_421: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_71, [6, 512, 512]);  expand_71 = None
    expand_72: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_216, [1, 6, 512, 64]);  permute_216 = None
    view_422: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_72, [6, 512, 64]);  expand_72 = None
    bmm_35: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_421, view_422)
    view_423: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 6, 512, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_224: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None
    clone_35: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_424: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_417, [1, -1, 6, 64]);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_11: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_35, view_424], 2);  clone_35 = view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_425: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_11, [1, 512, 768]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_426: "f32[512, 768]" = torch.ops.aten.view.default(view_425, [512, 768]);  view_425 = None
    permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
    addmm_81: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_273, view_426, permute_225);  primals_273 = None
    view_427: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_81, [1, 512, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_427, 0.1, True);  view_427 = None
    getitem_116: "f32[1, 512, 768]" = native_dropout_35[0]
    getitem_117: "b8[1, 512, 768]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_116, add_135);  getitem_116 = add_135 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_142: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-12);  getitem_118 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_141, getitem_119)
    mul_92: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = None
    mul_93: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, primals_274);  mul_92 = None
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, primals_275);  mul_93 = primals_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_428: "f32[512, 768]" = torch.ops.aten.view.default(add_143, [512, 768])
    permute_226: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_276, [1, 0]);  primals_276 = None
    addmm_82: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_277, view_428, permute_226);  primals_277 = None
    view_429: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_82, [1, 512, 3072]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_94: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.5)
    mul_95: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.7071067811865476)
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
    add_144: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_96: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_94, add_144);  mul_94 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_430: "f32[512, 3072]" = torch.ops.aten.view.default(mul_96, [512, 3072]);  mul_96 = None
    permute_227: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    addmm_83: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_279, view_430, permute_227);  primals_279 = None
    view_431: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_83, [1, 512, 768]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_431, 0.1, True);  view_431 = None
    getitem_120: "f32[1, 512, 768]" = native_dropout_36[0]
    getitem_121: "b8[1, 512, 768]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_145: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_120, add_143);  getitem_120 = add_143 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_123: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_146: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-12);  getitem_122 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_145, getitem_123)
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_24);  sub_49 = None
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, primals_280);  mul_97 = None
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_98, primals_281);  mul_98 = primals_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:873, code: hidden_states = self.dense(generator_hidden_states)
    view_432: "f32[512, 768]" = torch.ops.aten.view.default(add_147, [512, 768]);  add_147 = None
    permute_228: "f32[768, 768]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
    addmm_84: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_283, view_432, permute_228);  primals_283 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_84, [1, 512, 768]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, 0.5)
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, 0.7071067811865476)
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_148: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, add_148);  mul_99 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:875, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
    getitem_124: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_125: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_149: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-12);  getitem_124 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_50: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_125)
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_284);  mul_102 = None
    add_150: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_285);  mul_103 = primals_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:941, code: prediction_scores = self.generator_lm_head(prediction_scores)
    view_434: "f32[512, 768]" = torch.ops.aten.view.default(add_150, [512, 768]);  add_150 = None
    permute_229: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_286, [1, 0]);  primals_286 = None
    addmm_85: "f32[512, 30522]" = torch.ops.aten.addmm.default(primals_287, view_434, permute_229);  primals_287 = None
    view_435: "f32[1, 512, 30522]" = torch.ops.aten.view.default(addmm_85, [1, 512, 30522]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:947, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_436: "f32[512, 30522]" = torch.ops.aten.view.default(view_435, [-1, 30522])
    view_437: "i64[512]" = torch.ops.aten.view.default(primals_291, [-1]);  primals_291 = None
    amax_24: "f32[512, 1]" = torch.ops.aten.amax.default(view_436, [1], True)
    sub_51: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(view_436, amax_24);  view_436 = amax_24 = None
    exp_24: "f32[512, 30522]" = torch.ops.aten.exp.default(sub_51)
    sum_25: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_52: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(sub_51, log);  sub_51 = log = None
    alias_24: "f32[512, 30522]" = torch.ops.aten.alias.default(sub_52)
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_437, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[512]" = torch.ops.aten.where.self(ne, view_437, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_86: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_52, 1, unsqueeze_86);  sub_52 = unsqueeze_86 = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[512]" = torch.ops.aten.ne.Scalar(view_437, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[512]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[512]" = torch.ops.aten.ne.Scalar(view_437, -100)
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_36: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = None
    div_37: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_87: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_437, 1);  view_437 = None
    ne_3: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_87, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_2: "i64[512, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_87, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full_1: "f32[512, 30522]" = torch.ops.aten.full.default([512, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[512, 30522]" = torch.ops.aten.scatter.value(full_1, 1, where_2, -1.0);  full_1 = where_2 = None
    ne_4: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_87, -100);  unsqueeze_87 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[512, 1]" = torch.ops.aten.where.self(ne_4, div_37, scalar_tensor_3);  ne_4 = div_37 = scalar_tensor_3 = None
    mul_104: "f32[512, 30522]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_25: "f32[512, 30522]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    exp_25: "f32[512, 30522]" = torch.ops.aten.exp.default(alias_25);  alias_25 = None
    sum_28: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [1], True)
    mul_105: "f32[512, 30522]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
    sub_53: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    view_438: "f32[1, 512, 30522]" = torch.ops.aten.view.default(sub_53, [1, 512, 30522]);  sub_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:947, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    add_151: "f32[1, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_438);  tangents_2 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:941, code: prediction_scores = self.generator_lm_head(prediction_scores)
    view_439: "f32[512, 30522]" = torch.ops.aten.view.default(add_151, [512, 30522]);  add_151 = None
    permute_230: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_439, permute_230);  permute_230 = None
    permute_231: "f32[30522, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_13: "f32[30522, 768]" = torch.ops.aten.mm.default(permute_231, view_434);  permute_231 = view_434 = None
    permute_232: "f32[768, 30522]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_29: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[30522]" = torch.ops.aten.view.default(sum_29, [30522]);  sum_29 = None
    permute_233: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_441: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:875, code: hidden_states = self.LayerNorm(hidden_states)
    sub_54: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_125);  mul_101 = getitem_125 = None
    mul_106: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_25);  sub_54 = None
    mul_107: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_441, primals_284);  primals_284 = None
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, 768)
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_107, [2], True)
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, mul_106);  mul_107 = None
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True);  mul_109 = None
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_106, sum_31);  sum_31 = None
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_108, sum_30);  mul_108 = sum_30 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_110);  sub_55 = mul_110 = None
    div_38: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_56);  div_38 = sub_56 = None
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_441, mul_106);  mul_106 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_112, [0, 1]);  mul_112 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_441, [0, 1]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, 0.7071067811865476)
    erf_13: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_152: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, 0.5);  add_152 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, view_433)
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, -0.5);  mul_115 = None
    exp_26: "f32[1, 512, 768]" = torch.ops.aten.exp.default(mul_116);  mul_116 = None
    mul_117: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_118: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, mul_117);  view_433 = mul_117 = None
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, mul_118);  mul_114 = mul_118 = None
    mul_119: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_111, add_153);  mul_111 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:873, code: hidden_states = self.dense(generator_hidden_states)
    view_442: "f32[512, 768]" = torch.ops.aten.view.default(mul_119, [512, 768]);  mul_119 = None
    permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_442, permute_234);  permute_234 = None
    permute_235: "f32[768, 512]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_235, view_432);  permute_235 = view_432 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_444: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_57: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_145, getitem_123);  add_145 = getitem_123 = None
    mul_120: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_24);  sub_57 = None
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_444, primals_280);  primals_280 = None
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, 768)
    sum_35: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, mul_120);  mul_121 = None
    sum_36: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, sum_36);  sum_36 = None
    sub_58: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_122, sum_35);  mul_122 = sum_35 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_58, mul_124);  sub_58 = mul_124 = None
    div_39: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_59);  div_39 = sub_59 = None
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_444, mul_120);  mul_120 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_444, [0, 1]);  view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_1: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_125, mul_127);  mul_127 = None
    clone_36: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_128, memory_format = torch.contiguous_format);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_445: "f32[512, 768]" = torch.ops.aten.view.default(clone_36, [512, 768]);  clone_36 = None
    permute_238: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_445, permute_238);  permute_238 = None
    permute_239: "f32[768, 512]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_239, view_430);  permute_239 = view_430 = None
    permute_240: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_39: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[768]" = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
    permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_447: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_129: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.7071067811865476)
    erf_14: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_154: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_130: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_154, 0.5);  add_154 = None
    mul_131: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, view_429)
    mul_132: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_131, -0.5);  mul_131 = None
    exp_27: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_132);  mul_132 = None
    mul_133: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_134: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, mul_133);  view_429 = mul_133 = None
    add_155: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_130, mul_134);  mul_130 = mul_134 = None
    mul_135: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_447, add_155);  view_447 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_448: "f32[512, 3072]" = torch.ops.aten.view.default(mul_135, [512, 3072]);  mul_135 = None
    permute_242: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_448, permute_242);  permute_242 = None
    permute_243: "f32[3072, 512]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_243, view_428);  permute_243 = view_428 = None
    permute_244: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_40: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[3072]" = torch.ops.aten.view.default(sum_40, [3072]);  sum_40 = None
    permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_450: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_156: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_125, view_450);  mul_125 = view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_141, getitem_119);  add_141 = getitem_119 = None
    mul_136: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_23);  sub_60 = None
    mul_137: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_274);  primals_274 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, 768)
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True)
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, mul_136);  mul_137 = None
    sum_42: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_136, sum_42);  sum_42 = None
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_138, sum_41);  mul_138 = sum_41 = None
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_61, mul_140);  sub_61 = mul_140 = None
    div_40: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_62);  div_40 = sub_62 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_136);  mul_136 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1]);  mul_142 = None
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_141, mul_143);  mul_143 = None
    clone_37: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_144, memory_format = torch.contiguous_format);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_451: "f32[512, 768]" = torch.ops.aten.view.default(clone_37, [512, 768]);  clone_37 = None
    permute_246: "f32[768, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_451, permute_246);  permute_246 = None
    permute_247: "f32[768, 512]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_247, view_426);  permute_247 = view_426 = None
    permute_248: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
    permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_453: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_454: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_453, [1, 512, 12, 64]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_29: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_454, 2, 0, 6)
    slice_30: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_454, 2, 6, 12);  view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_455: "f32[512, 384]" = torch.ops.aten.view.default(slice_30, [512, 384]);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_250: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_29, [0, 2, 1, 3]);  slice_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_456: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_250, [6, 512, 64]);  permute_250 = None
    permute_251: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
    bmm_36: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_251, view_456);  permute_251 = None
    permute_252: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_422, [0, 2, 1]);  view_422 = None
    bmm_37: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_456, permute_252);  view_456 = permute_252 = None
    view_457: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_36, [1, 6, 512, 64]);  bmm_36 = None
    view_458: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_37, [1, 6, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_3: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_145: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_146: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_458, mul_145);  view_458 = mul_145 = None
    clone_38: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_146, memory_format = torch.contiguous_format);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_26: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_147: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_38, alias_26);  clone_38 = None
    sum_46: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [-1], True)
    mul_148: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_26, sum_46);  alias_26 = sum_46 = None
    sub_63: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_41: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_63, 8.0);  sub_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_459: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_41, [6, 512, 512]);  div_41 = None
    permute_253: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
    bmm_38: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_253, view_459);  permute_253 = None
    permute_254: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
    bmm_39: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_459, permute_254);  view_459 = permute_254 = None
    view_460: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_38, [1, 6, 64, 512]);  bmm_38 = None
    view_461: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 6, 512, 64]);  bmm_39 = None
    permute_255: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_460, [0, 1, 3, 2]);  view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_39: "f32[512, 384]" = torch.ops.aten.clone.default(view_455, memory_format = torch.contiguous_format);  view_455 = None
    view_462: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_39, [3072, 64, 1]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_463: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_462, [3072, 64, 1]);  view_462 = None
    permute_256: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_414, [0, 2, 1]);  view_414 = None
    bmm_40: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_256, view_463);  permute_256 = None
    permute_257: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_415, [0, 2, 1]);  view_415 = None
    bmm_41: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_463, permute_257);  view_463 = permute_257 = None
    view_464: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_40, [3072, 9, 1]);  bmm_40 = None
    view_465: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_41, [3072, 64, 9]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_466: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_465, [1, 512, 384, 9]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_467: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_466, [1, 512, 3456]);  view_466 = None
    permute_258: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_468: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_258, [1, 384, 9, 1, 512, 1]);  permute_258 = None
    permute_259: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_468, [0, 1, 2, 4, 3, 5]);  view_468 = None
    iota_48: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_88: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_48, 0);  iota_48 = None
    iota_49: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_89: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_49, -1);  iota_49 = None
    add_157: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_88, unsqueeze_89);  unsqueeze_88 = unsqueeze_89 = None
    unsqueeze_90: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_157, -1);  add_157 = None
    unsqueeze_91: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    iota_50: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_92: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_50, 0);  iota_50 = None
    iota_51: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_93: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_51, -1);  iota_51 = None
    add_158: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_92, unsqueeze_93);  unsqueeze_92 = unsqueeze_93 = None
    full_2: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_2, [None, None, unsqueeze_91, add_158], permute_259, True);  full_2 = unsqueeze_91 = add_158 = permute_259 = None
    constant_pad_nd_12: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put, [0, 0, -4, -4], 0.0);  _unsafe_index_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_1: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_12, -1);  constant_pad_nd_12 = None
    permute_260: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_1, [0, 2, 1]);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_469: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_260, [1, 512, 384]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_470: "f32[512, 384]" = torch.ops.aten.view.default(view_469, [512, 384]);  view_469 = None
    permute_261: "f32[384, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_470, permute_261);  permute_261 = None
    permute_262: "f32[384, 512]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_23: "f32[384, 768]" = torch.ops.aten.mm.default(permute_262, view_408);  permute_262 = view_408 = None
    permute_263: "f32[768, 384]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_47: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[384]" = torch.ops.aten.view.default(sum_47, [384]);  sum_47 = None
    permute_264: "f32[384, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_472: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_141, view_472);  mul_141 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_27: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    mul_149: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_464, alias_27);  view_464 = None
    sum_48: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [1], True)
    mul_150: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_27, sum_48);  alias_27 = sum_48 = None
    sub_64: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_473: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_64, [1, 512, 54]);  sub_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_49: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_473, [0, 1], True)
    view_474: "f32[54]" = torch.ops.aten.view.default(sum_49, [54]);  sum_49 = None
    view_475: "f32[512, 54]" = torch.ops.aten.view.default(view_473, [512, 54]);  view_473 = None
    permute_265: "f32[54, 512]" = torch.ops.aten.permute.default(view_475, [1, 0])
    mm_24: "f32[54, 384]" = torch.ops.aten.mm.default(permute_265, view_405);  permute_265 = view_405 = None
    permute_266: "f32[384, 54]" = torch.ops.aten.permute.default(mm_24, [1, 0]);  mm_24 = None
    permute_267: "f32[54, 512]" = torch.ops.aten.permute.default(view_475, [1, 0]);  view_475 = None
    mm_25: "f32[384, 512]" = torch.ops.aten.mm.default(permute_218, permute_267);  permute_218 = permute_267 = None
    permute_268: "f32[512, 384]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    view_476: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_268, [1, 512, 384]);  permute_268 = None
    permute_269: "f32[54, 384]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_151: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_476, permute_217);  permute_217 = None
    mul_152: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_476, view_397);  view_476 = view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_270: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_457, [0, 2, 1, 3]);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_40: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
    view_477: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_40, [1, 512, 384]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_271: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_255, [0, 2, 1, 3]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_478: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_271, [1, 512, 384]);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_272: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_461, [0, 2, 1, 3]);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_41: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
    view_479: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_41, [1, 512, 384]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_160: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_151, view_479);  mul_151 = view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_273: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_152, [0, 2, 1]);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_50: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_273, [0, 2], True)
    view_480: "f32[384, 1]" = torch.ops.aten.view.default(sum_50, [384, 1]);  sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_273, convolution_22, primals_267, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_273 = convolution_22 = primals_267 = None
    getitem_126: "f32[1, 768, 512]" = convolution_backward[0]
    getitem_127: "f32[384, 768, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(getitem_126, permute_212, primals_266, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_126 = permute_212 = primals_266 = None
    getitem_129: "f32[1, 768, 512]" = convolution_backward_1[0]
    getitem_130: "f32[768, 1, 9]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_274: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_129, [0, 2, 1]);  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_161: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_159, permute_274);  add_159 = permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_481: "f32[512, 384]" = torch.ops.aten.view.default(view_477, [512, 384]);  view_477 = None
    permute_275: "f32[384, 768]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_481, permute_275);  permute_275 = None
    permute_276: "f32[384, 512]" = torch.ops.aten.permute.default(view_481, [1, 0])
    mm_27: "f32[384, 768]" = torch.ops.aten.mm.default(permute_276, view_400);  permute_276 = view_400 = None
    permute_277: "f32[768, 384]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_51: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
    view_482: "f32[384]" = torch.ops.aten.view.default(sum_51, [384]);  sum_51 = None
    permute_278: "f32[384, 768]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    view_483: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_162: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_161, view_483);  add_161 = view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_484: "f32[512, 384]" = torch.ops.aten.view.default(view_478, [512, 384]);  view_478 = None
    permute_279: "f32[384, 768]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    mm_28: "f32[512, 768]" = torch.ops.aten.mm.default(view_484, permute_279);  permute_279 = None
    permute_280: "f32[384, 512]" = torch.ops.aten.permute.default(view_484, [1, 0])
    mm_29: "f32[384, 768]" = torch.ops.aten.mm.default(permute_280, view_398);  permute_280 = view_398 = None
    permute_281: "f32[768, 384]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_52: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
    view_485: "f32[384]" = torch.ops.aten.view.default(sum_52, [384]);  sum_52 = None
    permute_282: "f32[384, 768]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    view_486: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_28, [1, 512, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_163: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_162, view_486);  add_162 = view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_487: "f32[512, 384]" = torch.ops.aten.view.default(add_160, [512, 384]);  add_160 = None
    permute_283: "f32[384, 768]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_487, permute_283);  permute_283 = None
    permute_284: "f32[384, 512]" = torch.ops.aten.permute.default(view_487, [1, 0])
    mm_31: "f32[384, 768]" = torch.ops.aten.mm.default(permute_284, view_396);  permute_284 = view_396 = None
    permute_285: "f32[768, 384]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_53: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
    view_488: "f32[384]" = torch.ops.aten.view.default(sum_53, [384]);  sum_53 = None
    permute_286: "f32[384, 768]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_489: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_164: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_163, view_489);  add_163 = view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_65: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_133, getitem_113);  add_133 = getitem_113 = None
    mul_153: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_22);  sub_65 = None
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_258);  primals_258 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, 768)
    sum_54: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True)
    mul_156: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, mul_153);  mul_154 = None
    sum_55: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_153, sum_55);  sum_55 = None
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_155, sum_54);  mul_155 = sum_54 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_157);  sub_66 = mul_157 = None
    div_42: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_158: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_67);  div_42 = sub_67 = None
    mul_159: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_153);  mul_153 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 1]);  mul_159 = None
    sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_160: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_161: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_158, mul_160);  mul_160 = None
    clone_42: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_161, memory_format = torch.contiguous_format);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_490: "f32[512, 768]" = torch.ops.aten.view.default(clone_42, [512, 768]);  clone_42 = None
    permute_287: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    mm_32: "f32[512, 3072]" = torch.ops.aten.mm.default(view_490, permute_287);  permute_287 = None
    permute_288: "f32[768, 512]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_33: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_288, view_394);  permute_288 = view_394 = None
    permute_289: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[768]" = torch.ops.aten.view.default(sum_58, [768]);  sum_58 = None
    permute_290: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    view_492: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_32, [1, 512, 3072]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_162: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
    erf_15: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_162);  mul_162 = None
    add_165: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_163: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_165, 0.5);  add_165 = None
    mul_164: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, view_393)
    mul_165: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_164, -0.5);  mul_164 = None
    exp_28: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_165);  mul_165 = None
    mul_166: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_167: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, mul_166);  view_393 = mul_166 = None
    add_166: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_163, mul_167);  mul_163 = mul_167 = None
    mul_168: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_492, add_166);  view_492 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_493: "f32[512, 3072]" = torch.ops.aten.view.default(mul_168, [512, 3072]);  mul_168 = None
    permute_291: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_493, permute_291);  permute_291 = None
    permute_292: "f32[3072, 512]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_35: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_292, view_392);  permute_292 = view_392 = None
    permute_293: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_59: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[3072]" = torch.ops.aten.view.default(sum_59, [3072]);  sum_59 = None
    permute_294: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    view_495: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_167: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_158, view_495);  mul_158 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_68: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_129, getitem_109);  add_129 = getitem_109 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_21);  sub_68 = None
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_252);  primals_252 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_170, 768)
    sum_60: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True)
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_170, mul_169);  mul_170 = None
    sum_61: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_169, sum_61);  sum_61 = None
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_171, sum_60);  mul_171 = sum_60 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_173);  sub_69 = mul_173 = None
    div_43: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_174: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_70);  div_43 = sub_70 = None
    mul_175: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_169);  mul_169 = None
    sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_176: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_177: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_174, mul_176);  mul_176 = None
    clone_43: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_177, memory_format = torch.contiguous_format);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_496: "f32[512, 768]" = torch.ops.aten.view.default(clone_43, [512, 768]);  clone_43 = None
    permute_295: "f32[768, 768]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_496, permute_295);  permute_295 = None
    permute_296: "f32[768, 512]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_296, view_390);  permute_296 = view_390 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[768]" = torch.ops.aten.view.default(sum_64, [768]);  sum_64 = None
    permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_498: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_499: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_498, [1, 512, 12, 64]);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_31: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_499, 2, 0, 6)
    slice_32: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_499, 2, 6, 12);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_500: "f32[512, 384]" = torch.ops.aten.view.default(slice_32, [512, 384]);  slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_299: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_31, [0, 2, 1, 3]);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_501: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_299, [6, 512, 64]);  permute_299 = None
    permute_300: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    bmm_42: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_300, view_501);  permute_300 = None
    permute_301: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
    bmm_43: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_501, permute_301);  view_501 = permute_301 = None
    view_502: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_42, [1, 6, 512, 64]);  bmm_42 = None
    view_503: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_43, [1, 6, 512, 512]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_6: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_178: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_179: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_503, mul_178);  view_503 = mul_178 = None
    clone_44: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_179, memory_format = torch.contiguous_format);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_28: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_180: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_44, alias_28);  clone_44 = None
    sum_65: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [-1], True)
    mul_181: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_28, sum_65);  alias_28 = sum_65 = None
    sub_71: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_44: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_71, 8.0);  sub_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_504: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_44, [6, 512, 512]);  div_44 = None
    permute_302: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
    bmm_44: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_302, view_504);  permute_302 = None
    permute_303: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
    bmm_45: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_504, permute_303);  view_504 = permute_303 = None
    view_505: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_44, [1, 6, 64, 512]);  bmm_44 = None
    view_506: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_45, [1, 6, 512, 64]);  bmm_45 = None
    permute_304: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_505, [0, 1, 3, 2]);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_45: "f32[512, 384]" = torch.ops.aten.clone.default(view_500, memory_format = torch.contiguous_format);  view_500 = None
    view_507: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_45, [3072, 64, 1]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_508: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_507, [3072, 64, 1]);  view_507 = None
    permute_305: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None
    bmm_46: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_305, view_508);  permute_305 = None
    permute_306: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_379, [0, 2, 1]);  view_379 = None
    bmm_47: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_508, permute_306);  view_508 = permute_306 = None
    view_509: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_46, [3072, 9, 1]);  bmm_46 = None
    view_510: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_47, [3072, 64, 9]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_511: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_510, [1, 512, 384, 9]);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_512: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_511, [1, 512, 3456]);  view_511 = None
    permute_307: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_512, [0, 2, 1]);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_513: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_307, [1, 384, 9, 1, 512, 1]);  permute_307 = None
    permute_308: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_513, [0, 1, 2, 4, 3, 5]);  view_513 = None
    iota_52: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_94: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_52, 0);  iota_52 = None
    iota_53: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_95: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_53, -1);  iota_53 = None
    add_168: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_94, unsqueeze_95);  unsqueeze_94 = unsqueeze_95 = None
    unsqueeze_96: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_168, -1);  add_168 = None
    unsqueeze_97: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    iota_54: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_98: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_54, 0);  iota_54 = None
    iota_55: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_99: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_55, -1);  iota_55 = None
    add_169: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_98, unsqueeze_99);  unsqueeze_98 = unsqueeze_99 = None
    full_3: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_3, [None, None, unsqueeze_97, add_169], permute_308, True);  full_3 = unsqueeze_97 = add_169 = permute_308 = None
    constant_pad_nd_13: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_1, [0, 0, -4, -4], 0.0);  _unsafe_index_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_2: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_13, -1);  constant_pad_nd_13 = None
    permute_309: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_2, [0, 2, 1]);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_514: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_309, [1, 512, 384]);  permute_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_515: "f32[512, 384]" = torch.ops.aten.view.default(view_514, [512, 384]);  view_514 = None
    permute_310: "f32[384, 768]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    mm_38: "f32[512, 768]" = torch.ops.aten.mm.default(view_515, permute_310);  permute_310 = None
    permute_311: "f32[384, 512]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_39: "f32[384, 768]" = torch.ops.aten.mm.default(permute_311, view_372);  permute_311 = view_372 = None
    permute_312: "f32[768, 384]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_66: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[384]" = torch.ops.aten.view.default(sum_66, [384]);  sum_66 = None
    permute_313: "f32[384, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_517: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_38, [1, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_170: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_174, view_517);  mul_174 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_29: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_182: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_509, alias_29);  view_509 = None
    sum_67: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [1], True)
    mul_183: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_29, sum_67);  alias_29 = sum_67 = None
    sub_72: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_518: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_72, [1, 512, 54]);  sub_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_68: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_518, [0, 1], True)
    view_519: "f32[54]" = torch.ops.aten.view.default(sum_68, [54]);  sum_68 = None
    view_520: "f32[512, 54]" = torch.ops.aten.view.default(view_518, [512, 54]);  view_518 = None
    permute_314: "f32[54, 512]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_40: "f32[54, 384]" = torch.ops.aten.mm.default(permute_314, view_369);  permute_314 = view_369 = None
    permute_315: "f32[384, 54]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    permute_316: "f32[54, 512]" = torch.ops.aten.permute.default(view_520, [1, 0]);  view_520 = None
    mm_41: "f32[384, 512]" = torch.ops.aten.mm.default(permute_199, permute_316);  permute_199 = permute_316 = None
    permute_317: "f32[512, 384]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    view_521: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_317, [1, 512, 384]);  permute_317 = None
    permute_318: "f32[54, 384]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_184: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_521, permute_198);  permute_198 = None
    mul_185: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_521, view_361);  view_521 = view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_319: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_46: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_522: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_46, [1, 512, 384]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_320: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_304, [0, 2, 1, 3]);  permute_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_523: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_320, [1, 512, 384]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_321: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_47: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_524: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_47, [1, 512, 384]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_171: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_184, view_524);  mul_184 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_322: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_185, [0, 2, 1]);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_69: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_322, [0, 2], True)
    view_525: "f32[384, 1]" = torch.ops.aten.view.default(sum_69, [384, 1]);  sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_322, convolution_20, primals_245, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_322 = convolution_20 = primals_245 = None
    getitem_132: "f32[1, 768, 512]" = convolution_backward_2[0]
    getitem_133: "f32[384, 768, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(getitem_132, permute_193, primals_244, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_132 = permute_193 = primals_244 = None
    getitem_135: "f32[1, 768, 512]" = convolution_backward_3[0]
    getitem_136: "f32[768, 1, 9]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_323: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1]);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_172: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_170, permute_323);  add_170 = permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_526: "f32[512, 384]" = torch.ops.aten.view.default(view_522, [512, 384]);  view_522 = None
    permute_324: "f32[384, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_526, permute_324);  permute_324 = None
    permute_325: "f32[384, 512]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_43: "f32[384, 768]" = torch.ops.aten.mm.default(permute_325, view_364);  permute_325 = view_364 = None
    permute_326: "f32[768, 384]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_70: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[384]" = torch.ops.aten.view.default(sum_70, [384]);  sum_70 = None
    permute_327: "f32[384, 768]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    view_528: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_172, view_528);  add_172 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_529: "f32[512, 384]" = torch.ops.aten.view.default(view_523, [512, 384]);  view_523 = None
    permute_328: "f32[384, 768]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_529, permute_328);  permute_328 = None
    permute_329: "f32[384, 512]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_45: "f32[384, 768]" = torch.ops.aten.mm.default(permute_329, view_362);  permute_329 = view_362 = None
    permute_330: "f32[768, 384]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_71: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[384]" = torch.ops.aten.view.default(sum_71, [384]);  sum_71 = None
    permute_331: "f32[384, 768]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    view_531: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_174: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_173, view_531);  add_173 = view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_532: "f32[512, 384]" = torch.ops.aten.view.default(add_171, [512, 384]);  add_171 = None
    permute_332: "f32[384, 768]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_532, permute_332);  permute_332 = None
    permute_333: "f32[384, 512]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_47: "f32[384, 768]" = torch.ops.aten.mm.default(permute_333, view_360);  permute_333 = view_360 = None
    permute_334: "f32[768, 384]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_72: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
    view_533: "f32[384]" = torch.ops.aten.view.default(sum_72, [384]);  sum_72 = None
    permute_335: "f32[384, 768]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_534: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_175: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_174, view_534);  add_174 = view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_121, getitem_103);  add_121 = getitem_103 = None
    mul_186: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_20);  sub_73 = None
    mul_187: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_175, primals_236);  primals_236 = None
    mul_188: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_187, 768)
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True)
    mul_189: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_187, mul_186);  mul_187 = None
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [2], True);  mul_189 = None
    mul_190: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_186, sum_74);  sum_74 = None
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_188, sum_73);  mul_188 = sum_73 = None
    sub_75: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_190);  sub_74 = mul_190 = None
    div_45: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_191: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_75);  div_45 = sub_75 = None
    mul_192: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_175, mul_186);  mul_186 = None
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1]);  mul_192 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_175, [0, 1]);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_193: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_194: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_191, mul_193);  mul_193 = None
    clone_48: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_194, memory_format = torch.contiguous_format);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_535: "f32[512, 768]" = torch.ops.aten.view.default(clone_48, [512, 768]);  clone_48 = None
    permute_336: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    mm_48: "f32[512, 3072]" = torch.ops.aten.mm.default(view_535, permute_336);  permute_336 = None
    permute_337: "f32[768, 512]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_337, view_358);  permute_337 = view_358 = None
    permute_338: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
    view_536: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    permute_339: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_537: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_48, [1, 512, 3072]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_195: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476)
    erf_16: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_176: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_196: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
    mul_197: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, view_357)
    mul_198: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_197, -0.5);  mul_197 = None
    exp_29: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_198);  mul_198 = None
    mul_199: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_200: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, mul_199);  view_357 = mul_199 = None
    add_177: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_196, mul_200);  mul_196 = mul_200 = None
    mul_201: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_537, add_177);  view_537 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_538: "f32[512, 3072]" = torch.ops.aten.view.default(mul_201, [512, 3072]);  mul_201 = None
    permute_340: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_538, permute_340);  permute_340 = None
    permute_341: "f32[3072, 512]" = torch.ops.aten.permute.default(view_538, [1, 0])
    mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_341, view_356);  permute_341 = view_356 = None
    permute_342: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_78: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
    view_539: "f32[3072]" = torch.ops.aten.view.default(sum_78, [3072]);  sum_78 = None
    permute_343: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_540: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_178: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_191, view_540);  mul_191 = view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_76: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_117, getitem_99);  add_117 = getitem_99 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_19);  sub_76 = None
    mul_203: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_178, primals_230);  primals_230 = None
    mul_204: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, 768)
    sum_79: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True)
    mul_205: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_203, mul_202);  mul_203 = None
    sum_80: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [2], True);  mul_205 = None
    mul_206: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_202, sum_80);  sum_80 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_204, sum_79);  mul_204 = sum_79 = None
    sub_78: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_206);  sub_77 = mul_206 = None
    div_46: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_207: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_78);  div_46 = sub_78 = None
    mul_208: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_178, mul_202);  mul_202 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_208, [0, 1]);  mul_208 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_178, [0, 1]);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_209: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_210: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_207, mul_209);  mul_209 = None
    clone_49: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_210, memory_format = torch.contiguous_format);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_541: "f32[512, 768]" = torch.ops.aten.view.default(clone_49, [512, 768]);  clone_49 = None
    permute_344: "f32[768, 768]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    mm_52: "f32[512, 768]" = torch.ops.aten.mm.default(view_541, permute_344);  permute_344 = None
    permute_345: "f32[768, 512]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_53: "f32[768, 768]" = torch.ops.aten.mm.default(permute_345, view_354);  permute_345 = view_354 = None
    permute_346: "f32[768, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
    view_542: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    permute_347: "f32[768, 768]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_543: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_52, [1, 512, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_544: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_543, [1, 512, 12, 64]);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_33: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_544, 2, 0, 6)
    slice_34: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_544, 2, 6, 12);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_545: "f32[512, 384]" = torch.ops.aten.view.default(slice_34, [512, 384]);  slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_348: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_33, [0, 2, 1, 3]);  slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_546: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_348, [6, 512, 64]);  permute_348 = None
    permute_349: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    bmm_48: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_349, view_546);  permute_349 = None
    permute_350: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_350, [0, 2, 1]);  view_350 = None
    bmm_49: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_546, permute_350);  view_546 = permute_350 = None
    view_547: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 6, 512, 64]);  bmm_48 = None
    view_548: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 6, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_9: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_211: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_212: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_548, mul_211);  view_548 = mul_211 = None
    clone_50: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_212, memory_format = torch.contiguous_format);  mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_30: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_213: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_50, alias_30);  clone_50 = None
    sum_84: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [-1], True)
    mul_214: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_30, sum_84);  alias_30 = sum_84 = None
    sub_79: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_47: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_79, 8.0);  sub_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_549: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_47, [6, 512, 512]);  div_47 = None
    permute_351: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_346, [0, 2, 1]);  view_346 = None
    bmm_50: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_351, view_549);  permute_351 = None
    permute_352: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1]);  view_347 = None
    bmm_51: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_549, permute_352);  view_549 = permute_352 = None
    view_550: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 6, 64, 512]);  bmm_50 = None
    view_551: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 6, 512, 64]);  bmm_51 = None
    permute_353: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_550, [0, 1, 3, 2]);  view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_51: "f32[512, 384]" = torch.ops.aten.clone.default(view_545, memory_format = torch.contiguous_format);  view_545 = None
    view_552: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_51, [3072, 64, 1]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_553: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_552, [3072, 64, 1]);  view_552 = None
    permute_354: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
    bmm_52: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_354, view_553);  permute_354 = None
    permute_355: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
    bmm_53: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_553, permute_355);  view_553 = permute_355 = None
    view_554: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_52, [3072, 9, 1]);  bmm_52 = None
    view_555: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_53, [3072, 64, 9]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_556: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_555, [1, 512, 384, 9]);  view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_557: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_556, [1, 512, 3456]);  view_556 = None
    permute_356: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_557, [0, 2, 1]);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_558: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_356, [1, 384, 9, 1, 512, 1]);  permute_356 = None
    permute_357: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_558, [0, 1, 2, 4, 3, 5]);  view_558 = None
    iota_56: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_100: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_56, 0);  iota_56 = None
    iota_57: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_101: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_57, -1);  iota_57 = None
    add_179: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_100, unsqueeze_101);  unsqueeze_100 = unsqueeze_101 = None
    unsqueeze_102: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_179, -1);  add_179 = None
    unsqueeze_103: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    iota_58: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_104: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_58, 0);  iota_58 = None
    iota_59: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_105: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_59, -1);  iota_59 = None
    add_180: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_104, unsqueeze_105);  unsqueeze_104 = unsqueeze_105 = None
    full_4: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_4, [None, None, unsqueeze_103, add_180], permute_357, True);  full_4 = unsqueeze_103 = add_180 = permute_357 = None
    constant_pad_nd_14: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_2, [0, 0, -4, -4], 0.0);  _unsafe_index_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_3: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_14, -1);  constant_pad_nd_14 = None
    permute_358: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_3, [0, 2, 1]);  squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_559: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_358, [1, 512, 384]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_560: "f32[512, 384]" = torch.ops.aten.view.default(view_559, [512, 384]);  view_559 = None
    permute_359: "f32[384, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    mm_54: "f32[512, 768]" = torch.ops.aten.mm.default(view_560, permute_359);  permute_359 = None
    permute_360: "f32[384, 512]" = torch.ops.aten.permute.default(view_560, [1, 0])
    mm_55: "f32[384, 768]" = torch.ops.aten.mm.default(permute_360, view_336);  permute_360 = view_336 = None
    permute_361: "f32[768, 384]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_85: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
    view_561: "f32[384]" = torch.ops.aten.view.default(sum_85, [384]);  sum_85 = None
    permute_362: "f32[384, 768]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    view_562: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_54, [1, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_181: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_207, view_562);  mul_207 = view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_31: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_215: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_554, alias_31);  view_554 = None
    sum_86: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [1], True)
    mul_216: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_31, sum_86);  alias_31 = sum_86 = None
    sub_80: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_563: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_80, [1, 512, 54]);  sub_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_87: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1], True)
    view_564: "f32[54]" = torch.ops.aten.view.default(sum_87, [54]);  sum_87 = None
    view_565: "f32[512, 54]" = torch.ops.aten.view.default(view_563, [512, 54]);  view_563 = None
    permute_363: "f32[54, 512]" = torch.ops.aten.permute.default(view_565, [1, 0])
    mm_56: "f32[54, 384]" = torch.ops.aten.mm.default(permute_363, view_333);  permute_363 = view_333 = None
    permute_364: "f32[384, 54]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    permute_365: "f32[54, 512]" = torch.ops.aten.permute.default(view_565, [1, 0]);  view_565 = None
    mm_57: "f32[384, 512]" = torch.ops.aten.mm.default(permute_180, permute_365);  permute_180 = permute_365 = None
    permute_366: "f32[512, 384]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    view_566: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_366, [1, 512, 384]);  permute_366 = None
    permute_367: "f32[54, 384]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_217: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_566, permute_179);  permute_179 = None
    mul_218: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_566, view_325);  view_566 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_368: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_547, [0, 2, 1, 3]);  view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_52: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_567: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_52, [1, 512, 384]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_369: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_353, [0, 2, 1, 3]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_568: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_369, [1, 512, 384]);  permute_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_370: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_53: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_569: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_53, [1, 512, 384]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_182: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_217, view_569);  mul_217 = view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_371: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_218, [0, 2, 1]);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_88: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_371, [0, 2], True)
    view_570: "f32[384, 1]" = torch.ops.aten.view.default(sum_88, [384, 1]);  sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(permute_371, convolution_18, primals_223, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_371 = convolution_18 = primals_223 = None
    getitem_138: "f32[1, 768, 512]" = convolution_backward_4[0]
    getitem_139: "f32[384, 768, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(getitem_138, permute_174, primals_222, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_138 = permute_174 = primals_222 = None
    getitem_141: "f32[1, 768, 512]" = convolution_backward_5[0]
    getitem_142: "f32[768, 1, 9]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_372: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_141, [0, 2, 1]);  getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_183: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_181, permute_372);  add_181 = permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_571: "f32[512, 384]" = torch.ops.aten.view.default(view_567, [512, 384]);  view_567 = None
    permute_373: "f32[384, 768]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_571, permute_373);  permute_373 = None
    permute_374: "f32[384, 512]" = torch.ops.aten.permute.default(view_571, [1, 0])
    mm_59: "f32[384, 768]" = torch.ops.aten.mm.default(permute_374, view_328);  permute_374 = view_328 = None
    permute_375: "f32[768, 384]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_89: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
    view_572: "f32[384]" = torch.ops.aten.view.default(sum_89, [384]);  sum_89 = None
    permute_376: "f32[384, 768]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_573: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_184: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_183, view_573);  add_183 = view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_574: "f32[512, 384]" = torch.ops.aten.view.default(view_568, [512, 384]);  view_568 = None
    permute_377: "f32[384, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_574, permute_377);  permute_377 = None
    permute_378: "f32[384, 512]" = torch.ops.aten.permute.default(view_574, [1, 0])
    mm_61: "f32[384, 768]" = torch.ops.aten.mm.default(permute_378, view_326);  permute_378 = view_326 = None
    permute_379: "f32[768, 384]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_90: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
    view_575: "f32[384]" = torch.ops.aten.view.default(sum_90, [384]);  sum_90 = None
    permute_380: "f32[384, 768]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_576: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_185: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_184, view_576);  add_184 = view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_577: "f32[512, 384]" = torch.ops.aten.view.default(add_182, [512, 384]);  add_182 = None
    permute_381: "f32[384, 768]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    mm_62: "f32[512, 768]" = torch.ops.aten.mm.default(view_577, permute_381);  permute_381 = None
    permute_382: "f32[384, 512]" = torch.ops.aten.permute.default(view_577, [1, 0])
    mm_63: "f32[384, 768]" = torch.ops.aten.mm.default(permute_382, view_324);  permute_382 = view_324 = None
    permute_383: "f32[768, 384]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_91: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
    view_578: "f32[384]" = torch.ops.aten.view.default(sum_91, [384]);  sum_91 = None
    permute_384: "f32[384, 768]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    view_579: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_62, [1, 512, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_186: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_185, view_579);  add_185 = view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_109, getitem_93);  add_109 = getitem_93 = None
    mul_219: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_18);  sub_81 = None
    mul_220: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_186, primals_214);  primals_214 = None
    mul_221: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_220, 768)
    sum_92: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
    mul_222: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_220, mul_219);  mul_220 = None
    sum_93: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
    mul_223: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_219, sum_93);  sum_93 = None
    sub_82: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_221, sum_92);  mul_221 = sum_92 = None
    sub_83: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_82, mul_223);  sub_82 = mul_223 = None
    div_48: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_224: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_83);  div_48 = sub_83 = None
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_186, mul_219);  mul_219 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_186, [0, 1]);  add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, mul_226);  mul_226 = None
    clone_54: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_227, memory_format = torch.contiguous_format);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_580: "f32[512, 768]" = torch.ops.aten.view.default(clone_54, [512, 768]);  clone_54 = None
    permute_385: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    mm_64: "f32[512, 3072]" = torch.ops.aten.mm.default(view_580, permute_385);  permute_385 = None
    permute_386: "f32[768, 512]" = torch.ops.aten.permute.default(view_580, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_386, view_322);  permute_386 = view_322 = None
    permute_387: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
    view_581: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_388: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    view_582: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_64, [1, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_228: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, 0.7071067811865476)
    erf_17: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_228);  mul_228 = None
    add_187: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_229: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_187, 0.5);  add_187 = None
    mul_230: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, view_321)
    mul_231: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_230, -0.5);  mul_230 = None
    exp_30: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_231);  mul_231 = None
    mul_232: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_233: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, mul_232);  view_321 = mul_232 = None
    add_188: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_229, mul_233);  mul_229 = mul_233 = None
    mul_234: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_582, add_188);  view_582 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_583: "f32[512, 3072]" = torch.ops.aten.view.default(mul_234, [512, 3072]);  mul_234 = None
    permute_389: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_583, permute_389);  permute_389 = None
    permute_390: "f32[3072, 512]" = torch.ops.aten.permute.default(view_583, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_390, view_320);  permute_390 = view_320 = None
    permute_391: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_97: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_583, [0], True);  view_583 = None
    view_584: "f32[3072]" = torch.ops.aten.view.default(sum_97, [3072]);  sum_97 = None
    permute_392: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_391, [1, 0]);  permute_391 = None
    view_585: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_189: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_224, view_585);  mul_224 = view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_84: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_105, getitem_89);  add_105 = getitem_89 = None
    mul_235: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_17);  sub_84 = None
    mul_236: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, primals_208);  primals_208 = None
    mul_237: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_236, 768)
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
    mul_238: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_236, mul_235);  mul_236 = None
    sum_99: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
    mul_239: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_235, sum_99);  sum_99 = None
    sub_85: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_237, sum_98);  mul_237 = sum_98 = None
    sub_86: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_85, mul_239);  sub_85 = mul_239 = None
    div_49: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_240: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_86);  div_49 = sub_86 = None
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, mul_235);  mul_235 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_240, mul_242);  mul_242 = None
    clone_55: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_243, memory_format = torch.contiguous_format);  mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_586: "f32[512, 768]" = torch.ops.aten.view.default(clone_55, [512, 768]);  clone_55 = None
    permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_586, permute_393);  permute_393 = None
    permute_394: "f32[768, 512]" = torch.ops.aten.permute.default(view_586, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_318);  permute_394 = view_318 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
    view_587: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_588: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_589: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_588, [1, 512, 12, 64]);  view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_35: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_589, 2, 0, 6)
    slice_36: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_589, 2, 6, 12);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_590: "f32[512, 384]" = torch.ops.aten.view.default(slice_36, [512, 384]);  slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_397: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_35, [0, 2, 1, 3]);  slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_591: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_397, [6, 512, 64]);  permute_397 = None
    permute_398: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    bmm_54: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_398, view_591);  permute_398 = None
    permute_399: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_314, [0, 2, 1]);  view_314 = None
    bmm_55: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_591, permute_399);  view_591 = permute_399 = None
    view_592: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_54, [1, 6, 512, 64]);  bmm_54 = None
    view_593: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_55, [1, 6, 512, 512]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_12: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_244: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_245: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_593, mul_244);  view_593 = mul_244 = None
    clone_56: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_245, memory_format = torch.contiguous_format);  mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_32: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_246: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_56, alias_32);  clone_56 = None
    sum_103: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_246, [-1], True)
    mul_247: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_32, sum_103);  alias_32 = sum_103 = None
    sub_87: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_50: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_87, 8.0);  sub_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_594: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_50, [6, 512, 512]);  div_50 = None
    permute_400: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_310, [0, 2, 1]);  view_310 = None
    bmm_56: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_400, view_594);  permute_400 = None
    permute_401: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
    bmm_57: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_594, permute_401);  view_594 = permute_401 = None
    view_595: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_56, [1, 6, 64, 512]);  bmm_56 = None
    view_596: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_57, [1, 6, 512, 64]);  bmm_57 = None
    permute_402: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_595, [0, 1, 3, 2]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_57: "f32[512, 384]" = torch.ops.aten.clone.default(view_590, memory_format = torch.contiguous_format);  view_590 = None
    view_597: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_57, [3072, 64, 1]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_598: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_597, [3072, 64, 1]);  view_597 = None
    permute_403: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None
    bmm_58: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_403, view_598);  permute_403 = None
    permute_404: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
    bmm_59: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_598, permute_404);  view_598 = permute_404 = None
    view_599: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_58, [3072, 9, 1]);  bmm_58 = None
    view_600: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_59, [3072, 64, 9]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_601: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_600, [1, 512, 384, 9]);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_602: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_601, [1, 512, 3456]);  view_601 = None
    permute_405: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_602, [0, 2, 1]);  view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_603: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_405, [1, 384, 9, 1, 512, 1]);  permute_405 = None
    permute_406: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_603, [0, 1, 2, 4, 3, 5]);  view_603 = None
    iota_60: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_106: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_60, 0);  iota_60 = None
    iota_61: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_107: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_61, -1);  iota_61 = None
    add_190: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_106, unsqueeze_107);  unsqueeze_106 = unsqueeze_107 = None
    unsqueeze_108: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_190, -1);  add_190 = None
    unsqueeze_109: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    iota_62: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_110: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_62, 0);  iota_62 = None
    iota_63: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_111: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_63, -1);  iota_63 = None
    add_191: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_110, unsqueeze_111);  unsqueeze_110 = unsqueeze_111 = None
    full_5: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_3: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_5, [None, None, unsqueeze_109, add_191], permute_406, True);  full_5 = unsqueeze_109 = add_191 = permute_406 = None
    constant_pad_nd_15: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_3, [0, 0, -4, -4], 0.0);  _unsafe_index_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_4: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_15, -1);  constant_pad_nd_15 = None
    permute_407: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_4, [0, 2, 1]);  squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_604: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_407, [1, 512, 384]);  permute_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_605: "f32[512, 384]" = torch.ops.aten.view.default(view_604, [512, 384]);  view_604 = None
    permute_408: "f32[384, 768]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    mm_70: "f32[512, 768]" = torch.ops.aten.mm.default(view_605, permute_408);  permute_408 = None
    permute_409: "f32[384, 512]" = torch.ops.aten.permute.default(view_605, [1, 0])
    mm_71: "f32[384, 768]" = torch.ops.aten.mm.default(permute_409, view_300);  permute_409 = view_300 = None
    permute_410: "f32[768, 384]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_104: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_605, [0], True);  view_605 = None
    view_606: "f32[384]" = torch.ops.aten.view.default(sum_104, [384]);  sum_104 = None
    permute_411: "f32[384, 768]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_607: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_70, [1, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_192: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_240, view_607);  mul_240 = view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_33: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_248: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_599, alias_33);  view_599 = None
    sum_105: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [1], True)
    mul_249: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_33, sum_105);  alias_33 = sum_105 = None
    sub_88: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_608: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_88, [1, 512, 54]);  sub_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_106: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_608, [0, 1], True)
    view_609: "f32[54]" = torch.ops.aten.view.default(sum_106, [54]);  sum_106 = None
    view_610: "f32[512, 54]" = torch.ops.aten.view.default(view_608, [512, 54]);  view_608 = None
    permute_412: "f32[54, 512]" = torch.ops.aten.permute.default(view_610, [1, 0])
    mm_72: "f32[54, 384]" = torch.ops.aten.mm.default(permute_412, view_297);  permute_412 = view_297 = None
    permute_413: "f32[384, 54]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    permute_414: "f32[54, 512]" = torch.ops.aten.permute.default(view_610, [1, 0]);  view_610 = None
    mm_73: "f32[384, 512]" = torch.ops.aten.mm.default(permute_161, permute_414);  permute_161 = permute_414 = None
    permute_415: "f32[512, 384]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    view_611: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_415, [1, 512, 384]);  permute_415 = None
    permute_416: "f32[54, 384]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_250: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_611, permute_160);  permute_160 = None
    mul_251: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_611, view_289);  view_611 = view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_417: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_58: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None
    view_612: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_58, [1, 512, 384]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_418: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_402, [0, 2, 1, 3]);  permute_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_613: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_418, [1, 512, 384]);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_419: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_59: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_614: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_59, [1, 512, 384]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_193: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_250, view_614);  mul_250 = view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_420: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_251, [0, 2, 1]);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_107: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_420, [0, 2], True)
    view_615: "f32[384, 1]" = torch.ops.aten.view.default(sum_107, [384, 1]);  sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(permute_420, convolution_16, primals_201, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_420 = convolution_16 = primals_201 = None
    getitem_144: "f32[1, 768, 512]" = convolution_backward_6[0]
    getitem_145: "f32[384, 768, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(getitem_144, permute_155, primals_200, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_144 = permute_155 = primals_200 = None
    getitem_147: "f32[1, 768, 512]" = convolution_backward_7[0]
    getitem_148: "f32[768, 1, 9]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_421: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1]);  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_194: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_192, permute_421);  add_192 = permute_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_616: "f32[512, 384]" = torch.ops.aten.view.default(view_612, [512, 384]);  view_612 = None
    permute_422: "f32[384, 768]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_616, permute_422);  permute_422 = None
    permute_423: "f32[384, 512]" = torch.ops.aten.permute.default(view_616, [1, 0])
    mm_75: "f32[384, 768]" = torch.ops.aten.mm.default(permute_423, view_292);  permute_423 = view_292 = None
    permute_424: "f32[768, 384]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_108: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_616, [0], True);  view_616 = None
    view_617: "f32[384]" = torch.ops.aten.view.default(sum_108, [384]);  sum_108 = None
    permute_425: "f32[384, 768]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_618: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_74, [1, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_195: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_194, view_618);  add_194 = view_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_619: "f32[512, 384]" = torch.ops.aten.view.default(view_613, [512, 384]);  view_613 = None
    permute_426: "f32[384, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    mm_76: "f32[512, 768]" = torch.ops.aten.mm.default(view_619, permute_426);  permute_426 = None
    permute_427: "f32[384, 512]" = torch.ops.aten.permute.default(view_619, [1, 0])
    mm_77: "f32[384, 768]" = torch.ops.aten.mm.default(permute_427, view_290);  permute_427 = view_290 = None
    permute_428: "f32[768, 384]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_109: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_619, [0], True);  view_619 = None
    view_620: "f32[384]" = torch.ops.aten.view.default(sum_109, [384]);  sum_109 = None
    permute_429: "f32[384, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_621: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_76, [1, 512, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_196: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_195, view_621);  add_195 = view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_622: "f32[512, 384]" = torch.ops.aten.view.default(add_193, [512, 384]);  add_193 = None
    permute_430: "f32[384, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    mm_78: "f32[512, 768]" = torch.ops.aten.mm.default(view_622, permute_430);  permute_430 = None
    permute_431: "f32[384, 512]" = torch.ops.aten.permute.default(view_622, [1, 0])
    mm_79: "f32[384, 768]" = torch.ops.aten.mm.default(permute_431, view_288);  permute_431 = view_288 = None
    permute_432: "f32[768, 384]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_110: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
    view_623: "f32[384]" = torch.ops.aten.view.default(sum_110, [384]);  sum_110 = None
    permute_433: "f32[384, 768]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_624: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_78, [1, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_197: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_196, view_624);  add_196 = view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_89: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_83);  add_97 = getitem_83 = None
    mul_252: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_16);  sub_89 = None
    mul_253: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_197, primals_192);  primals_192 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, 768)
    sum_111: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, mul_252);  mul_253 = None
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    mul_256: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_252, sum_112);  sum_112 = None
    sub_90: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_254, sum_111);  mul_254 = sum_111 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_256);  sub_90 = mul_256 = None
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_91);  div_51 = sub_91 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_197, mul_252);  mul_252 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_260: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, mul_259);  mul_259 = None
    clone_60: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_260, memory_format = torch.contiguous_format);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_625: "f32[512, 768]" = torch.ops.aten.view.default(clone_60, [512, 768]);  clone_60 = None
    permute_434: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    mm_80: "f32[512, 3072]" = torch.ops.aten.mm.default(view_625, permute_434);  permute_434 = None
    permute_435: "f32[768, 512]" = torch.ops.aten.permute.default(view_625, [1, 0])
    mm_81: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_435, view_286);  permute_435 = view_286 = None
    permute_436: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
    view_626: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_437: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_627: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_80, [1, 512, 3072]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_261: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476)
    erf_18: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_261);  mul_261 = None
    add_198: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_262: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_198, 0.5);  add_198 = None
    mul_263: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, view_285)
    mul_264: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
    exp_31: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
    mul_265: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_266: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, mul_265);  view_285 = mul_265 = None
    add_199: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
    mul_267: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_627, add_199);  view_627 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_628: "f32[512, 3072]" = torch.ops.aten.view.default(mul_267, [512, 3072]);  mul_267 = None
    permute_438: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_628, permute_438);  permute_438 = None
    permute_439: "f32[3072, 512]" = torch.ops.aten.permute.default(view_628, [1, 0])
    mm_83: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_439, view_284);  permute_439 = view_284 = None
    permute_440: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_116: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
    view_629: "f32[3072]" = torch.ops.aten.view.default(sum_116, [3072]);  sum_116 = None
    permute_441: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_630: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_200: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_257, view_630);  mul_257 = view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_92: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_93, getitem_79);  add_93 = getitem_79 = None
    mul_268: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_15);  sub_92 = None
    mul_269: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_200, primals_186);  primals_186 = None
    mul_270: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_269, 768)
    sum_117: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_269, mul_268);  mul_269 = None
    sum_118: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_268, sum_118);  sum_118 = None
    sub_93: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_270, sum_117);  mul_270 = sum_117 = None
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_93, mul_272);  sub_93 = mul_272 = None
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_94);  div_52 = sub_94 = None
    mul_274: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_200, mul_268);  mul_268 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_200, [0, 1]);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_275: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_276: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_273, mul_275);  mul_275 = None
    clone_61: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_276, memory_format = torch.contiguous_format);  mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_631: "f32[512, 768]" = torch.ops.aten.view.default(clone_61, [512, 768]);  clone_61 = None
    permute_442: "f32[768, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_631, permute_442);  permute_442 = None
    permute_443: "f32[768, 512]" = torch.ops.aten.permute.default(view_631, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_443, view_282);  permute_443 = view_282 = None
    permute_444: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
    view_632: "f32[768]" = torch.ops.aten.view.default(sum_121, [768]);  sum_121 = None
    permute_445: "f32[768, 768]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    view_633: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_634: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_633, [1, 512, 12, 64]);  view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_37: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_634, 2, 0, 6)
    slice_38: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_634, 2, 6, 12);  view_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_635: "f32[512, 384]" = torch.ops.aten.view.default(slice_38, [512, 384]);  slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_446: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_37, [0, 2, 1, 3]);  slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_636: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_446, [6, 512, 64]);  permute_446 = None
    permute_447: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
    bmm_60: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_447, view_636);  permute_447 = None
    permute_448: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_278, [0, 2, 1]);  view_278 = None
    bmm_61: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_636, permute_448);  view_636 = permute_448 = None
    view_637: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 6, 512, 64]);  bmm_60 = None
    view_638: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 6, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_15: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_277: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_278: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_638, mul_277);  view_638 = mul_277 = None
    clone_62: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_278, memory_format = torch.contiguous_format);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_34: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_279: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_62, alias_34);  clone_62 = None
    sum_122: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [-1], True)
    mul_280: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_34, sum_122);  alias_34 = sum_122 = None
    sub_95: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_53: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_95, 8.0);  sub_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_639: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_53, [6, 512, 512]);  div_53 = None
    permute_449: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
    bmm_62: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_449, view_639);  permute_449 = None
    permute_450: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
    bmm_63: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_639, permute_450);  view_639 = permute_450 = None
    view_640: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 6, 64, 512]);  bmm_62 = None
    view_641: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 6, 512, 64]);  bmm_63 = None
    permute_451: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_640, [0, 1, 3, 2]);  view_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_63: "f32[512, 384]" = torch.ops.aten.clone.default(view_635, memory_format = torch.contiguous_format);  view_635 = None
    view_642: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_63, [3072, 64, 1]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_643: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_642, [3072, 64, 1]);  view_642 = None
    permute_452: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
    bmm_64: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_452, view_643);  permute_452 = None
    permute_453: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
    bmm_65: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_643, permute_453);  view_643 = permute_453 = None
    view_644: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_64, [3072, 9, 1]);  bmm_64 = None
    view_645: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_65, [3072, 64, 9]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_646: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_645, [1, 512, 384, 9]);  view_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_647: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_646, [1, 512, 3456]);  view_646 = None
    permute_454: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_647, [0, 2, 1]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_648: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_454, [1, 384, 9, 1, 512, 1]);  permute_454 = None
    permute_455: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_648, [0, 1, 2, 4, 3, 5]);  view_648 = None
    iota_64: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_112: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_64, 0);  iota_64 = None
    iota_65: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_113: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_65, -1);  iota_65 = None
    add_201: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_112, unsqueeze_113);  unsqueeze_112 = unsqueeze_113 = None
    unsqueeze_114: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_201, -1);  add_201 = None
    unsqueeze_115: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    iota_66: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_116: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_66, 0);  iota_66 = None
    iota_67: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_117: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_67, -1);  iota_67 = None
    add_202: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_116, unsqueeze_117);  unsqueeze_116 = unsqueeze_117 = None
    full_6: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_4: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_6, [None, None, unsqueeze_115, add_202], permute_455, True);  full_6 = unsqueeze_115 = add_202 = permute_455 = None
    constant_pad_nd_16: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_4, [0, 0, -4, -4], 0.0);  _unsafe_index_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_5: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_16, -1);  constant_pad_nd_16 = None
    permute_456: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_5, [0, 2, 1]);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_649: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_456, [1, 512, 384]);  permute_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_650: "f32[512, 384]" = torch.ops.aten.view.default(view_649, [512, 384]);  view_649 = None
    permute_457: "f32[384, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_86: "f32[512, 768]" = torch.ops.aten.mm.default(view_650, permute_457);  permute_457 = None
    permute_458: "f32[384, 512]" = torch.ops.aten.permute.default(view_650, [1, 0])
    mm_87: "f32[384, 768]" = torch.ops.aten.mm.default(permute_458, view_264);  permute_458 = view_264 = None
    permute_459: "f32[768, 384]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_123: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
    view_651: "f32[384]" = torch.ops.aten.view.default(sum_123, [384]);  sum_123 = None
    permute_460: "f32[384, 768]" = torch.ops.aten.permute.default(permute_459, [1, 0]);  permute_459 = None
    view_652: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_86, [1, 512, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_203: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_273, view_652);  mul_273 = view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_35: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_281: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_644, alias_35);  view_644 = None
    sum_124: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [1], True)
    mul_282: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_35, sum_124);  alias_35 = sum_124 = None
    sub_96: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_653: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_96, [1, 512, 54]);  sub_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_125: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_653, [0, 1], True)
    view_654: "f32[54]" = torch.ops.aten.view.default(sum_125, [54]);  sum_125 = None
    view_655: "f32[512, 54]" = torch.ops.aten.view.default(view_653, [512, 54]);  view_653 = None
    permute_461: "f32[54, 512]" = torch.ops.aten.permute.default(view_655, [1, 0])
    mm_88: "f32[54, 384]" = torch.ops.aten.mm.default(permute_461, view_261);  permute_461 = view_261 = None
    permute_462: "f32[384, 54]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    permute_463: "f32[54, 512]" = torch.ops.aten.permute.default(view_655, [1, 0]);  view_655 = None
    mm_89: "f32[384, 512]" = torch.ops.aten.mm.default(permute_142, permute_463);  permute_142 = permute_463 = None
    permute_464: "f32[512, 384]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    view_656: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_464, [1, 512, 384]);  permute_464 = None
    permute_465: "f32[54, 384]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_283: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_656, permute_141);  permute_141 = None
    mul_284: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_656, view_253);  view_656 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_466: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_637, [0, 2, 1, 3]);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_64: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    view_657: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_64, [1, 512, 384]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_467: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_451, [0, 2, 1, 3]);  permute_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_658: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_467, [1, 512, 384]);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_468: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_641, [0, 2, 1, 3]);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_65: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    view_659: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_65, [1, 512, 384]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_204: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_283, view_659);  mul_283 = view_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_469: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_284, [0, 2, 1]);  mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_126: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_469, [0, 2], True)
    view_660: "f32[384, 1]" = torch.ops.aten.view.default(sum_126, [384, 1]);  sum_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(permute_469, convolution_14, primals_179, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_469 = convolution_14 = primals_179 = None
    getitem_150: "f32[1, 768, 512]" = convolution_backward_8[0]
    getitem_151: "f32[384, 768, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(getitem_150, permute_136, primals_178, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_150 = permute_136 = primals_178 = None
    getitem_153: "f32[1, 768, 512]" = convolution_backward_9[0]
    getitem_154: "f32[768, 1, 9]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_470: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1]);  getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_205: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_203, permute_470);  add_203 = permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_661: "f32[512, 384]" = torch.ops.aten.view.default(view_657, [512, 384]);  view_657 = None
    permute_471: "f32[384, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_661, permute_471);  permute_471 = None
    permute_472: "f32[384, 512]" = torch.ops.aten.permute.default(view_661, [1, 0])
    mm_91: "f32[384, 768]" = torch.ops.aten.mm.default(permute_472, view_256);  permute_472 = view_256 = None
    permute_473: "f32[768, 384]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_127: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_661, [0], True);  view_661 = None
    view_662: "f32[384]" = torch.ops.aten.view.default(sum_127, [384]);  sum_127 = None
    permute_474: "f32[384, 768]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_663: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_206: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_205, view_663);  add_205 = view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_664: "f32[512, 384]" = torch.ops.aten.view.default(view_658, [512, 384]);  view_658 = None
    permute_475: "f32[384, 768]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_664, permute_475);  permute_475 = None
    permute_476: "f32[384, 512]" = torch.ops.aten.permute.default(view_664, [1, 0])
    mm_93: "f32[384, 768]" = torch.ops.aten.mm.default(permute_476, view_254);  permute_476 = view_254 = None
    permute_477: "f32[768, 384]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_128: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_664, [0], True);  view_664 = None
    view_665: "f32[384]" = torch.ops.aten.view.default(sum_128, [384]);  sum_128 = None
    permute_478: "f32[384, 768]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_666: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_207: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_206, view_666);  add_206 = view_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_667: "f32[512, 384]" = torch.ops.aten.view.default(add_204, [512, 384]);  add_204 = None
    permute_479: "f32[384, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_94: "f32[512, 768]" = torch.ops.aten.mm.default(view_667, permute_479);  permute_479 = None
    permute_480: "f32[384, 512]" = torch.ops.aten.permute.default(view_667, [1, 0])
    mm_95: "f32[384, 768]" = torch.ops.aten.mm.default(permute_480, view_252);  permute_480 = view_252 = None
    permute_481: "f32[768, 384]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_129: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_667, [0], True);  view_667 = None
    view_668: "f32[384]" = torch.ops.aten.view.default(sum_129, [384]);  sum_129 = None
    permute_482: "f32[384, 768]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    view_669: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_94, [1, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_208: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_207, view_669);  add_207 = view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_97: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_73);  add_85 = getitem_73 = None
    mul_285: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_14);  sub_97 = None
    mul_286: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_208, primals_170);  primals_170 = None
    mul_287: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_286, 768)
    sum_130: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
    mul_288: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_286, mul_285);  mul_286 = None
    sum_131: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
    mul_289: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_285, sum_131);  sum_131 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_287, sum_130);  mul_287 = sum_130 = None
    sub_99: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_289);  sub_98 = mul_289 = None
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_290: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_99);  div_54 = sub_99 = None
    mul_291: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_208, mul_285);  mul_285 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_208, [0, 1]);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_292: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_293: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_290, mul_292);  mul_292 = None
    clone_66: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_293, memory_format = torch.contiguous_format);  mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_670: "f32[512, 768]" = torch.ops.aten.view.default(clone_66, [512, 768]);  clone_66 = None
    permute_483: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_96: "f32[512, 3072]" = torch.ops.aten.mm.default(view_670, permute_483);  permute_483 = None
    permute_484: "f32[768, 512]" = torch.ops.aten.permute.default(view_670, [1, 0])
    mm_97: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_484, view_250);  permute_484 = view_250 = None
    permute_485: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
    view_671: "f32[768]" = torch.ops.aten.view.default(sum_134, [768]);  sum_134 = None
    permute_486: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_485, [1, 0]);  permute_485 = None
    view_672: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_96, [1, 512, 3072]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_294: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476)
    erf_19: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_294);  mul_294 = None
    add_209: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_295: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_209, 0.5);  add_209 = None
    mul_296: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, view_249)
    mul_297: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_296, -0.5);  mul_296 = None
    exp_32: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_297);  mul_297 = None
    mul_298: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_299: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, mul_298);  view_249 = mul_298 = None
    add_210: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_295, mul_299);  mul_295 = mul_299 = None
    mul_300: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_672, add_210);  view_672 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_673: "f32[512, 3072]" = torch.ops.aten.view.default(mul_300, [512, 3072]);  mul_300 = None
    permute_487: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_673, permute_487);  permute_487 = None
    permute_488: "f32[3072, 512]" = torch.ops.aten.permute.default(view_673, [1, 0])
    mm_99: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_488, view_248);  permute_488 = view_248 = None
    permute_489: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_135: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_673, [0], True);  view_673 = None
    view_674: "f32[3072]" = torch.ops.aten.view.default(sum_135, [3072]);  sum_135 = None
    permute_490: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    view_675: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_98, [1, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_211: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_290, view_675);  mul_290 = view_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_100: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_69);  add_81 = getitem_69 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_13);  sub_100 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_211, primals_164);  primals_164 = None
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_302, 768)
    sum_136: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True)
    mul_304: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_302, mul_301);  mul_302 = None
    sum_137: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
    mul_305: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_301, sum_137);  sum_137 = None
    sub_101: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_303, sum_136);  mul_303 = sum_136 = None
    sub_102: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_305);  sub_101 = mul_305 = None
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_306: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_102);  div_55 = sub_102 = None
    mul_307: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_211, mul_301);  mul_301 = None
    sum_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
    sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_211, [0, 1]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_308: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_309: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_306, mul_308);  mul_308 = None
    clone_67: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_309, memory_format = torch.contiguous_format);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_676: "f32[512, 768]" = torch.ops.aten.view.default(clone_67, [512, 768]);  clone_67 = None
    permute_491: "f32[768, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_100: "f32[512, 768]" = torch.ops.aten.mm.default(view_676, permute_491);  permute_491 = None
    permute_492: "f32[768, 512]" = torch.ops.aten.permute.default(view_676, [1, 0])
    mm_101: "f32[768, 768]" = torch.ops.aten.mm.default(permute_492, view_246);  permute_492 = view_246 = None
    permute_493: "f32[768, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_676, [0], True);  view_676 = None
    view_677: "f32[768]" = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(permute_493, [1, 0]);  permute_493 = None
    view_678: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_100, [1, 512, 768]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_679: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_678, [1, 512, 12, 64]);  view_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_39: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_679, 2, 0, 6)
    slice_40: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_679, 2, 6, 12);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_680: "f32[512, 384]" = torch.ops.aten.view.default(slice_40, [512, 384]);  slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_495: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_39, [0, 2, 1, 3]);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_681: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_495, [6, 512, 64]);  permute_495 = None
    permute_496: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
    bmm_66: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_496, view_681);  permute_496 = None
    permute_497: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None
    bmm_67: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_681, permute_497);  view_681 = permute_497 = None
    view_682: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_66, [1, 6, 512, 64]);  bmm_66 = None
    view_683: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_67, [1, 6, 512, 512]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_18: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_310: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_311: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_683, mul_310);  view_683 = mul_310 = None
    clone_68: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_311, memory_format = torch.contiguous_format);  mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_36: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_312: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_68, alias_36);  clone_68 = None
    sum_141: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [-1], True)
    mul_313: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_36, sum_141);  alias_36 = sum_141 = None
    sub_103: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_56: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_103, 8.0);  sub_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_684: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_56, [6, 512, 512]);  div_56 = None
    permute_498: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
    bmm_68: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_498, view_684);  permute_498 = None
    permute_499: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
    bmm_69: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_684, permute_499);  view_684 = permute_499 = None
    view_685: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_68, [1, 6, 64, 512]);  bmm_68 = None
    view_686: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_69, [1, 6, 512, 64]);  bmm_69 = None
    permute_500: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_685, [0, 1, 3, 2]);  view_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_69: "f32[512, 384]" = torch.ops.aten.clone.default(view_680, memory_format = torch.contiguous_format);  view_680 = None
    view_687: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_69, [3072, 64, 1]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_688: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_687, [3072, 64, 1]);  view_687 = None
    permute_501: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
    bmm_70: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_501, view_688);  permute_501 = None
    permute_502: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
    bmm_71: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_688, permute_502);  view_688 = permute_502 = None
    view_689: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_70, [3072, 9, 1]);  bmm_70 = None
    view_690: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_71, [3072, 64, 9]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_691: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_690, [1, 512, 384, 9]);  view_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_692: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_691, [1, 512, 3456]);  view_691 = None
    permute_503: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_692, [0, 2, 1]);  view_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_693: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_503, [1, 384, 9, 1, 512, 1]);  permute_503 = None
    permute_504: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_693, [0, 1, 2, 4, 3, 5]);  view_693 = None
    iota_68: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_118: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_68, 0);  iota_68 = None
    iota_69: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_119: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_69, -1);  iota_69 = None
    add_212: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_118, unsqueeze_119);  unsqueeze_118 = unsqueeze_119 = None
    unsqueeze_120: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_212, -1);  add_212 = None
    unsqueeze_121: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    iota_70: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_122: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_70, 0);  iota_70 = None
    iota_71: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_123: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_71, -1);  iota_71 = None
    add_213: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_122, unsqueeze_123);  unsqueeze_122 = unsqueeze_123 = None
    full_7: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_5: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_7, [None, None, unsqueeze_121, add_213], permute_504, True);  full_7 = unsqueeze_121 = add_213 = permute_504 = None
    constant_pad_nd_17: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_5, [0, 0, -4, -4], 0.0);  _unsafe_index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_6: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_17, -1);  constant_pad_nd_17 = None
    permute_505: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_6, [0, 2, 1]);  squeeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_694: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_505, [1, 512, 384]);  permute_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_695: "f32[512, 384]" = torch.ops.aten.view.default(view_694, [512, 384]);  view_694 = None
    permute_506: "f32[384, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_102: "f32[512, 768]" = torch.ops.aten.mm.default(view_695, permute_506);  permute_506 = None
    permute_507: "f32[384, 512]" = torch.ops.aten.permute.default(view_695, [1, 0])
    mm_103: "f32[384, 768]" = torch.ops.aten.mm.default(permute_507, view_228);  permute_507 = view_228 = None
    permute_508: "f32[768, 384]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_142: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
    view_696: "f32[384]" = torch.ops.aten.view.default(sum_142, [384]);  sum_142 = None
    permute_509: "f32[384, 768]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_697: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_102, [1, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_214: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_306, view_697);  mul_306 = view_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_37: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_314: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_689, alias_37);  view_689 = None
    sum_143: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [1], True)
    mul_315: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_37, sum_143);  alias_37 = sum_143 = None
    sub_104: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_698: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_104, [1, 512, 54]);  sub_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_144: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_698, [0, 1], True)
    view_699: "f32[54]" = torch.ops.aten.view.default(sum_144, [54]);  sum_144 = None
    view_700: "f32[512, 54]" = torch.ops.aten.view.default(view_698, [512, 54]);  view_698 = None
    permute_510: "f32[54, 512]" = torch.ops.aten.permute.default(view_700, [1, 0])
    mm_104: "f32[54, 384]" = torch.ops.aten.mm.default(permute_510, view_225);  permute_510 = view_225 = None
    permute_511: "f32[384, 54]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    permute_512: "f32[54, 512]" = torch.ops.aten.permute.default(view_700, [1, 0]);  view_700 = None
    mm_105: "f32[384, 512]" = torch.ops.aten.mm.default(permute_123, permute_512);  permute_123 = permute_512 = None
    permute_513: "f32[512, 384]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    view_701: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_513, [1, 512, 384]);  permute_513 = None
    permute_514: "f32[54, 384]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_316: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_701, permute_122);  permute_122 = None
    mul_317: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_701, view_217);  view_701 = view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_515: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_70: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_515, memory_format = torch.contiguous_format);  permute_515 = None
    view_702: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_70, [1, 512, 384]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_516: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_500, [0, 2, 1, 3]);  permute_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_703: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_516, [1, 512, 384]);  permute_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_517: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_686, [0, 2, 1, 3]);  view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_71: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
    view_704: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_71, [1, 512, 384]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_215: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_316, view_704);  mul_316 = view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_518: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_317, [0, 2, 1]);  mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_145: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_518, [0, 2], True)
    view_705: "f32[384, 1]" = torch.ops.aten.view.default(sum_145, [384, 1]);  sum_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(permute_518, convolution_12, primals_157, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_518 = convolution_12 = primals_157 = None
    getitem_156: "f32[1, 768, 512]" = convolution_backward_10[0]
    getitem_157: "f32[384, 768, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(getitem_156, permute_117, primals_156, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_156 = permute_117 = primals_156 = None
    getitem_159: "f32[1, 768, 512]" = convolution_backward_11[0]
    getitem_160: "f32[768, 1, 9]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_519: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1]);  getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_216: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_214, permute_519);  add_214 = permute_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_706: "f32[512, 384]" = torch.ops.aten.view.default(view_702, [512, 384]);  view_702 = None
    permute_520: "f32[384, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_706, permute_520);  permute_520 = None
    permute_521: "f32[384, 512]" = torch.ops.aten.permute.default(view_706, [1, 0])
    mm_107: "f32[384, 768]" = torch.ops.aten.mm.default(permute_521, view_220);  permute_521 = view_220 = None
    permute_522: "f32[768, 384]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_146: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
    view_707: "f32[384]" = torch.ops.aten.view.default(sum_146, [384]);  sum_146 = None
    permute_523: "f32[384, 768]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    view_708: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_217: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_216, view_708);  add_216 = view_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_709: "f32[512, 384]" = torch.ops.aten.view.default(view_703, [512, 384]);  view_703 = None
    permute_524: "f32[384, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_709, permute_524);  permute_524 = None
    permute_525: "f32[384, 512]" = torch.ops.aten.permute.default(view_709, [1, 0])
    mm_109: "f32[384, 768]" = torch.ops.aten.mm.default(permute_525, view_218);  permute_525 = view_218 = None
    permute_526: "f32[768, 384]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_147: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
    view_710: "f32[384]" = torch.ops.aten.view.default(sum_147, [384]);  sum_147 = None
    permute_527: "f32[384, 768]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    view_711: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_218: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_217, view_711);  add_217 = view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_712: "f32[512, 384]" = torch.ops.aten.view.default(add_215, [512, 384]);  add_215 = None
    permute_528: "f32[384, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    mm_110: "f32[512, 768]" = torch.ops.aten.mm.default(view_712, permute_528);  permute_528 = None
    permute_529: "f32[384, 512]" = torch.ops.aten.permute.default(view_712, [1, 0])
    mm_111: "f32[384, 768]" = torch.ops.aten.mm.default(permute_529, view_216);  permute_529 = view_216 = None
    permute_530: "f32[768, 384]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_148: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
    view_713: "f32[384]" = torch.ops.aten.view.default(sum_148, [384]);  sum_148 = None
    permute_531: "f32[384, 768]" = torch.ops.aten.permute.default(permute_530, [1, 0]);  permute_530 = None
    view_714: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_110, [1, 512, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_219: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_218, view_714);  add_218 = view_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_63);  add_73 = getitem_63 = None
    mul_318: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_12);  sub_105 = None
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, primals_148);  primals_148 = None
    mul_320: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_319, 768)
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
    mul_321: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_319, mul_318);  mul_319 = None
    sum_150: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
    mul_322: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_318, sum_150);  sum_150 = None
    sub_106: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_320, sum_149);  mul_320 = sum_149 = None
    sub_107: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_106, mul_322);  sub_106 = mul_322 = None
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_323: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_107);  div_57 = sub_107 = None
    mul_324: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, mul_318);  mul_318 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_219, [0, 1]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_325: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_323, mul_325);  mul_325 = None
    clone_72: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_326, memory_format = torch.contiguous_format);  mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_715: "f32[512, 768]" = torch.ops.aten.view.default(clone_72, [512, 768]);  clone_72 = None
    permute_532: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_112: "f32[512, 3072]" = torch.ops.aten.mm.default(view_715, permute_532);  permute_532 = None
    permute_533: "f32[768, 512]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_533, view_214);  permute_533 = view_214 = None
    permute_534: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
    view_716: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    permute_535: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_534, [1, 0]);  permute_534 = None
    view_717: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_112, [1, 512, 3072]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_327: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_20: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_327);  mul_327 = None
    add_220: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_328: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_220, 0.5);  add_220 = None
    mul_329: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, view_213)
    mul_330: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_329, -0.5);  mul_329 = None
    exp_33: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_330);  mul_330 = None
    mul_331: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_332: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, mul_331);  view_213 = mul_331 = None
    add_221: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_328, mul_332);  mul_328 = mul_332 = None
    mul_333: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_717, add_221);  view_717 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_718: "f32[512, 3072]" = torch.ops.aten.view.default(mul_333, [512, 3072]);  mul_333 = None
    permute_536: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_114: "f32[512, 768]" = torch.ops.aten.mm.default(view_718, permute_536);  permute_536 = None
    permute_537: "f32[3072, 512]" = torch.ops.aten.permute.default(view_718, [1, 0])
    mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_537, view_212);  permute_537 = view_212 = None
    permute_538: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_154: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_718, [0], True);  view_718 = None
    view_719: "f32[3072]" = torch.ops.aten.view.default(sum_154, [3072]);  sum_154 = None
    permute_539: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_720: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_114, [1, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_222: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_323, view_720);  mul_323 = view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_108: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_59);  add_69 = getitem_59 = None
    mul_334: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_11);  sub_108 = None
    mul_335: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, primals_142);  primals_142 = None
    mul_336: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_335, 768)
    sum_155: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True)
    mul_337: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_335, mul_334);  mul_335 = None
    sum_156: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True);  mul_337 = None
    mul_338: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_334, sum_156);  sum_156 = None
    sub_109: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_336, sum_155);  mul_336 = sum_155 = None
    sub_110: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_109, mul_338);  sub_109 = mul_338 = None
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_339: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_110);  div_58 = sub_110 = None
    mul_340: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, mul_334);  mul_334 = None
    sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1]);  mul_340 = None
    sum_158: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 1]);  add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_339, mul_341);  mul_341 = None
    clone_73: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_342, memory_format = torch.contiguous_format);  mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_721: "f32[512, 768]" = torch.ops.aten.view.default(clone_73, [512, 768]);  clone_73 = None
    permute_540: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_116: "f32[512, 768]" = torch.ops.aten.mm.default(view_721, permute_540);  permute_540 = None
    permute_541: "f32[768, 512]" = torch.ops.aten.permute.default(view_721, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_541, view_210);  permute_541 = view_210 = None
    permute_542: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_159: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_721, [0], True);  view_721 = None
    view_722: "f32[768]" = torch.ops.aten.view.default(sum_159, [768]);  sum_159 = None
    permute_543: "f32[768, 768]" = torch.ops.aten.permute.default(permute_542, [1, 0]);  permute_542 = None
    view_723: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_116, [1, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_724: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_723, [1, 512, 12, 64]);  view_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_41: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_724, 2, 0, 6)
    slice_42: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_724, 2, 6, 12);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_725: "f32[512, 384]" = torch.ops.aten.view.default(slice_42, [512, 384]);  slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_544: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_41, [0, 2, 1, 3]);  slice_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_726: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_544, [6, 512, 64]);  permute_544 = None
    permute_545: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    bmm_72: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_545, view_726);  permute_545 = None
    permute_546: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    bmm_73: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_726, permute_546);  view_726 = permute_546 = None
    view_727: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_72, [1, 6, 512, 64]);  bmm_72 = None
    view_728: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_73, [1, 6, 512, 512]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_21: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_343: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_344: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_728, mul_343);  view_728 = mul_343 = None
    clone_74: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_344, memory_format = torch.contiguous_format);  mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_38: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_345: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_74, alias_38);  clone_74 = None
    sum_160: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [-1], True)
    mul_346: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_38, sum_160);  alias_38 = sum_160 = None
    sub_111: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_345, mul_346);  mul_345 = mul_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_59: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_111, 8.0);  sub_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_729: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_59, [6, 512, 512]);  div_59 = None
    permute_547: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_202, [0, 2, 1]);  view_202 = None
    bmm_74: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_547, view_729);  permute_547 = None
    permute_548: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
    bmm_75: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_729, permute_548);  view_729 = permute_548 = None
    view_730: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_74, [1, 6, 64, 512]);  bmm_74 = None
    view_731: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_75, [1, 6, 512, 64]);  bmm_75 = None
    permute_549: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_730, [0, 1, 3, 2]);  view_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_75: "f32[512, 384]" = torch.ops.aten.clone.default(view_725, memory_format = torch.contiguous_format);  view_725 = None
    view_732: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_75, [3072, 64, 1]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_733: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_732, [3072, 64, 1]);  view_732 = None
    permute_550: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_198, [0, 2, 1]);  view_198 = None
    bmm_76: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_550, view_733);  permute_550 = None
    permute_551: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
    bmm_77: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_733, permute_551);  view_733 = permute_551 = None
    view_734: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_76, [3072, 9, 1]);  bmm_76 = None
    view_735: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_77, [3072, 64, 9]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_736: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_735, [1, 512, 384, 9]);  view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_737: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_736, [1, 512, 3456]);  view_736 = None
    permute_552: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_737, [0, 2, 1]);  view_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_738: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_552, [1, 384, 9, 1, 512, 1]);  permute_552 = None
    permute_553: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_738, [0, 1, 2, 4, 3, 5]);  view_738 = None
    iota_72: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_124: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_72, 0);  iota_72 = None
    iota_73: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_125: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_73, -1);  iota_73 = None
    add_223: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_124, unsqueeze_125);  unsqueeze_124 = unsqueeze_125 = None
    unsqueeze_126: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_223, -1);  add_223 = None
    unsqueeze_127: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    iota_74: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_128: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_74, 0);  iota_74 = None
    iota_75: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_129: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_75, -1);  iota_75 = None
    add_224: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_128, unsqueeze_129);  unsqueeze_128 = unsqueeze_129 = None
    full_8: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_6: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_8, [None, None, unsqueeze_127, add_224], permute_553, True);  full_8 = unsqueeze_127 = add_224 = permute_553 = None
    constant_pad_nd_18: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_6, [0, 0, -4, -4], 0.0);  _unsafe_index_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_7: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_18, -1);  constant_pad_nd_18 = None
    permute_554: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_7, [0, 2, 1]);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_739: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_554, [1, 512, 384]);  permute_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_740: "f32[512, 384]" = torch.ops.aten.view.default(view_739, [512, 384]);  view_739 = None
    permute_555: "f32[384, 768]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_118: "f32[512, 768]" = torch.ops.aten.mm.default(view_740, permute_555);  permute_555 = None
    permute_556: "f32[384, 512]" = torch.ops.aten.permute.default(view_740, [1, 0])
    mm_119: "f32[384, 768]" = torch.ops.aten.mm.default(permute_556, view_192);  permute_556 = view_192 = None
    permute_557: "f32[768, 384]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_161: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
    view_741: "f32[384]" = torch.ops.aten.view.default(sum_161, [384]);  sum_161 = None
    permute_558: "f32[384, 768]" = torch.ops.aten.permute.default(permute_557, [1, 0]);  permute_557 = None
    view_742: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_118, [1, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_225: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_339, view_742);  mul_339 = view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_39: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_347: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_734, alias_39);  view_734 = None
    sum_162: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [1], True)
    mul_348: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_39, sum_162);  alias_39 = sum_162 = None
    sub_112: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_743: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_112, [1, 512, 54]);  sub_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_163: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_743, [0, 1], True)
    view_744: "f32[54]" = torch.ops.aten.view.default(sum_163, [54]);  sum_163 = None
    view_745: "f32[512, 54]" = torch.ops.aten.view.default(view_743, [512, 54]);  view_743 = None
    permute_559: "f32[54, 512]" = torch.ops.aten.permute.default(view_745, [1, 0])
    mm_120: "f32[54, 384]" = torch.ops.aten.mm.default(permute_559, view_189);  permute_559 = view_189 = None
    permute_560: "f32[384, 54]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    permute_561: "f32[54, 512]" = torch.ops.aten.permute.default(view_745, [1, 0]);  view_745 = None
    mm_121: "f32[384, 512]" = torch.ops.aten.mm.default(permute_104, permute_561);  permute_104 = permute_561 = None
    permute_562: "f32[512, 384]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    view_746: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_562, [1, 512, 384]);  permute_562 = None
    permute_563: "f32[54, 384]" = torch.ops.aten.permute.default(permute_560, [1, 0]);  permute_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_349: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_746, permute_103);  permute_103 = None
    mul_350: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_746, view_181);  view_746 = view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_564: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_727, [0, 2, 1, 3]);  view_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_76: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_564, memory_format = torch.contiguous_format);  permute_564 = None
    view_747: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_76, [1, 512, 384]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_565: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_549, [0, 2, 1, 3]);  permute_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_748: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_565, [1, 512, 384]);  permute_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_566: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_731, [0, 2, 1, 3]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_77: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
    view_749: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_77, [1, 512, 384]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_226: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_349, view_749);  mul_349 = view_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_567: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_350, [0, 2, 1]);  mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_164: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_567, [0, 2], True)
    view_750: "f32[384, 1]" = torch.ops.aten.view.default(sum_164, [384, 1]);  sum_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(permute_567, convolution_10, primals_135, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_567 = convolution_10 = primals_135 = None
    getitem_162: "f32[1, 768, 512]" = convolution_backward_12[0]
    getitem_163: "f32[384, 768, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(getitem_162, permute_98, primals_134, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_162 = permute_98 = primals_134 = None
    getitem_165: "f32[1, 768, 512]" = convolution_backward_13[0]
    getitem_166: "f32[768, 1, 9]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_568: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1]);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_227: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_225, permute_568);  add_225 = permute_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_751: "f32[512, 384]" = torch.ops.aten.view.default(view_747, [512, 384]);  view_747 = None
    permute_569: "f32[384, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_122: "f32[512, 768]" = torch.ops.aten.mm.default(view_751, permute_569);  permute_569 = None
    permute_570: "f32[384, 512]" = torch.ops.aten.permute.default(view_751, [1, 0])
    mm_123: "f32[384, 768]" = torch.ops.aten.mm.default(permute_570, view_184);  permute_570 = view_184 = None
    permute_571: "f32[768, 384]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_165: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_751, [0], True);  view_751 = None
    view_752: "f32[384]" = torch.ops.aten.view.default(sum_165, [384]);  sum_165 = None
    permute_572: "f32[384, 768]" = torch.ops.aten.permute.default(permute_571, [1, 0]);  permute_571 = None
    view_753: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_122, [1, 512, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_228: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_227, view_753);  add_227 = view_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_754: "f32[512, 384]" = torch.ops.aten.view.default(view_748, [512, 384]);  view_748 = None
    permute_573: "f32[384, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_124: "f32[512, 768]" = torch.ops.aten.mm.default(view_754, permute_573);  permute_573 = None
    permute_574: "f32[384, 512]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_125: "f32[384, 768]" = torch.ops.aten.mm.default(permute_574, view_182);  permute_574 = view_182 = None
    permute_575: "f32[768, 384]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_166: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_754, [0], True);  view_754 = None
    view_755: "f32[384]" = torch.ops.aten.view.default(sum_166, [384]);  sum_166 = None
    permute_576: "f32[384, 768]" = torch.ops.aten.permute.default(permute_575, [1, 0]);  permute_575 = None
    view_756: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_124, [1, 512, 768]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_229: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_228, view_756);  add_228 = view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_757: "f32[512, 384]" = torch.ops.aten.view.default(add_226, [512, 384]);  add_226 = None
    permute_577: "f32[384, 768]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    mm_126: "f32[512, 768]" = torch.ops.aten.mm.default(view_757, permute_577);  permute_577 = None
    permute_578: "f32[384, 512]" = torch.ops.aten.permute.default(view_757, [1, 0])
    mm_127: "f32[384, 768]" = torch.ops.aten.mm.default(permute_578, view_180);  permute_578 = view_180 = None
    permute_579: "f32[768, 384]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_167: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_757, [0], True);  view_757 = None
    view_758: "f32[384]" = torch.ops.aten.view.default(sum_167, [384]);  sum_167 = None
    permute_580: "f32[384, 768]" = torch.ops.aten.permute.default(permute_579, [1, 0]);  permute_579 = None
    view_759: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_126, [1, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_230: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_229, view_759);  add_229 = view_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_113: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_61, getitem_53);  add_61 = getitem_53 = None
    mul_351: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_10);  sub_113 = None
    mul_352: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_230, primals_126);  primals_126 = None
    mul_353: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_352, 768)
    sum_168: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True)
    mul_354: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_352, mul_351);  mul_352 = None
    sum_169: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True);  mul_354 = None
    mul_355: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_351, sum_169);  sum_169 = None
    sub_114: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_353, sum_168);  mul_353 = sum_168 = None
    sub_115: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_114, mul_355);  sub_114 = mul_355 = None
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_356: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_115);  div_60 = sub_115 = None
    mul_357: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_230, mul_351);  mul_351 = None
    sum_170: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 1]);  mul_357 = None
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_358: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_359: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_356, mul_358);  mul_358 = None
    clone_78: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_359, memory_format = torch.contiguous_format);  mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_760: "f32[512, 768]" = torch.ops.aten.view.default(clone_78, [512, 768]);  clone_78 = None
    permute_581: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    mm_128: "f32[512, 3072]" = torch.ops.aten.mm.default(view_760, permute_581);  permute_581 = None
    permute_582: "f32[768, 512]" = torch.ops.aten.permute.default(view_760, [1, 0])
    mm_129: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_582, view_178);  permute_582 = view_178 = None
    permute_583: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_760, [0], True);  view_760 = None
    view_761: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    permute_584: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_583, [1, 0]);  permute_583 = None
    view_762: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_128, [1, 512, 3072]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_360: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476)
    erf_21: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_360);  mul_360 = None
    add_231: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_361: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_231, 0.5);  add_231 = None
    mul_362: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, view_177)
    mul_363: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_362, -0.5);  mul_362 = None
    exp_34: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_363);  mul_363 = None
    mul_364: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_365: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, mul_364);  view_177 = mul_364 = None
    add_232: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_361, mul_365);  mul_361 = mul_365 = None
    mul_366: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_762, add_232);  view_762 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_763: "f32[512, 3072]" = torch.ops.aten.view.default(mul_366, [512, 3072]);  mul_366 = None
    permute_585: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    mm_130: "f32[512, 768]" = torch.ops.aten.mm.default(view_763, permute_585);  permute_585 = None
    permute_586: "f32[3072, 512]" = torch.ops.aten.permute.default(view_763, [1, 0])
    mm_131: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_586, view_176);  permute_586 = view_176 = None
    permute_587: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_173: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True);  view_763 = None
    view_764: "f32[3072]" = torch.ops.aten.view.default(sum_173, [3072]);  sum_173 = None
    permute_588: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_587, [1, 0]);  permute_587 = None
    view_765: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_130, [1, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_233: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_356, view_765);  mul_356 = view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_116: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_49);  add_57 = getitem_49 = None
    mul_367: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_9);  sub_116 = None
    mul_368: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_233, primals_120);  primals_120 = None
    mul_369: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_368, 768)
    sum_174: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [2], True)
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_368, mul_367);  mul_368 = None
    sum_175: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True);  mul_370 = None
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_367, sum_175);  sum_175 = None
    sub_117: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_369, sum_174);  mul_369 = sum_174 = None
    sub_118: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_117, mul_371);  sub_117 = mul_371 = None
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_118);  div_61 = sub_118 = None
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_233, mul_367);  mul_367 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 1]);  mul_373 = None
    sum_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_372, mul_374);  mul_374 = None
    clone_79: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_375, memory_format = torch.contiguous_format);  mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_766: "f32[512, 768]" = torch.ops.aten.view.default(clone_79, [512, 768]);  clone_79 = None
    permute_589: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_132: "f32[512, 768]" = torch.ops.aten.mm.default(view_766, permute_589);  permute_589 = None
    permute_590: "f32[768, 512]" = torch.ops.aten.permute.default(view_766, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_590, view_174);  permute_590 = view_174 = None
    permute_591: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_178: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_766, [0], True);  view_766 = None
    view_767: "f32[768]" = torch.ops.aten.view.default(sum_178, [768]);  sum_178 = None
    permute_592: "f32[768, 768]" = torch.ops.aten.permute.default(permute_591, [1, 0]);  permute_591 = None
    view_768: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_132, [1, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_769: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_768, [1, 512, 12, 64]);  view_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_43: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_769, 2, 0, 6)
    slice_44: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_769, 2, 6, 12);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_770: "f32[512, 384]" = torch.ops.aten.view.default(slice_44, [512, 384]);  slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_593: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_43, [0, 2, 1, 3]);  slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_771: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_593, [6, 512, 64]);  permute_593 = None
    permute_594: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    bmm_78: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_594, view_771);  permute_594 = None
    permute_595: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
    bmm_79: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_771, permute_595);  view_771 = permute_595 = None
    view_772: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_78, [1, 6, 512, 64]);  bmm_78 = None
    view_773: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_79, [1, 6, 512, 512]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_24: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_376: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_377: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_773, mul_376);  view_773 = mul_376 = None
    clone_80: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_377, memory_format = torch.contiguous_format);  mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_40: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_378: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_80, alias_40);  clone_80 = None
    sum_179: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [-1], True)
    mul_379: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_40, sum_179);  alias_40 = sum_179 = None
    sub_119: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_378, mul_379);  mul_378 = mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_62: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_119, 8.0);  sub_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_774: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_62, [6, 512, 512]);  div_62 = None
    permute_596: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_80: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_596, view_774);  permute_596 = None
    permute_597: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_81: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_774, permute_597);  view_774 = permute_597 = None
    view_775: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_80, [1, 6, 64, 512]);  bmm_80 = None
    view_776: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_81, [1, 6, 512, 64]);  bmm_81 = None
    permute_598: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_775, [0, 1, 3, 2]);  view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_81: "f32[512, 384]" = torch.ops.aten.clone.default(view_770, memory_format = torch.contiguous_format);  view_770 = None
    view_777: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_81, [3072, 64, 1]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_778: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_777, [3072, 64, 1]);  view_777 = None
    permute_599: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1]);  view_162 = None
    bmm_82: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_599, view_778);  permute_599 = None
    permute_600: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
    bmm_83: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_778, permute_600);  view_778 = permute_600 = None
    view_779: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_82, [3072, 9, 1]);  bmm_82 = None
    view_780: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_83, [3072, 64, 9]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_781: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_780, [1, 512, 384, 9]);  view_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_782: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_781, [1, 512, 3456]);  view_781 = None
    permute_601: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_782, [0, 2, 1]);  view_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_783: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_601, [1, 384, 9, 1, 512, 1]);  permute_601 = None
    permute_602: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_783, [0, 1, 2, 4, 3, 5]);  view_783 = None
    iota_76: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_130: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_76, 0);  iota_76 = None
    iota_77: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_131: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_77, -1);  iota_77 = None
    add_234: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_130, unsqueeze_131);  unsqueeze_130 = unsqueeze_131 = None
    unsqueeze_132: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_234, -1);  add_234 = None
    unsqueeze_133: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    iota_78: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_134: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_78, 0);  iota_78 = None
    iota_79: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_135: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_79, -1);  iota_79 = None
    add_235: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_134, unsqueeze_135);  unsqueeze_134 = unsqueeze_135 = None
    full_9: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_7: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_9, [None, None, unsqueeze_133, add_235], permute_602, True);  full_9 = unsqueeze_133 = add_235 = permute_602 = None
    constant_pad_nd_19: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_7, [0, 0, -4, -4], 0.0);  _unsafe_index_put_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_8: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_19, -1);  constant_pad_nd_19 = None
    permute_603: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_8, [0, 2, 1]);  squeeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_784: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_603, [1, 512, 384]);  permute_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_785: "f32[512, 384]" = torch.ops.aten.view.default(view_784, [512, 384]);  view_784 = None
    permute_604: "f32[384, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_134: "f32[512, 768]" = torch.ops.aten.mm.default(view_785, permute_604);  permute_604 = None
    permute_605: "f32[384, 512]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_135: "f32[384, 768]" = torch.ops.aten.mm.default(permute_605, view_156);  permute_605 = view_156 = None
    permute_606: "f32[768, 384]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_180: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
    view_786: "f32[384]" = torch.ops.aten.view.default(sum_180, [384]);  sum_180 = None
    permute_607: "f32[384, 768]" = torch.ops.aten.permute.default(permute_606, [1, 0]);  permute_606 = None
    view_787: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_134, [1, 512, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_236: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_372, view_787);  mul_372 = view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_41: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_380: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_779, alias_41);  view_779 = None
    sum_181: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [1], True)
    mul_381: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_41, sum_181);  alias_41 = sum_181 = None
    sub_120: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_788: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_120, [1, 512, 54]);  sub_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_182: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_788, [0, 1], True)
    view_789: "f32[54]" = torch.ops.aten.view.default(sum_182, [54]);  sum_182 = None
    view_790: "f32[512, 54]" = torch.ops.aten.view.default(view_788, [512, 54]);  view_788 = None
    permute_608: "f32[54, 512]" = torch.ops.aten.permute.default(view_790, [1, 0])
    mm_136: "f32[54, 384]" = torch.ops.aten.mm.default(permute_608, view_153);  permute_608 = view_153 = None
    permute_609: "f32[384, 54]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    permute_610: "f32[54, 512]" = torch.ops.aten.permute.default(view_790, [1, 0]);  view_790 = None
    mm_137: "f32[384, 512]" = torch.ops.aten.mm.default(permute_85, permute_610);  permute_85 = permute_610 = None
    permute_611: "f32[512, 384]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    view_791: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_611, [1, 512, 384]);  permute_611 = None
    permute_612: "f32[54, 384]" = torch.ops.aten.permute.default(permute_609, [1, 0]);  permute_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_382: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_791, permute_84);  permute_84 = None
    mul_383: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_791, view_145);  view_791 = view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_613: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_82: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_613, memory_format = torch.contiguous_format);  permute_613 = None
    view_792: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_82, [1, 512, 384]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_614: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_598, [0, 2, 1, 3]);  permute_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_793: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_614, [1, 512, 384]);  permute_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_615: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_83: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
    view_794: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_83, [1, 512, 384]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_237: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_382, view_794);  mul_382 = view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_616: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_383, [0, 2, 1]);  mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_183: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_616, [0, 2], True)
    view_795: "f32[384, 1]" = torch.ops.aten.view.default(sum_183, [384, 1]);  sum_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(permute_616, convolution_8, primals_113, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_616 = convolution_8 = primals_113 = None
    getitem_168: "f32[1, 768, 512]" = convolution_backward_14[0]
    getitem_169: "f32[384, 768, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(getitem_168, permute_79, primals_112, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_168 = permute_79 = primals_112 = None
    getitem_171: "f32[1, 768, 512]" = convolution_backward_15[0]
    getitem_172: "f32[768, 1, 9]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_617: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1]);  getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_238: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_236, permute_617);  add_236 = permute_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_796: "f32[512, 384]" = torch.ops.aten.view.default(view_792, [512, 384]);  view_792 = None
    permute_618: "f32[384, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_138: "f32[512, 768]" = torch.ops.aten.mm.default(view_796, permute_618);  permute_618 = None
    permute_619: "f32[384, 512]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_139: "f32[384, 768]" = torch.ops.aten.mm.default(permute_619, view_148);  permute_619 = view_148 = None
    permute_620: "f32[768, 384]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_184: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
    view_797: "f32[384]" = torch.ops.aten.view.default(sum_184, [384]);  sum_184 = None
    permute_621: "f32[384, 768]" = torch.ops.aten.permute.default(permute_620, [1, 0]);  permute_620 = None
    view_798: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_138, [1, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_239: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_238, view_798);  add_238 = view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_799: "f32[512, 384]" = torch.ops.aten.view.default(view_793, [512, 384]);  view_793 = None
    permute_622: "f32[384, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_140: "f32[512, 768]" = torch.ops.aten.mm.default(view_799, permute_622);  permute_622 = None
    permute_623: "f32[384, 512]" = torch.ops.aten.permute.default(view_799, [1, 0])
    mm_141: "f32[384, 768]" = torch.ops.aten.mm.default(permute_623, view_146);  permute_623 = view_146 = None
    permute_624: "f32[768, 384]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_185: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
    view_800: "f32[384]" = torch.ops.aten.view.default(sum_185, [384]);  sum_185 = None
    permute_625: "f32[384, 768]" = torch.ops.aten.permute.default(permute_624, [1, 0]);  permute_624 = None
    view_801: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_140, [1, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_240: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_239, view_801);  add_239 = view_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_802: "f32[512, 384]" = torch.ops.aten.view.default(add_237, [512, 384]);  add_237 = None
    permute_626: "f32[384, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_142: "f32[512, 768]" = torch.ops.aten.mm.default(view_802, permute_626);  permute_626 = None
    permute_627: "f32[384, 512]" = torch.ops.aten.permute.default(view_802, [1, 0])
    mm_143: "f32[384, 768]" = torch.ops.aten.mm.default(permute_627, view_144);  permute_627 = view_144 = None
    permute_628: "f32[768, 384]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_186: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_802, [0], True);  view_802 = None
    view_803: "f32[384]" = torch.ops.aten.view.default(sum_186, [384]);  sum_186 = None
    permute_629: "f32[384, 768]" = torch.ops.aten.permute.default(permute_628, [1, 0]);  permute_628 = None
    view_804: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_142, [1, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_241: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_240, view_804);  add_240 = view_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_121: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_43);  add_49 = getitem_43 = None
    mul_384: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_8);  sub_121 = None
    mul_385: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_241, primals_104);  primals_104 = None
    mul_386: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_385, 768)
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True)
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_385, mul_384);  mul_385 = None
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True);  mul_387 = None
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_384, sum_188);  sum_188 = None
    sub_122: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_386, sum_187);  mul_386 = sum_187 = None
    sub_123: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_122, mul_388);  sub_122 = mul_388 = None
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_63, sub_123);  div_63 = sub_123 = None
    mul_390: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_241, mul_384);  mul_384 = None
    sum_189: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 1]);  mul_390 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_241, [0, 1]);  add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_391: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_392: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_389, mul_391);  mul_391 = None
    clone_84: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_392, memory_format = torch.contiguous_format);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_805: "f32[512, 768]" = torch.ops.aten.view.default(clone_84, [512, 768]);  clone_84 = None
    permute_630: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_144: "f32[512, 3072]" = torch.ops.aten.mm.default(view_805, permute_630);  permute_630 = None
    permute_631: "f32[768, 512]" = torch.ops.aten.permute.default(view_805, [1, 0])
    mm_145: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_631, view_142);  permute_631 = view_142 = None
    permute_632: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_191: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_805, [0], True);  view_805 = None
    view_806: "f32[768]" = torch.ops.aten.view.default(sum_191, [768]);  sum_191 = None
    permute_633: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    view_807: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_144, [1, 512, 3072]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_393: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_22: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_393);  mul_393 = None
    add_242: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_394: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_242, 0.5);  add_242 = None
    mul_395: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, view_141)
    mul_396: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_395, -0.5);  mul_395 = None
    exp_35: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_396);  mul_396 = None
    mul_397: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_398: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, mul_397);  view_141 = mul_397 = None
    add_243: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_394, mul_398);  mul_394 = mul_398 = None
    mul_399: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_807, add_243);  view_807 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_808: "f32[512, 3072]" = torch.ops.aten.view.default(mul_399, [512, 3072]);  mul_399 = None
    permute_634: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_146: "f32[512, 768]" = torch.ops.aten.mm.default(view_808, permute_634);  permute_634 = None
    permute_635: "f32[3072, 512]" = torch.ops.aten.permute.default(view_808, [1, 0])
    mm_147: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_635, view_140);  permute_635 = view_140 = None
    permute_636: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_192: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_808, [0], True);  view_808 = None
    view_809: "f32[3072]" = torch.ops.aten.view.default(sum_192, [3072]);  sum_192 = None
    permute_637: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_636, [1, 0]);  permute_636 = None
    view_810: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_146, [1, 512, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_244: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_389, view_810);  mul_389 = view_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_124: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_39);  add_45 = getitem_39 = None
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_7);  sub_124 = None
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_244, primals_98);  primals_98 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_401, 768)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True)
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_401, mul_400);  mul_401 = None
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [2], True);  mul_403 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_400, sum_194);  sum_194 = None
    sub_125: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_402, sum_193);  mul_402 = sum_193 = None
    sub_126: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_404);  sub_125 = mul_404 = None
    div_64: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_64, sub_126);  div_64 = sub_126 = None
    mul_406: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_244, mul_400);  mul_400 = None
    sum_195: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 1]);  mul_406 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_244, [0, 1]);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_407: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_408: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_405, mul_407);  mul_407 = None
    clone_85: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_408, memory_format = torch.contiguous_format);  mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_811: "f32[512, 768]" = torch.ops.aten.view.default(clone_85, [512, 768]);  clone_85 = None
    permute_638: "f32[768, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_148: "f32[512, 768]" = torch.ops.aten.mm.default(view_811, permute_638);  permute_638 = None
    permute_639: "f32[768, 512]" = torch.ops.aten.permute.default(view_811, [1, 0])
    mm_149: "f32[768, 768]" = torch.ops.aten.mm.default(permute_639, view_138);  permute_639 = view_138 = None
    permute_640: "f32[768, 768]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_197: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_811, [0], True);  view_811 = None
    view_812: "f32[768]" = torch.ops.aten.view.default(sum_197, [768]);  sum_197 = None
    permute_641: "f32[768, 768]" = torch.ops.aten.permute.default(permute_640, [1, 0]);  permute_640 = None
    view_813: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_148, [1, 512, 768]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_814: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_813, [1, 512, 12, 64]);  view_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_45: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_814, 2, 0, 6)
    slice_46: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_814, 2, 6, 12);  view_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_815: "f32[512, 384]" = torch.ops.aten.view.default(slice_46, [512, 384]);  slice_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_642: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_45, [0, 2, 1, 3]);  slice_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_816: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_642, [6, 512, 64]);  permute_642 = None
    permute_643: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_84: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_643, view_816);  permute_643 = None
    permute_644: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    bmm_85: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_816, permute_644);  view_816 = permute_644 = None
    view_817: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_84, [1, 6, 512, 64]);  bmm_84 = None
    view_818: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_85, [1, 6, 512, 512]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_27: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_409: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_410: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_818, mul_409);  view_818 = mul_409 = None
    clone_86: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_410, memory_format = torch.contiguous_format);  mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_42: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_411: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_86, alias_42);  clone_86 = None
    sum_198: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [-1], True)
    mul_412: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_42, sum_198);  alias_42 = sum_198 = None
    sub_127: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_65: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_127, 8.0);  sub_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_819: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_65, [6, 512, 512]);  div_65 = None
    permute_645: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
    bmm_86: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_645, view_819);  permute_645 = None
    permute_646: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
    bmm_87: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_819, permute_646);  view_819 = permute_646 = None
    view_820: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_86, [1, 6, 64, 512]);  bmm_86 = None
    view_821: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_87, [1, 6, 512, 64]);  bmm_87 = None
    permute_647: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_820, [0, 1, 3, 2]);  view_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_87: "f32[512, 384]" = torch.ops.aten.clone.default(view_815, memory_format = torch.contiguous_format);  view_815 = None
    view_822: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_87, [3072, 64, 1]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_823: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_822, [3072, 64, 1]);  view_822 = None
    permute_648: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
    bmm_88: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_648, view_823);  permute_648 = None
    permute_649: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
    bmm_89: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_823, permute_649);  view_823 = permute_649 = None
    view_824: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_88, [3072, 9, 1]);  bmm_88 = None
    view_825: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_89, [3072, 64, 9]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_826: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_825, [1, 512, 384, 9]);  view_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_827: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_826, [1, 512, 3456]);  view_826 = None
    permute_650: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_827, [0, 2, 1]);  view_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_828: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_650, [1, 384, 9, 1, 512, 1]);  permute_650 = None
    permute_651: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_828, [0, 1, 2, 4, 3, 5]);  view_828 = None
    iota_80: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_136: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_80, 0);  iota_80 = None
    iota_81: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_137: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_81, -1);  iota_81 = None
    add_245: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_136, unsqueeze_137);  unsqueeze_136 = unsqueeze_137 = None
    unsqueeze_138: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_245, -1);  add_245 = None
    unsqueeze_139: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    iota_82: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_140: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_82, 0);  iota_82 = None
    iota_83: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_141: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_83, -1);  iota_83 = None
    add_246: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_140, unsqueeze_141);  unsqueeze_140 = unsqueeze_141 = None
    full_10: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_8: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_10, [None, None, unsqueeze_139, add_246], permute_651, True);  full_10 = unsqueeze_139 = add_246 = permute_651 = None
    constant_pad_nd_20: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_8, [0, 0, -4, -4], 0.0);  _unsafe_index_put_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_9: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_20, -1);  constant_pad_nd_20 = None
    permute_652: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_9, [0, 2, 1]);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_829: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_652, [1, 512, 384]);  permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_830: "f32[512, 384]" = torch.ops.aten.view.default(view_829, [512, 384]);  view_829 = None
    permute_653: "f32[384, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_150: "f32[512, 768]" = torch.ops.aten.mm.default(view_830, permute_653);  permute_653 = None
    permute_654: "f32[384, 512]" = torch.ops.aten.permute.default(view_830, [1, 0])
    mm_151: "f32[384, 768]" = torch.ops.aten.mm.default(permute_654, view_120);  permute_654 = view_120 = None
    permute_655: "f32[768, 384]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_199: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_830, [0], True);  view_830 = None
    view_831: "f32[384]" = torch.ops.aten.view.default(sum_199, [384]);  sum_199 = None
    permute_656: "f32[384, 768]" = torch.ops.aten.permute.default(permute_655, [1, 0]);  permute_655 = None
    view_832: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_150, [1, 512, 768]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_247: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_405, view_832);  mul_405 = view_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_43: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_413: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_824, alias_43);  view_824 = None
    sum_200: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [1], True)
    mul_414: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_43, sum_200);  alias_43 = sum_200 = None
    sub_128: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_413, mul_414);  mul_413 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_833: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_128, [1, 512, 54]);  sub_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_201: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_833, [0, 1], True)
    view_834: "f32[54]" = torch.ops.aten.view.default(sum_201, [54]);  sum_201 = None
    view_835: "f32[512, 54]" = torch.ops.aten.view.default(view_833, [512, 54]);  view_833 = None
    permute_657: "f32[54, 512]" = torch.ops.aten.permute.default(view_835, [1, 0])
    mm_152: "f32[54, 384]" = torch.ops.aten.mm.default(permute_657, view_117);  permute_657 = view_117 = None
    permute_658: "f32[384, 54]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    permute_659: "f32[54, 512]" = torch.ops.aten.permute.default(view_835, [1, 0]);  view_835 = None
    mm_153: "f32[384, 512]" = torch.ops.aten.mm.default(permute_66, permute_659);  permute_66 = permute_659 = None
    permute_660: "f32[512, 384]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    view_836: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_660, [1, 512, 384]);  permute_660 = None
    permute_661: "f32[54, 384]" = torch.ops.aten.permute.default(permute_658, [1, 0]);  permute_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_415: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_836, permute_65);  permute_65 = None
    mul_416: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_836, view_109);  view_836 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_662: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_817, [0, 2, 1, 3]);  view_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_88: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_662, memory_format = torch.contiguous_format);  permute_662 = None
    view_837: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_88, [1, 512, 384]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_663: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_647, [0, 2, 1, 3]);  permute_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_838: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_663, [1, 512, 384]);  permute_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_664: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_821, [0, 2, 1, 3]);  view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_89: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_664, memory_format = torch.contiguous_format);  permute_664 = None
    view_839: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_89, [1, 512, 384]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_248: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_415, view_839);  mul_415 = view_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_665: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_416, [0, 2, 1]);  mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_202: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_665, [0, 2], True)
    view_840: "f32[384, 1]" = torch.ops.aten.view.default(sum_202, [384, 1]);  sum_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(permute_665, convolution_6, primals_91, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_665 = convolution_6 = primals_91 = None
    getitem_174: "f32[1, 768, 512]" = convolution_backward_16[0]
    getitem_175: "f32[384, 768, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(getitem_174, permute_60, primals_90, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_174 = permute_60 = primals_90 = None
    getitem_177: "f32[1, 768, 512]" = convolution_backward_17[0]
    getitem_178: "f32[768, 1, 9]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_666: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_177, [0, 2, 1]);  getitem_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_249: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_247, permute_666);  add_247 = permute_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_841: "f32[512, 384]" = torch.ops.aten.view.default(view_837, [512, 384]);  view_837 = None
    permute_667: "f32[384, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_154: "f32[512, 768]" = torch.ops.aten.mm.default(view_841, permute_667);  permute_667 = None
    permute_668: "f32[384, 512]" = torch.ops.aten.permute.default(view_841, [1, 0])
    mm_155: "f32[384, 768]" = torch.ops.aten.mm.default(permute_668, view_112);  permute_668 = view_112 = None
    permute_669: "f32[768, 384]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_203: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
    view_842: "f32[384]" = torch.ops.aten.view.default(sum_203, [384]);  sum_203 = None
    permute_670: "f32[384, 768]" = torch.ops.aten.permute.default(permute_669, [1, 0]);  permute_669 = None
    view_843: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_154, [1, 512, 768]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_250: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_249, view_843);  add_249 = view_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_844: "f32[512, 384]" = torch.ops.aten.view.default(view_838, [512, 384]);  view_838 = None
    permute_671: "f32[384, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_156: "f32[512, 768]" = torch.ops.aten.mm.default(view_844, permute_671);  permute_671 = None
    permute_672: "f32[384, 512]" = torch.ops.aten.permute.default(view_844, [1, 0])
    mm_157: "f32[384, 768]" = torch.ops.aten.mm.default(permute_672, view_110);  permute_672 = view_110 = None
    permute_673: "f32[768, 384]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_204: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_844, [0], True);  view_844 = None
    view_845: "f32[384]" = torch.ops.aten.view.default(sum_204, [384]);  sum_204 = None
    permute_674: "f32[384, 768]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    view_846: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_156, [1, 512, 768]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_251: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_250, view_846);  add_250 = view_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_847: "f32[512, 384]" = torch.ops.aten.view.default(add_248, [512, 384]);  add_248 = None
    permute_675: "f32[384, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_158: "f32[512, 768]" = torch.ops.aten.mm.default(view_847, permute_675);  permute_675 = None
    permute_676: "f32[384, 512]" = torch.ops.aten.permute.default(view_847, [1, 0])
    mm_159: "f32[384, 768]" = torch.ops.aten.mm.default(permute_676, view_108);  permute_676 = view_108 = None
    permute_677: "f32[768, 384]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_205: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_847, [0], True);  view_847 = None
    view_848: "f32[384]" = torch.ops.aten.view.default(sum_205, [384]);  sum_205 = None
    permute_678: "f32[384, 768]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    view_849: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_158, [1, 512, 768]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_252: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_251, view_849);  add_251 = view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_129: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_33);  add_37 = getitem_33 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt_6);  sub_129 = None
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_252, primals_82);  primals_82 = None
    mul_419: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_418, 768)
    sum_206: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True)
    mul_420: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_418, mul_417);  mul_418 = None
    sum_207: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_420, [2], True);  mul_420 = None
    mul_421: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_417, sum_207);  sum_207 = None
    sub_130: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_419, sum_206);  mul_419 = sum_206 = None
    sub_131: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_130, mul_421);  sub_130 = mul_421 = None
    div_66: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_422: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_66, sub_131);  div_66 = sub_131 = None
    mul_423: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_252, mul_417);  mul_417 = None
    sum_208: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_423, [0, 1]);  mul_423 = None
    sum_209: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_252, [0, 1]);  add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_424: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_425: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_422, mul_424);  mul_424 = None
    clone_90: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_425, memory_format = torch.contiguous_format);  mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_850: "f32[512, 768]" = torch.ops.aten.view.default(clone_90, [512, 768]);  clone_90 = None
    permute_679: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_160: "f32[512, 3072]" = torch.ops.aten.mm.default(view_850, permute_679);  permute_679 = None
    permute_680: "f32[768, 512]" = torch.ops.aten.permute.default(view_850, [1, 0])
    mm_161: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_680, view_106);  permute_680 = view_106 = None
    permute_681: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_210: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_850, [0], True);  view_850 = None
    view_851: "f32[768]" = torch.ops.aten.view.default(sum_210, [768]);  sum_210 = None
    permute_682: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_681, [1, 0]);  permute_681 = None
    view_852: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_160, [1, 512, 3072]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_426: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476)
    erf_23: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_426);  mul_426 = None
    add_253: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_427: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_253, 0.5);  add_253 = None
    mul_428: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, view_105)
    mul_429: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_428, -0.5);  mul_428 = None
    exp_36: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_429);  mul_429 = None
    mul_430: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_431: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, mul_430);  view_105 = mul_430 = None
    add_254: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_427, mul_431);  mul_427 = mul_431 = None
    mul_432: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_852, add_254);  view_852 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_853: "f32[512, 3072]" = torch.ops.aten.view.default(mul_432, [512, 3072]);  mul_432 = None
    permute_683: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_162: "f32[512, 768]" = torch.ops.aten.mm.default(view_853, permute_683);  permute_683 = None
    permute_684: "f32[3072, 512]" = torch.ops.aten.permute.default(view_853, [1, 0])
    mm_163: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_684, view_104);  permute_684 = view_104 = None
    permute_685: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_211: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_853, [0], True);  view_853 = None
    view_854: "f32[3072]" = torch.ops.aten.view.default(sum_211, [3072]);  sum_211 = None
    permute_686: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_685, [1, 0]);  permute_685 = None
    view_855: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_162, [1, 512, 768]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_255: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_422, view_855);  mul_422 = view_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_132: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_29);  add_33 = getitem_29 = None
    mul_433: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_132, rsqrt_5);  sub_132 = None
    mul_434: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_255, primals_76);  primals_76 = None
    mul_435: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_434, 768)
    sum_212: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True)
    mul_436: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_434, mul_433);  mul_434 = None
    sum_213: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True);  mul_436 = None
    mul_437: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_433, sum_213);  sum_213 = None
    sub_133: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_435, sum_212);  mul_435 = sum_212 = None
    sub_134: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_133, mul_437);  sub_133 = mul_437 = None
    div_67: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_438: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_67, sub_134);  div_67 = sub_134 = None
    mul_439: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_255, mul_433);  mul_433 = None
    sum_214: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 1]);  mul_439 = None
    sum_215: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_255, [0, 1]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_440: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_441: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_438, mul_440);  mul_440 = None
    clone_91: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_441, memory_format = torch.contiguous_format);  mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_856: "f32[512, 768]" = torch.ops.aten.view.default(clone_91, [512, 768]);  clone_91 = None
    permute_687: "f32[768, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_164: "f32[512, 768]" = torch.ops.aten.mm.default(view_856, permute_687);  permute_687 = None
    permute_688: "f32[768, 512]" = torch.ops.aten.permute.default(view_856, [1, 0])
    mm_165: "f32[768, 768]" = torch.ops.aten.mm.default(permute_688, view_102);  permute_688 = view_102 = None
    permute_689: "f32[768, 768]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_216: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_856, [0], True);  view_856 = None
    view_857: "f32[768]" = torch.ops.aten.view.default(sum_216, [768]);  sum_216 = None
    permute_690: "f32[768, 768]" = torch.ops.aten.permute.default(permute_689, [1, 0]);  permute_689 = None
    view_858: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_164, [1, 512, 768]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_859: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_858, [1, 512, 12, 64]);  view_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_47: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_859, 2, 0, 6)
    slice_48: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_859, 2, 6, 12);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_860: "f32[512, 384]" = torch.ops.aten.view.default(slice_48, [512, 384]);  slice_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_691: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_47, [0, 2, 1, 3]);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_861: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_691, [6, 512, 64]);  permute_691 = None
    permute_692: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    bmm_90: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_692, view_861);  permute_692 = None
    permute_693: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    bmm_91: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_861, permute_693);  view_861 = permute_693 = None
    view_862: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_90, [1, 6, 512, 64]);  bmm_90 = None
    view_863: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_91, [1, 6, 512, 512]);  bmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_30: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_442: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_443: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_863, mul_442);  view_863 = mul_442 = None
    clone_92: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_443, memory_format = torch.contiguous_format);  mul_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_44: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_444: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_92, alias_44);  clone_92 = None
    sum_217: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [-1], True)
    mul_445: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_44, sum_217);  alias_44 = sum_217 = None
    sub_135: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_68: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_135, 8.0);  sub_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_864: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_68, [6, 512, 512]);  div_68 = None
    permute_694: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    bmm_92: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_694, view_864);  permute_694 = None
    permute_695: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    bmm_93: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_864, permute_695);  view_864 = permute_695 = None
    view_865: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_92, [1, 6, 64, 512]);  bmm_92 = None
    view_866: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_93, [1, 6, 512, 64]);  bmm_93 = None
    permute_696: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_865, [0, 1, 3, 2]);  view_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_93: "f32[512, 384]" = torch.ops.aten.clone.default(view_860, memory_format = torch.contiguous_format);  view_860 = None
    view_867: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_93, [3072, 64, 1]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_868: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_867, [3072, 64, 1]);  view_867 = None
    permute_697: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
    bmm_94: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_697, view_868);  permute_697 = None
    permute_698: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    bmm_95: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_868, permute_698);  view_868 = permute_698 = None
    view_869: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_94, [3072, 9, 1]);  bmm_94 = None
    view_870: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_95, [3072, 64, 9]);  bmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_871: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_870, [1, 512, 384, 9]);  view_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_872: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_871, [1, 512, 3456]);  view_871 = None
    permute_699: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_872, [0, 2, 1]);  view_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_873: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_699, [1, 384, 9, 1, 512, 1]);  permute_699 = None
    permute_700: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_873, [0, 1, 2, 4, 3, 5]);  view_873 = None
    iota_84: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_142: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_84, 0);  iota_84 = None
    iota_85: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_143: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_85, -1);  iota_85 = None
    add_256: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_142, unsqueeze_143);  unsqueeze_142 = unsqueeze_143 = None
    unsqueeze_144: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_256, -1);  add_256 = None
    unsqueeze_145: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    iota_86: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_146: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_86, 0);  iota_86 = None
    iota_87: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_147: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_87, -1);  iota_87 = None
    add_257: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_146, unsqueeze_147);  unsqueeze_146 = unsqueeze_147 = None
    full_11: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_9: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_11, [None, None, unsqueeze_145, add_257], permute_700, True);  full_11 = unsqueeze_145 = add_257 = permute_700 = None
    constant_pad_nd_21: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_9, [0, 0, -4, -4], 0.0);  _unsafe_index_put_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_10: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_21, -1);  constant_pad_nd_21 = None
    permute_701: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_10, [0, 2, 1]);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_874: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_701, [1, 512, 384]);  permute_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_875: "f32[512, 384]" = torch.ops.aten.view.default(view_874, [512, 384]);  view_874 = None
    permute_702: "f32[384, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_166: "f32[512, 768]" = torch.ops.aten.mm.default(view_875, permute_702);  permute_702 = None
    permute_703: "f32[384, 512]" = torch.ops.aten.permute.default(view_875, [1, 0])
    mm_167: "f32[384, 768]" = torch.ops.aten.mm.default(permute_703, view_84);  permute_703 = view_84 = None
    permute_704: "f32[768, 384]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_218: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_875, [0], True);  view_875 = None
    view_876: "f32[384]" = torch.ops.aten.view.default(sum_218, [384]);  sum_218 = None
    permute_705: "f32[384, 768]" = torch.ops.aten.permute.default(permute_704, [1, 0]);  permute_704 = None
    view_877: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_166, [1, 512, 768]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_258: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_438, view_877);  mul_438 = view_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_45: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_446: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_869, alias_45);  view_869 = None
    sum_219: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [1], True)
    mul_447: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_45, sum_219);  alias_45 = sum_219 = None
    sub_136: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_446, mul_447);  mul_446 = mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_878: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_136, [1, 512, 54]);  sub_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_220: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_878, [0, 1], True)
    view_879: "f32[54]" = torch.ops.aten.view.default(sum_220, [54]);  sum_220 = None
    view_880: "f32[512, 54]" = torch.ops.aten.view.default(view_878, [512, 54]);  view_878 = None
    permute_706: "f32[54, 512]" = torch.ops.aten.permute.default(view_880, [1, 0])
    mm_168: "f32[54, 384]" = torch.ops.aten.mm.default(permute_706, view_81);  permute_706 = view_81 = None
    permute_707: "f32[384, 54]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    permute_708: "f32[54, 512]" = torch.ops.aten.permute.default(view_880, [1, 0]);  view_880 = None
    mm_169: "f32[384, 512]" = torch.ops.aten.mm.default(permute_47, permute_708);  permute_47 = permute_708 = None
    permute_709: "f32[512, 384]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    view_881: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_709, [1, 512, 384]);  permute_709 = None
    permute_710: "f32[54, 384]" = torch.ops.aten.permute.default(permute_707, [1, 0]);  permute_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_448: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_881, permute_46);  permute_46 = None
    mul_449: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_881, view_73);  view_881 = view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_711: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_862, [0, 2, 1, 3]);  view_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_94: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_711, memory_format = torch.contiguous_format);  permute_711 = None
    view_882: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_94, [1, 512, 384]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_712: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_696, [0, 2, 1, 3]);  permute_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_883: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_712, [1, 512, 384]);  permute_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_713: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_866, [0, 2, 1, 3]);  view_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_95: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_713, memory_format = torch.contiguous_format);  permute_713 = None
    view_884: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_95, [1, 512, 384]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_259: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_448, view_884);  mul_448 = view_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_714: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_449, [0, 2, 1]);  mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_221: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_714, [0, 2], True)
    view_885: "f32[384, 1]" = torch.ops.aten.view.default(sum_221, [384, 1]);  sum_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(permute_714, convolution_4, primals_69, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_714 = convolution_4 = primals_69 = None
    getitem_180: "f32[1, 768, 512]" = convolution_backward_18[0]
    getitem_181: "f32[384, 768, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(getitem_180, permute_41, primals_68, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_180 = permute_41 = primals_68 = None
    getitem_183: "f32[1, 768, 512]" = convolution_backward_19[0]
    getitem_184: "f32[768, 1, 9]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_715: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_183, [0, 2, 1]);  getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_260: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_258, permute_715);  add_258 = permute_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_886: "f32[512, 384]" = torch.ops.aten.view.default(view_882, [512, 384]);  view_882 = None
    permute_716: "f32[384, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_170: "f32[512, 768]" = torch.ops.aten.mm.default(view_886, permute_716);  permute_716 = None
    permute_717: "f32[384, 512]" = torch.ops.aten.permute.default(view_886, [1, 0])
    mm_171: "f32[384, 768]" = torch.ops.aten.mm.default(permute_717, view_76);  permute_717 = view_76 = None
    permute_718: "f32[768, 384]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_222: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_886, [0], True);  view_886 = None
    view_887: "f32[384]" = torch.ops.aten.view.default(sum_222, [384]);  sum_222 = None
    permute_719: "f32[384, 768]" = torch.ops.aten.permute.default(permute_718, [1, 0]);  permute_718 = None
    view_888: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_170, [1, 512, 768]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_261: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_260, view_888);  add_260 = view_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_889: "f32[512, 384]" = torch.ops.aten.view.default(view_883, [512, 384]);  view_883 = None
    permute_720: "f32[384, 768]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    mm_172: "f32[512, 768]" = torch.ops.aten.mm.default(view_889, permute_720);  permute_720 = None
    permute_721: "f32[384, 512]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_173: "f32[384, 768]" = torch.ops.aten.mm.default(permute_721, view_74);  permute_721 = view_74 = None
    permute_722: "f32[768, 384]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_223: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
    view_890: "f32[384]" = torch.ops.aten.view.default(sum_223, [384]);  sum_223 = None
    permute_723: "f32[384, 768]" = torch.ops.aten.permute.default(permute_722, [1, 0]);  permute_722 = None
    view_891: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_172, [1, 512, 768]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_262: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_261, view_891);  add_261 = view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_892: "f32[512, 384]" = torch.ops.aten.view.default(add_259, [512, 384]);  add_259 = None
    permute_724: "f32[384, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    mm_174: "f32[512, 768]" = torch.ops.aten.mm.default(view_892, permute_724);  permute_724 = None
    permute_725: "f32[384, 512]" = torch.ops.aten.permute.default(view_892, [1, 0])
    mm_175: "f32[384, 768]" = torch.ops.aten.mm.default(permute_725, view_72);  permute_725 = view_72 = None
    permute_726: "f32[768, 384]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_224: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
    view_893: "f32[384]" = torch.ops.aten.view.default(sum_224, [384]);  sum_224 = None
    permute_727: "f32[384, 768]" = torch.ops.aten.permute.default(permute_726, [1, 0]);  permute_726 = None
    view_894: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_174, [1, 512, 768]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_263: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_262, view_894);  add_262 = view_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_137: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_23);  add_25 = getitem_23 = None
    mul_450: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_4);  sub_137 = None
    mul_451: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_263, primals_60);  primals_60 = None
    mul_452: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_451, 768)
    sum_225: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True)
    mul_453: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_451, mul_450);  mul_451 = None
    sum_226: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True);  mul_453 = None
    mul_454: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_450, sum_226);  sum_226 = None
    sub_138: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_452, sum_225);  mul_452 = sum_225 = None
    sub_139: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_138, mul_454);  sub_138 = mul_454 = None
    div_69: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_455: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_69, sub_139);  div_69 = sub_139 = None
    mul_456: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_263, mul_450);  mul_450 = None
    sum_227: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_456, [0, 1]);  mul_456 = None
    sum_228: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_457: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_458: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_455, mul_457);  mul_457 = None
    clone_96: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_458, memory_format = torch.contiguous_format);  mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_895: "f32[512, 768]" = torch.ops.aten.view.default(clone_96, [512, 768]);  clone_96 = None
    permute_728: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_176: "f32[512, 3072]" = torch.ops.aten.mm.default(view_895, permute_728);  permute_728 = None
    permute_729: "f32[768, 512]" = torch.ops.aten.permute.default(view_895, [1, 0])
    mm_177: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_729, view_70);  permute_729 = view_70 = None
    permute_730: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_229: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
    view_896: "f32[768]" = torch.ops.aten.view.default(sum_229, [768]);  sum_229 = None
    permute_731: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_730, [1, 0]);  permute_730 = None
    view_897: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_176, [1, 512, 3072]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_459: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_24: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_459);  mul_459 = None
    add_264: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_460: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_264, 0.5);  add_264 = None
    mul_461: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, view_69)
    mul_462: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_461, -0.5);  mul_461 = None
    exp_37: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_462);  mul_462 = None
    mul_463: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_464: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, mul_463);  view_69 = mul_463 = None
    add_265: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_460, mul_464);  mul_460 = mul_464 = None
    mul_465: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_897, add_265);  view_897 = add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_898: "f32[512, 3072]" = torch.ops.aten.view.default(mul_465, [512, 3072]);  mul_465 = None
    permute_732: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_178: "f32[512, 768]" = torch.ops.aten.mm.default(view_898, permute_732);  permute_732 = None
    permute_733: "f32[3072, 512]" = torch.ops.aten.permute.default(view_898, [1, 0])
    mm_179: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_733, view_68);  permute_733 = view_68 = None
    permute_734: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_230: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
    view_899: "f32[3072]" = torch.ops.aten.view.default(sum_230, [3072]);  sum_230 = None
    permute_735: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_734, [1, 0]);  permute_734 = None
    view_900: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_178, [1, 512, 768]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_266: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_455, view_900);  mul_455 = view_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_140: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_19);  add_21 = getitem_19 = None
    mul_466: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_3);  sub_140 = None
    mul_467: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_266, primals_54);  primals_54 = None
    mul_468: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_467, 768)
    sum_231: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
    mul_469: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_467, mul_466);  mul_467 = None
    sum_232: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
    mul_470: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_466, sum_232);  sum_232 = None
    sub_141: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_468, sum_231);  mul_468 = sum_231 = None
    sub_142: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_141, mul_470);  sub_141 = mul_470 = None
    div_70: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_471: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_70, sub_142);  div_70 = sub_142 = None
    mul_472: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_266, mul_466);  mul_466 = None
    sum_233: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
    sum_234: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_473: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_474: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_471, mul_473);  mul_473 = None
    clone_97: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_474, memory_format = torch.contiguous_format);  mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_901: "f32[512, 768]" = torch.ops.aten.view.default(clone_97, [512, 768]);  clone_97 = None
    permute_736: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_180: "f32[512, 768]" = torch.ops.aten.mm.default(view_901, permute_736);  permute_736 = None
    permute_737: "f32[768, 512]" = torch.ops.aten.permute.default(view_901, [1, 0])
    mm_181: "f32[768, 768]" = torch.ops.aten.mm.default(permute_737, view_66);  permute_737 = view_66 = None
    permute_738: "f32[768, 768]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_235: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_901, [0], True);  view_901 = None
    view_902: "f32[768]" = torch.ops.aten.view.default(sum_235, [768]);  sum_235 = None
    permute_739: "f32[768, 768]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
    view_903: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_180, [1, 512, 768]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_904: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_903, [1, 512, 12, 64]);  view_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_49: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_904, 2, 0, 6)
    slice_50: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_904, 2, 6, 12);  view_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_905: "f32[512, 384]" = torch.ops.aten.view.default(slice_50, [512, 384]);  slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_740: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_49, [0, 2, 1, 3]);  slice_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_906: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_740, [6, 512, 64]);  permute_740 = None
    permute_741: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    bmm_96: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_741, view_906);  permute_741 = None
    permute_742: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    bmm_97: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_906, permute_742);  view_906 = permute_742 = None
    view_907: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_96, [1, 6, 512, 64]);  bmm_96 = None
    view_908: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_97, [1, 6, 512, 512]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_33: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
    mul_475: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_476: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_908, mul_475);  view_908 = mul_475 = None
    clone_98: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_476, memory_format = torch.contiguous_format);  mul_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_46: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_477: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_98, alias_46);  clone_98 = None
    sum_236: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [-1], True)
    mul_478: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_46, sum_236);  alias_46 = sum_236 = None
    sub_143: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_71: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_143, 8.0);  sub_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_909: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_71, [6, 512, 512]);  div_71 = None
    permute_743: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_98: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_743, view_909);  permute_743 = None
    permute_744: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_99: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_909, permute_744);  view_909 = permute_744 = None
    view_910: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_98, [1, 6, 64, 512]);  bmm_98 = None
    view_911: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_99, [1, 6, 512, 64]);  bmm_99 = None
    permute_745: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_910, [0, 1, 3, 2]);  view_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_99: "f32[512, 384]" = torch.ops.aten.clone.default(view_905, memory_format = torch.contiguous_format);  view_905 = None
    view_912: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_99, [3072, 64, 1]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_913: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_912, [3072, 64, 1]);  view_912 = None
    permute_746: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    bmm_100: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_746, view_913);  permute_746 = None
    permute_747: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    bmm_101: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_913, permute_747);  view_913 = permute_747 = None
    view_914: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_100, [3072, 9, 1]);  bmm_100 = None
    view_915: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_101, [3072, 64, 9]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_916: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_915, [1, 512, 384, 9]);  view_915 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_917: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_916, [1, 512, 3456]);  view_916 = None
    permute_748: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_917, [0, 2, 1]);  view_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_918: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_748, [1, 384, 9, 1, 512, 1]);  permute_748 = None
    permute_749: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_918, [0, 1, 2, 4, 3, 5]);  view_918 = None
    iota_88: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_148: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_88, 0);  iota_88 = None
    iota_89: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_149: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_89, -1);  iota_89 = None
    add_267: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_148, unsqueeze_149);  unsqueeze_148 = unsqueeze_149 = None
    unsqueeze_150: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_267, -1);  add_267 = None
    unsqueeze_151: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    iota_90: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_152: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_90, 0);  iota_90 = None
    iota_91: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_153: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_91, -1);  iota_91 = None
    add_268: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_152, unsqueeze_153);  unsqueeze_152 = unsqueeze_153 = None
    full_12: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_10: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_12, [None, None, unsqueeze_151, add_268], permute_749, True);  full_12 = unsqueeze_151 = add_268 = permute_749 = None
    constant_pad_nd_22: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_10, [0, 0, -4, -4], 0.0);  _unsafe_index_put_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_11: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_22, -1);  constant_pad_nd_22 = None
    permute_750: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_11, [0, 2, 1]);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_919: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_750, [1, 512, 384]);  permute_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_920: "f32[512, 384]" = torch.ops.aten.view.default(view_919, [512, 384]);  view_919 = None
    permute_751: "f32[384, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    mm_182: "f32[512, 768]" = torch.ops.aten.mm.default(view_920, permute_751);  permute_751 = None
    permute_752: "f32[384, 512]" = torch.ops.aten.permute.default(view_920, [1, 0])
    mm_183: "f32[384, 768]" = torch.ops.aten.mm.default(permute_752, view_48);  permute_752 = view_48 = None
    permute_753: "f32[768, 384]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_237: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
    view_921: "f32[384]" = torch.ops.aten.view.default(sum_237, [384]);  sum_237 = None
    permute_754: "f32[384, 768]" = torch.ops.aten.permute.default(permute_753, [1, 0]);  permute_753 = None
    view_922: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_182, [1, 512, 768]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_269: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_471, view_922);  mul_471 = view_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_47: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_479: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_914, alias_47);  view_914 = None
    sum_238: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [1], True)
    mul_480: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_47, sum_238);  alias_47 = sum_238 = None
    sub_144: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_923: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_144, [1, 512, 54]);  sub_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_239: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_923, [0, 1], True)
    view_924: "f32[54]" = torch.ops.aten.view.default(sum_239, [54]);  sum_239 = None
    view_925: "f32[512, 54]" = torch.ops.aten.view.default(view_923, [512, 54]);  view_923 = None
    permute_755: "f32[54, 512]" = torch.ops.aten.permute.default(view_925, [1, 0])
    mm_184: "f32[54, 384]" = torch.ops.aten.mm.default(permute_755, view_45);  permute_755 = view_45 = None
    permute_756: "f32[384, 54]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    permute_757: "f32[54, 512]" = torch.ops.aten.permute.default(view_925, [1, 0]);  view_925 = None
    mm_185: "f32[384, 512]" = torch.ops.aten.mm.default(permute_28, permute_757);  permute_28 = permute_757 = None
    permute_758: "f32[512, 384]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    view_926: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_758, [1, 512, 384]);  permute_758 = None
    permute_759: "f32[54, 384]" = torch.ops.aten.permute.default(permute_756, [1, 0]);  permute_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_481: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_926, permute_27);  permute_27 = None
    mul_482: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_926, view_37);  view_926 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_760: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_907, [0, 2, 1, 3]);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_100: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_760, memory_format = torch.contiguous_format);  permute_760 = None
    view_927: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_100, [1, 512, 384]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_761: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_745, [0, 2, 1, 3]);  permute_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_928: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_761, [1, 512, 384]);  permute_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_762: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_911, [0, 2, 1, 3]);  view_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_101: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_762, memory_format = torch.contiguous_format);  permute_762 = None
    view_929: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_101, [1, 512, 384]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_270: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_481, view_929);  mul_481 = view_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_763: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_482, [0, 2, 1]);  mul_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_240: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_763, [0, 2], True)
    view_930: "f32[384, 1]" = torch.ops.aten.view.default(sum_240, [384, 1]);  sum_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(permute_763, convolution_2, primals_47, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_763 = convolution_2 = primals_47 = None
    getitem_186: "f32[1, 768, 512]" = convolution_backward_20[0]
    getitem_187: "f32[384, 768, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(getitem_186, permute_22, primals_46, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_186 = permute_22 = primals_46 = None
    getitem_189: "f32[1, 768, 512]" = convolution_backward_21[0]
    getitem_190: "f32[768, 1, 9]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_764: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1]);  getitem_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_271: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_269, permute_764);  add_269 = permute_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_931: "f32[512, 384]" = torch.ops.aten.view.default(view_927, [512, 384]);  view_927 = None
    permute_765: "f32[384, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_186: "f32[512, 768]" = torch.ops.aten.mm.default(view_931, permute_765);  permute_765 = None
    permute_766: "f32[384, 512]" = torch.ops.aten.permute.default(view_931, [1, 0])
    mm_187: "f32[384, 768]" = torch.ops.aten.mm.default(permute_766, view_40);  permute_766 = view_40 = None
    permute_767: "f32[768, 384]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_241: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_931, [0], True);  view_931 = None
    view_932: "f32[384]" = torch.ops.aten.view.default(sum_241, [384]);  sum_241 = None
    permute_768: "f32[384, 768]" = torch.ops.aten.permute.default(permute_767, [1, 0]);  permute_767 = None
    view_933: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_186, [1, 512, 768]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_272: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_271, view_933);  add_271 = view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_934: "f32[512, 384]" = torch.ops.aten.view.default(view_928, [512, 384]);  view_928 = None
    permute_769: "f32[384, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_188: "f32[512, 768]" = torch.ops.aten.mm.default(view_934, permute_769);  permute_769 = None
    permute_770: "f32[384, 512]" = torch.ops.aten.permute.default(view_934, [1, 0])
    mm_189: "f32[384, 768]" = torch.ops.aten.mm.default(permute_770, view_38);  permute_770 = view_38 = None
    permute_771: "f32[768, 384]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_242: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_934, [0], True);  view_934 = None
    view_935: "f32[384]" = torch.ops.aten.view.default(sum_242, [384]);  sum_242 = None
    permute_772: "f32[384, 768]" = torch.ops.aten.permute.default(permute_771, [1, 0]);  permute_771 = None
    view_936: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_188, [1, 512, 768]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_273: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_272, view_936);  add_272 = view_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_937: "f32[512, 384]" = torch.ops.aten.view.default(add_270, [512, 384]);  add_270 = None
    permute_773: "f32[384, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_190: "f32[512, 768]" = torch.ops.aten.mm.default(view_937, permute_773);  permute_773 = None
    permute_774: "f32[384, 512]" = torch.ops.aten.permute.default(view_937, [1, 0])
    mm_191: "f32[384, 768]" = torch.ops.aten.mm.default(permute_774, view_36);  permute_774 = view_36 = None
    permute_775: "f32[768, 384]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_243: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_937, [0], True);  view_937 = None
    view_938: "f32[384]" = torch.ops.aten.view.default(sum_243, [384]);  sum_243 = None
    permute_776: "f32[384, 768]" = torch.ops.aten.permute.default(permute_775, [1, 0]);  permute_775 = None
    view_939: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_190, [1, 512, 768]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_274: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_273, view_939);  add_273 = view_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_145: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_13);  add_13 = getitem_13 = None
    mul_483: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_2);  sub_145 = None
    mul_484: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_274, primals_38);  primals_38 = None
    mul_485: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_484, 768)
    sum_244: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True)
    mul_486: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_484, mul_483);  mul_484 = None
    sum_245: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_486, [2], True);  mul_486 = None
    mul_487: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_483, sum_245);  sum_245 = None
    sub_146: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_485, sum_244);  mul_485 = sum_244 = None
    sub_147: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_146, mul_487);  sub_146 = mul_487 = None
    div_72: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_488: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_72, sub_147);  div_72 = sub_147 = None
    mul_489: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_274, mul_483);  mul_483 = None
    sum_246: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_489, [0, 1]);  mul_489 = None
    sum_247: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_490: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_491: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_488, mul_490);  mul_490 = None
    clone_102: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_491, memory_format = torch.contiguous_format);  mul_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_940: "f32[512, 768]" = torch.ops.aten.view.default(clone_102, [512, 768]);  clone_102 = None
    permute_777: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_192: "f32[512, 3072]" = torch.ops.aten.mm.default(view_940, permute_777);  permute_777 = None
    permute_778: "f32[768, 512]" = torch.ops.aten.permute.default(view_940, [1, 0])
    mm_193: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_778, view_34);  permute_778 = view_34 = None
    permute_779: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_248: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_940, [0], True);  view_940 = None
    view_941: "f32[768]" = torch.ops.aten.view.default(sum_248, [768]);  sum_248 = None
    permute_780: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_779, [1, 0]);  permute_779 = None
    view_942: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_192, [1, 512, 3072]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_492: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_25: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_492);  mul_492 = None
    add_275: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_493: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_275, 0.5);  add_275 = None
    mul_494: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, view_33)
    mul_495: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_494, -0.5);  mul_494 = None
    exp_38: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_495);  mul_495 = None
    mul_496: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_497: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, mul_496);  view_33 = mul_496 = None
    add_276: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_493, mul_497);  mul_493 = mul_497 = None
    mul_498: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_942, add_276);  view_942 = add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_943: "f32[512, 3072]" = torch.ops.aten.view.default(mul_498, [512, 3072]);  mul_498 = None
    permute_781: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_194: "f32[512, 768]" = torch.ops.aten.mm.default(view_943, permute_781);  permute_781 = None
    permute_782: "f32[3072, 512]" = torch.ops.aten.permute.default(view_943, [1, 0])
    mm_195: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_782, view_32);  permute_782 = view_32 = None
    permute_783: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_249: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_943, [0], True);  view_943 = None
    view_944: "f32[3072]" = torch.ops.aten.view.default(sum_249, [3072]);  sum_249 = None
    permute_784: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_783, [1, 0]);  permute_783 = None
    view_945: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_194, [1, 512, 768]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    add_277: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_488, view_945);  mul_488 = view_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_148: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_9);  add_9 = getitem_9 = None
    mul_499: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_1);  sub_148 = None
    mul_500: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_277, primals_32);  primals_32 = None
    mul_501: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_500, 768)
    sum_250: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [2], True)
    mul_502: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_500, mul_499);  mul_500 = None
    sum_251: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [2], True);  mul_502 = None
    mul_503: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_499, sum_251);  sum_251 = None
    sub_149: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_501, sum_250);  mul_501 = sum_250 = None
    sub_150: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_149, mul_503);  sub_149 = mul_503 = None
    div_73: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_504: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_73, sub_150);  div_73 = sub_150 = None
    mul_505: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_277, mul_499);  mul_499 = None
    sum_252: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 1]);  mul_505 = None
    sum_253: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_277, [0, 1]);  add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_506: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_507: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_504, mul_506);  mul_506 = None
    clone_103: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_507, memory_format = torch.contiguous_format);  mul_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_946: "f32[512, 768]" = torch.ops.aten.view.default(clone_103, [512, 768]);  clone_103 = None
    permute_785: "f32[768, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm_196: "f32[512, 768]" = torch.ops.aten.mm.default(view_946, permute_785);  permute_785 = None
    permute_786: "f32[768, 512]" = torch.ops.aten.permute.default(view_946, [1, 0])
    mm_197: "f32[768, 768]" = torch.ops.aten.mm.default(permute_786, view_30);  permute_786 = view_30 = None
    permute_787: "f32[768, 768]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_254: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_946, [0], True);  view_946 = None
    view_947: "f32[768]" = torch.ops.aten.view.default(sum_254, [768]);  sum_254 = None
    permute_788: "f32[768, 768]" = torch.ops.aten.permute.default(permute_787, [1, 0]);  permute_787 = None
    view_948: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_196, [1, 512, 768]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_949: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_948, [1, 512, 12, 64]);  view_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    slice_51: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_949, 2, 0, 6)
    slice_52: "f32[1, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_949, 2, 6, 12);  view_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_950: "f32[512, 384]" = torch.ops.aten.view.default(slice_52, [512, 384]);  slice_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_789: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(slice_51, [0, 2, 1, 3]);  slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_951: "f32[6, 512, 64]" = torch.ops.aten.view.default(permute_789, [6, 512, 64]);  permute_789 = None
    permute_790: "f32[6, 512, 512]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    bmm_102: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(permute_790, view_951);  permute_790 = None
    permute_791: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    bmm_103: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_951, permute_791);  view_951 = permute_791 = None
    view_952: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_102, [1, 6, 512, 64]);  bmm_102 = None
    view_953: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_103, [1, 6, 512, 512]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_36: "f32[1, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_508: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_509: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(view_953, mul_508);  view_953 = mul_508 = None
    clone_104: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(mul_509, memory_format = torch.contiguous_format);  mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_48: "f32[1, 6, 512, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_510: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(clone_104, alias_48);  clone_104 = None
    sum_255: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_510, [-1], True)
    mul_511: "f32[1, 6, 512, 512]" = torch.ops.aten.mul.Tensor(alias_48, sum_255);  alias_48 = sum_255 = None
    sub_151: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_510, mul_511);  mul_510 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_74: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_151, 8.0);  sub_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_954: "f32[6, 512, 512]" = torch.ops.aten.view.default(div_74, [6, 512, 512]);  div_74 = None
    permute_792: "f32[6, 64, 512]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
    bmm_104: "f32[6, 64, 512]" = torch.ops.aten.bmm.default(permute_792, view_954);  permute_792 = None
    permute_793: "f32[6, 512, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1]);  view_23 = None
    bmm_105: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_954, permute_793);  view_954 = permute_793 = None
    view_955: "f32[1, 6, 64, 512]" = torch.ops.aten.view.default(bmm_104, [1, 6, 64, 512]);  bmm_104 = None
    view_956: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_105, [1, 6, 512, 64]);  bmm_105 = None
    permute_794: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_955, [0, 1, 3, 2]);  view_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    clone_105: "f32[512, 384]" = torch.ops.aten.clone.default(view_950, memory_format = torch.contiguous_format);  view_950 = None
    view_957: "f32[3072, 64, 1]" = torch.ops.aten.view.default(clone_105, [3072, 64, 1]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    view_958: "f32[3072, 64, 1]" = torch.ops.aten.view.default(view_957, [3072, 64, 1]);  view_957 = None
    permute_795: "f32[3072, 9, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
    bmm_106: "f32[3072, 9, 1]" = torch.ops.aten.bmm.default(permute_795, view_958);  permute_795 = None
    permute_796: "f32[3072, 1, 9]" = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None
    bmm_107: "f32[3072, 64, 9]" = torch.ops.aten.bmm.default(view_958, permute_796);  view_958 = permute_796 = None
    view_959: "f32[3072, 9, 1]" = torch.ops.aten.view.default(bmm_106, [3072, 9, 1]);  bmm_106 = None
    view_960: "f32[3072, 64, 9]" = torch.ops.aten.view.default(bmm_107, [3072, 64, 9]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    view_961: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(view_960, [1, 512, 384, 9]);  view_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    view_962: "f32[1, 512, 3456]" = torch.ops.aten.view.default(view_961, [1, 512, 3456]);  view_961 = None
    permute_797: "f32[1, 3456, 512]" = torch.ops.aten.permute.default(view_962, [0, 2, 1]);  view_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    view_963: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(permute_797, [1, 384, 9, 1, 512, 1]);  permute_797 = None
    permute_798: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_963, [0, 1, 2, 4, 3, 5]);  view_963 = None
    iota_92: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_154: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_92, 0);  iota_92 = None
    iota_93: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_155: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_93, -1);  iota_93 = None
    add_278: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_154, unsqueeze_155);  unsqueeze_154 = unsqueeze_155 = None
    unsqueeze_156: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_278, -1);  add_278 = None
    unsqueeze_157: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    iota_94: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_158: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_94, 0);  iota_94 = None
    iota_95: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_159: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_95, -1);  iota_95 = None
    add_279: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_158, unsqueeze_159);  unsqueeze_158 = unsqueeze_159 = None
    full_13: "f32[1, 384, 520, 1]" = torch.ops.aten.full.default([1, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_11: "f32[1, 384, 520, 1]" = torch.ops.aten._unsafe_index_put.default(full_13, [None, None, unsqueeze_157, add_279], permute_798, True);  full_13 = unsqueeze_157 = add_279 = permute_798 = None
    constant_pad_nd_23: "f32[1, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_11, [0, 0, -4, -4], 0.0);  _unsafe_index_put_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    squeeze_12: "f32[1, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_23, -1);  constant_pad_nd_23 = None
    permute_799: "f32[1, 512, 384]" = torch.ops.aten.permute.default(squeeze_12, [0, 2, 1]);  squeeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_964: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_799, [1, 512, 384]);  permute_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_965: "f32[512, 384]" = torch.ops.aten.view.default(view_964, [512, 384]);  view_964 = None
    permute_800: "f32[384, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_198: "f32[512, 768]" = torch.ops.aten.mm.default(view_965, permute_800);  permute_800 = None
    permute_801: "f32[384, 512]" = torch.ops.aten.permute.default(view_965, [1, 0])
    mm_199: "f32[384, 768]" = torch.ops.aten.mm.default(permute_801, view_12);  permute_801 = view_12 = None
    permute_802: "f32[768, 384]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_256: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
    view_966: "f32[384]" = torch.ops.aten.view.default(sum_256, [384]);  sum_256 = None
    permute_803: "f32[384, 768]" = torch.ops.aten.permute.default(permute_802, [1, 0]);  permute_802 = None
    view_967: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_198, [1, 512, 768]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    add_280: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_504, view_967);  mul_504 = view_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    alias_49: "f32[3072, 9, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_512: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(view_959, alias_49);  view_959 = None
    sum_257: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [1], True)
    mul_513: "f32[3072, 9, 1]" = torch.ops.aten.mul.Tensor(alias_49, sum_257);  alias_49 = sum_257 = None
    sub_152: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_968: "f32[1, 512, 54]" = torch.ops.aten.view.default(sub_152, [1, 512, 54]);  sub_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    sum_258: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_968, [0, 1], True)
    view_969: "f32[54]" = torch.ops.aten.view.default(sum_258, [54]);  sum_258 = None
    view_970: "f32[512, 54]" = torch.ops.aten.view.default(view_968, [512, 54]);  view_968 = None
    permute_804: "f32[54, 512]" = torch.ops.aten.permute.default(view_970, [1, 0])
    mm_200: "f32[54, 384]" = torch.ops.aten.mm.default(permute_804, view_9);  permute_804 = view_9 = None
    permute_805: "f32[384, 54]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    permute_806: "f32[54, 512]" = torch.ops.aten.permute.default(view_970, [1, 0]);  view_970 = None
    mm_201: "f32[384, 512]" = torch.ops.aten.mm.default(permute_9, permute_806);  permute_9 = permute_806 = None
    permute_807: "f32[512, 384]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    view_971: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_807, [1, 512, 384]);  permute_807 = None
    permute_808: "f32[54, 384]" = torch.ops.aten.permute.default(permute_805, [1, 0]);  permute_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    mul_514: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_971, permute_8);  permute_8 = None
    mul_515: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(view_971, view_1);  view_971 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_809: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_952, [0, 2, 1, 3]);  view_952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_106: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_809, memory_format = torch.contiguous_format);  permute_809 = None
    view_972: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_106, [1, 512, 384]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_810: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(permute_794, [0, 2, 1, 3]);  permute_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_973: "f32[1, 512, 384]" = torch.ops.aten.view.default(permute_810, [1, 512, 384]);  permute_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_811: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_956, [0, 2, 1, 3]);  view_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    clone_107: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_811, memory_format = torch.contiguous_format);  permute_811 = None
    view_974: "f32[1, 512, 384]" = torch.ops.aten.view.default(clone_107, [1, 512, 384]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    add_281: "f32[1, 512, 384]" = torch.ops.aten.add.Tensor(mul_514, view_974);  mul_514 = view_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:364, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
    permute_812: "f32[1, 384, 512]" = torch.ops.aten.permute.default(mul_515, [0, 2, 1]);  mul_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    sum_259: "f32[1, 384, 1]" = torch.ops.aten.sum.dim_IntList(permute_812, [0, 2], True)
    view_975: "f32[384, 1]" = torch.ops.aten.view.default(sum_259, [384, 1]);  sum_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(permute_812, convolution, primals_25, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_812 = convolution = primals_25 = None
    getitem_192: "f32[1, 768, 512]" = convolution_backward_22[0]
    getitem_193: "f32[384, 768, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(getitem_192, permute_3, primals_24, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_192 = permute_3 = primals_24 = None
    getitem_195: "f32[1, 768, 512]" = convolution_backward_23[0]
    getitem_196: "f32[768, 1, 9]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_813: "f32[1, 512, 768]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1]);  getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    add_282: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_280, permute_813);  add_280 = permute_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_976: "f32[512, 384]" = torch.ops.aten.view.default(view_972, [512, 384]);  view_972 = None
    permute_814: "f32[384, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_202: "f32[512, 768]" = torch.ops.aten.mm.default(view_976, permute_814);  permute_814 = None
    permute_815: "f32[384, 512]" = torch.ops.aten.permute.default(view_976, [1, 0])
    mm_203: "f32[384, 768]" = torch.ops.aten.mm.default(permute_815, view_4);  permute_815 = view_4 = None
    permute_816: "f32[768, 384]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_260: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True);  view_976 = None
    view_977: "f32[384]" = torch.ops.aten.view.default(sum_260, [384]);  sum_260 = None
    permute_817: "f32[384, 768]" = torch.ops.aten.permute.default(permute_816, [1, 0]);  permute_816 = None
    view_978: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_202, [1, 512, 768]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    add_283: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_282, view_978);  add_282 = view_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_979: "f32[512, 384]" = torch.ops.aten.view.default(view_973, [512, 384]);  view_973 = None
    permute_818: "f32[384, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_204: "f32[512, 768]" = torch.ops.aten.mm.default(view_979, permute_818);  permute_818 = None
    permute_819: "f32[384, 512]" = torch.ops.aten.permute.default(view_979, [1, 0])
    mm_205: "f32[384, 768]" = torch.ops.aten.mm.default(permute_819, view_2);  permute_819 = view_2 = None
    permute_820: "f32[768, 384]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_261: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_979, [0], True);  view_979 = None
    view_980: "f32[384]" = torch.ops.aten.view.default(sum_261, [384]);  sum_261 = None
    permute_821: "f32[384, 768]" = torch.ops.aten.permute.default(permute_820, [1, 0]);  permute_820 = None
    view_981: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_204, [1, 512, 768]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    add_284: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_283, view_981);  add_283 = view_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_982: "f32[512, 384]" = torch.ops.aten.view.default(add_281, [512, 384]);  add_281 = None
    permute_822: "f32[384, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_206: "f32[512, 768]" = torch.ops.aten.mm.default(view_982, permute_822);  permute_822 = None
    permute_823: "f32[384, 512]" = torch.ops.aten.permute.default(view_982, [1, 0])
    mm_207: "f32[384, 768]" = torch.ops.aten.mm.default(permute_823, view);  permute_823 = view = None
    permute_824: "f32[768, 384]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_262: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
    view_983: "f32[384]" = torch.ops.aten.view.default(sum_262, [384]);  sum_262 = None
    permute_825: "f32[384, 768]" = torch.ops.aten.permute.default(permute_824, [1, 0]);  permute_824 = None
    view_984: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_206, [1, 512, 768]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    add_285: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_284, view_984);  add_284 = view_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:236, code: embeddings = self.dropout(embeddings)
    convert_element_type_37: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_516: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_517: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_285, mul_516);  add_285 = mul_516 = None
    clone_108: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_517, memory_format = torch.contiguous_format);  mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:235, code: embeddings = self.LayerNorm(embeddings)
    sub_153: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_518: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_153, rsqrt);  sub_153 = None
    mul_519: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_108, primals_16);  primals_16 = None
    mul_520: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_519, 768)
    sum_263: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_519, [2], True)
    mul_521: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_519, mul_518);  mul_519 = None
    sum_264: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_521, [2], True);  mul_521 = None
    mul_522: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_518, sum_264);  sum_264 = None
    sub_154: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_520, sum_263);  mul_520 = sum_263 = None
    sub_155: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_154, mul_522);  sub_154 = mul_522 = None
    div_75: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_523: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_75, sub_155);  div_75 = sub_155 = None
    mul_524: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_108, mul_518);  mul_518 = None
    sum_265: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 1]);  mul_524 = None
    sum_266: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_108, [0, 1]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:232, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_160: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_160, scalar_tensor_4, mul_523);  unsqueeze_160 = scalar_tensor_4 = None
    full_14: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_12: "f32[2, 768]" = torch.ops.aten._unsafe_index_put.default(full_14, [expand], where_4, True);  full_14 = expand = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:231, code: position_embeddings = self.position_embeddings(position_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_4, -1)
    unsqueeze_161: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_161, scalar_tensor_5, mul_523);  unsqueeze_161 = scalar_tensor_5 = None
    full_15: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_13: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_15, [slice_4], where_5, True);  full_15 = slice_4 = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:230, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_290, 0)
    unsqueeze_162: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_162, scalar_tensor_6, mul_523);  unsqueeze_162 = scalar_tensor_6 = mul_523 = None
    full_16: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_14: "f32[30522, 768]" = torch.ops.aten._unsafe_index_put.default(full_16, [primals_290], where_6, True);  full_16 = primals_290 = where_6 = None
    return pytree.tree_unflatten([div_36, view_435, view_975, view_930, view_885, view_840, view_795, view_750, view_705, view_660, view_615, view_570, view_525, view_480, _unsafe_index_put_14, _unsafe_index_put_13, _unsafe_index_put_12, sum_265, sum_266, permute_825, view_983, permute_821, view_980, permute_817, view_977, getitem_196, getitem_193, permute_808, view_969, permute_803, view_966, permute_788, view_947, sum_252, sum_253, permute_784, view_944, permute_780, view_941, sum_246, sum_247, permute_776, view_938, permute_772, view_935, permute_768, view_932, getitem_190, getitem_187, permute_759, view_924, permute_754, view_921, permute_739, view_902, sum_233, sum_234, permute_735, view_899, permute_731, view_896, sum_227, sum_228, permute_727, view_893, permute_723, view_890, permute_719, view_887, getitem_184, getitem_181, permute_710, view_879, permute_705, view_876, permute_690, view_857, sum_214, sum_215, permute_686, view_854, permute_682, view_851, sum_208, sum_209, permute_678, view_848, permute_674, view_845, permute_670, view_842, getitem_178, getitem_175, permute_661, view_834, permute_656, view_831, permute_641, view_812, sum_195, sum_196, permute_637, view_809, permute_633, view_806, sum_189, sum_190, permute_629, view_803, permute_625, view_800, permute_621, view_797, getitem_172, getitem_169, permute_612, view_789, permute_607, view_786, permute_592, view_767, sum_176, sum_177, permute_588, view_764, permute_584, view_761, sum_170, sum_171, permute_580, view_758, permute_576, view_755, permute_572, view_752, getitem_166, getitem_163, permute_563, view_744, permute_558, view_741, permute_543, view_722, sum_157, sum_158, permute_539, view_719, permute_535, view_716, sum_151, sum_152, permute_531, view_713, permute_527, view_710, permute_523, view_707, getitem_160, getitem_157, permute_514, view_699, permute_509, view_696, permute_494, view_677, sum_138, sum_139, permute_490, view_674, permute_486, view_671, sum_132, sum_133, permute_482, view_668, permute_478, view_665, permute_474, view_662, getitem_154, getitem_151, permute_465, view_654, permute_460, view_651, permute_445, view_632, sum_119, sum_120, permute_441, view_629, permute_437, view_626, sum_113, sum_114, permute_433, view_623, permute_429, view_620, permute_425, view_617, getitem_148, getitem_145, permute_416, view_609, permute_411, view_606, permute_396, view_587, sum_100, sum_101, permute_392, view_584, permute_388, view_581, sum_94, sum_95, permute_384, view_578, permute_380, view_575, permute_376, view_572, getitem_142, getitem_139, permute_367, view_564, permute_362, view_561, permute_347, view_542, sum_81, sum_82, permute_343, view_539, permute_339, view_536, sum_75, sum_76, permute_335, view_533, permute_331, view_530, permute_327, view_527, getitem_136, getitem_133, permute_318, view_519, permute_313, view_516, permute_298, view_497, sum_62, sum_63, permute_294, view_494, permute_290, view_491, sum_56, sum_57, permute_286, view_488, permute_282, view_485, permute_278, view_482, getitem_130, getitem_127, permute_269, view_474, permute_264, view_471, permute_249, view_452, sum_43, sum_44, permute_245, view_449, permute_241, view_446, sum_37, sum_38, permute_237, view_443, sum_32, sum_33, permute_233, view_440, None, None, None, None], self._out_spec)
    