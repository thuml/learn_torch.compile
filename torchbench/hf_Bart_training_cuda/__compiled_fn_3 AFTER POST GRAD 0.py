from __future__ import annotations



def forward(self, primals_1: "f32[1026, 768]", primals_2: "f32[1026, 768]", primals_3: "f32[50265, 768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "f32[768, 768]", primals_9: "f32[768]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[3072, 768]", primals_17: "f32[3072]", primals_18: "f32[768, 3072]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768, 768]", primals_23: "f32[768]", primals_24: "f32[768, 768]", primals_25: "f32[768]", primals_26: "f32[768, 768]", primals_27: "f32[768]", primals_28: "f32[768, 768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[3072, 768]", primals_33: "f32[3072]", primals_34: "f32[768, 3072]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[768, 768]", primals_39: "f32[768]", primals_40: "f32[768, 768]", primals_41: "f32[768]", primals_42: "f32[768, 768]", primals_43: "f32[768]", primals_44: "f32[768, 768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[3072, 768]", primals_49: "f32[3072]", primals_50: "f32[768, 3072]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768, 768]", primals_55: "f32[768]", primals_56: "f32[768, 768]", primals_57: "f32[768]", primals_58: "f32[768, 768]", primals_59: "f32[768]", primals_60: "f32[768, 768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[3072, 768]", primals_65: "f32[3072]", primals_66: "f32[768, 3072]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768, 768]", primals_71: "f32[768]", primals_72: "f32[768, 768]", primals_73: "f32[768]", primals_74: "f32[768, 768]", primals_75: "f32[768]", primals_76: "f32[768, 768]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[3072, 768]", primals_81: "f32[3072]", primals_82: "f32[768, 3072]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[768, 768]", primals_87: "f32[768]", primals_88: "f32[768, 768]", primals_89: "f32[768]", primals_90: "f32[768, 768]", primals_91: "f32[768]", primals_92: "f32[768, 768]", primals_93: "f32[768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[3072, 768]", primals_97: "f32[3072]", primals_98: "f32[768, 3072]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[50265, 768]", primals_103: "f32[768]", primals_104: "f32[768]", primals_105: "f32[768, 768]", primals_106: "f32[768]", primals_107: "f32[768, 768]", primals_108: "f32[768]", primals_109: "f32[768, 768]", primals_110: "f32[768]", primals_111: "f32[768, 768]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[768, 768]", primals_116: "f32[768]", primals_117: "f32[768, 768]", primals_118: "f32[768]", primals_119: "f32[768, 768]", primals_120: "f32[768]", primals_121: "f32[768, 768]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768]", primals_125: "f32[3072, 768]", primals_126: "f32[3072]", primals_127: "f32[768, 3072]", primals_128: "f32[768]", primals_129: "f32[768]", primals_130: "f32[768]", primals_131: "f32[768, 768]", primals_132: "f32[768]", primals_133: "f32[768, 768]", primals_134: "f32[768]", primals_135: "f32[768, 768]", primals_136: "f32[768]", primals_137: "f32[768, 768]", primals_138: "f32[768]", primals_139: "f32[768]", primals_140: "f32[768]", primals_141: "f32[768, 768]", primals_142: "f32[768]", primals_143: "f32[768, 768]", primals_144: "f32[768]", primals_145: "f32[768, 768]", primals_146: "f32[768]", primals_147: "f32[768, 768]", primals_148: "f32[768]", primals_149: "f32[768]", primals_150: "f32[768]", primals_151: "f32[3072, 768]", primals_152: "f32[3072]", primals_153: "f32[768, 3072]", primals_154: "f32[768]", primals_155: "f32[768]", primals_156: "f32[768]", primals_157: "f32[768, 768]", primals_158: "f32[768]", primals_159: "f32[768, 768]", primals_160: "f32[768]", primals_161: "f32[768, 768]", primals_162: "f32[768]", primals_163: "f32[768, 768]", primals_164: "f32[768]", primals_165: "f32[768]", primals_166: "f32[768]", primals_167: "f32[768, 768]", primals_168: "f32[768]", primals_169: "f32[768, 768]", primals_170: "f32[768]", primals_171: "f32[768, 768]", primals_172: "f32[768]", primals_173: "f32[768, 768]", primals_174: "f32[768]", primals_175: "f32[768]", primals_176: "f32[768]", primals_177: "f32[3072, 768]", primals_178: "f32[3072]", primals_179: "f32[768, 3072]", primals_180: "f32[768]", primals_181: "f32[768]", primals_182: "f32[768]", primals_183: "f32[768, 768]", primals_184: "f32[768]", primals_185: "f32[768, 768]", primals_186: "f32[768]", primals_187: "f32[768, 768]", primals_188: "f32[768]", primals_189: "f32[768, 768]", primals_190: "f32[768]", primals_191: "f32[768]", primals_192: "f32[768]", primals_193: "f32[768, 768]", primals_194: "f32[768]", primals_195: "f32[768, 768]", primals_196: "f32[768]", primals_197: "f32[768, 768]", primals_198: "f32[768]", primals_199: "f32[768, 768]", primals_200: "f32[768]", primals_201: "f32[768]", primals_202: "f32[768]", primals_203: "f32[3072, 768]", primals_204: "f32[3072]", primals_205: "f32[768, 3072]", primals_206: "f32[768]", primals_207: "f32[768]", primals_208: "f32[768]", primals_209: "f32[768, 768]", primals_210: "f32[768]", primals_211: "f32[768, 768]", primals_212: "f32[768]", primals_213: "f32[768, 768]", primals_214: "f32[768]", primals_215: "f32[768, 768]", primals_216: "f32[768]", primals_217: "f32[768]", primals_218: "f32[768]", primals_219: "f32[768, 768]", primals_220: "f32[768]", primals_221: "f32[768, 768]", primals_222: "f32[768]", primals_223: "f32[768, 768]", primals_224: "f32[768]", primals_225: "f32[768, 768]", primals_226: "f32[768]", primals_227: "f32[768]", primals_228: "f32[768]", primals_229: "f32[3072, 768]", primals_230: "f32[3072]", primals_231: "f32[768, 3072]", primals_232: "f32[768]", primals_233: "f32[768]", primals_234: "f32[768]", primals_235: "f32[768, 768]", primals_236: "f32[768]", primals_237: "f32[768, 768]", primals_238: "f32[768]", primals_239: "f32[768, 768]", primals_240: "f32[768]", primals_241: "f32[768, 768]", primals_242: "f32[768]", primals_243: "f32[768]", primals_244: "f32[768]", primals_245: "f32[768, 768]", primals_246: "f32[768]", primals_247: "f32[768, 768]", primals_248: "f32[768]", primals_249: "f32[768, 768]", primals_250: "f32[768]", primals_251: "f32[768, 768]", primals_252: "f32[768]", primals_253: "f32[768]", primals_254: "f32[768]", primals_255: "f32[3072, 768]", primals_256: "f32[3072]", primals_257: "f32[768, 3072]", primals_258: "f32[768]", primals_259: "f32[768]", primals_260: "f32[768]", primals_261: "f32[50265, 768]", primals_262: "f32[1, 50265]", primals_263: "i64[4, 512]", primals_264: "i64[4, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:811, code: input_ids = input_ids.view(-1, input_ids.shape[-1])
    view: "i64[4, 512]" = torch.ops.aten.reshape.default(primals_263, [-1, 512]);  primals_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:818, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_3, view, 1);  primals_3 = None
    mul: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:135, code: positions = torch.arange(
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:137, code: ).expand(bsz, -1)
    expand: "i64[4, 512]" = torch.ops.aten.expand.default(iota, [4, -1])
    
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
    sub: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_4)
    add_3: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_1: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_3, [2048, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    
    # No stacktrace found for following nodes
    mm_default_84: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1, permute)
    add_tensor_83: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_84, primals_7);  mm_default_84 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_2: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_83, [4, 512, 768]);  add_tensor_83 = None
    mul_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_2, 0.125);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    
    # No stacktrace found for following nodes
    mm_default_83: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1, permute_1)
    add_tensor_82: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_83, primals_9);  mm_default_83 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_4: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_82, [4, 512, 768]);  add_tensor_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [4, -1, 12, 64]);  view_4 = None
    permute_2: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    clone_1: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    
    # No stacktrace found for following nodes
    mm_default_82: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1, permute_3)
    add_tensor_81: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_82, primals_11);  mm_default_82 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_7: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_81, [4, 512, 768]);  add_tensor_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_7, [4, -1, 12, 64]);  view_7 = None
    permute_4: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_3, [4, 512, 12, 64]);  mul_3 = None
    permute_5: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_3: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_10: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_3, [48, -1, 64]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_11: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_1, [48, -1, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_12: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_2, [48, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_10, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm, [-1], True)
    sub_1: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm, amax)
    exp: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div, view_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_13: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [4, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_14: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_5, [4, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_15: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_14, [2048, 768]);  view_14 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    
    # No stacktrace found for following nodes
    mm_default_81: "f32[2048, 768]" = torch.ops.aten.mm.default(view_15, permute_8)
    add_tensor_80: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_81, primals_13);  mm_default_81 = primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_16: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_80, [4, 512, 768]);  add_tensor_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_4: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_3, view_16);  add_3 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_2: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  add_4 = getitem_3 = None
    mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, primals_14)
    add_6: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, primals_15);  mul_5 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_17: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_6, [2048, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_4: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_17, view_17, permute_9);  primals_17 = None
    view_18: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_7: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_7: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_7);  mul_6 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_19: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_8, [2048, 3072]);  mul_8 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    
    # No stacktrace found for following nodes
    mm_default_80: "f32[2048, 768]" = torch.ops.aten.mm.default(view_19, permute_10)
    add_tensor_79: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_80, primals_19);  mm_default_80 = primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_20: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_79, [4, 512, 768]);  add_tensor_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_8: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_6, view_20);  add_6 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  add_8 = getitem_5 = None
    mul_9: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_10: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_20)
    add_10: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_21);  mul_10 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_21: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_10, [2048, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    
    # No stacktrace found for following nodes
    mm_default_79: "f32[2048, 768]" = torch.ops.aten.mm.default(view_21, permute_11)
    add_tensor_78: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_79, primals_23);  mm_default_79 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_22: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_78, [4, 512, 768]);  add_tensor_78 = None
    mul_11: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_22, 0.125);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    
    # No stacktrace found for following nodes
    mm_default_78: "f32[2048, 768]" = torch.ops.aten.mm.default(view_21, permute_12)
    add_tensor_77: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_78, primals_25);  mm_default_78 = primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_24: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_77, [4, 512, 768]);  add_tensor_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_24, [4, -1, 12, 64]);  view_24 = None
    permute_13: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_9: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    
    # No stacktrace found for following nodes
    mm_default_77: "f32[2048, 768]" = torch.ops.aten.mm.default(view_21, permute_14)
    add_tensor_76: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_77, primals_27);  mm_default_77 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_27: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_76, [4, 512, 768]);  add_tensor_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_27, [4, -1, 12, 64]);  view_27 = None
    permute_15: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_10: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_11, [4, 512, 12, 64]);  mul_11 = None
    permute_16: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_11: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_30: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_11, [48, -1, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_31: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_9, [48, -1, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_32: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_10, [48, -1, 64]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    bmm_2: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_30, permute_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_2, [-1], True)
    sub_4: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1)
    exp_1: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_1, view_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_33: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [4, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_34: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_13, [4, 512, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_35: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_34, [2048, 768]);  view_34 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    
    # No stacktrace found for following nodes
    mm_default_76: "f32[2048, 768]" = torch.ops.aten.mm.default(view_35, permute_19)
    add_tensor_75: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_76, primals_29);  mm_default_76 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_36: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_75, [4, 512, 768]);  add_tensor_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_11: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_10, view_36);  add_10 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[4, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  add_11 = getitem_7 = None
    mul_12: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_13: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, primals_30)
    add_13: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_13, primals_31);  mul_13 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_37: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_13, [2048, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_10: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_33, view_37, permute_20);  primals_33 = None
    view_38: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_15: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_1: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_14: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_14);  mul_14 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_39: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_16, [2048, 3072]);  mul_16 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    
    # No stacktrace found for following nodes
    mm_default_75: "f32[2048, 768]" = torch.ops.aten.mm.default(view_39, permute_21)
    add_tensor_74: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_75, primals_35);  mm_default_75 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_40: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_74, [4, 512, 768]);  add_tensor_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_15: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_13, view_40);  add_13 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_8: "f32[4, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_16: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_9);  add_15 = getitem_9 = None
    mul_17: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_36)
    add_17: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_37);  mul_18 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_41: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_17, [2048, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    
    # No stacktrace found for following nodes
    mm_default_74: "f32[2048, 768]" = torch.ops.aten.mm.default(view_41, permute_22)
    add_tensor_73: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_74, primals_39);  mm_default_74 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_42: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_73, [4, 512, 768]);  add_tensor_73 = None
    mul_19: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_42, 0.125);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    
    # No stacktrace found for following nodes
    mm_default_73: "f32[2048, 768]" = torch.ops.aten.mm.default(view_41, permute_23)
    add_tensor_72: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_73, primals_41);  mm_default_73 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_44: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_72, [4, 512, 768]);  add_tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_45: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_44, [4, -1, 12, 64]);  view_44 = None
    permute_24: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_17: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    
    # No stacktrace found for following nodes
    mm_default_72: "f32[2048, 768]" = torch.ops.aten.mm.default(view_41, permute_25)
    add_tensor_71: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_72, primals_43);  mm_default_72 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_47: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_71, [4, 512, 768]);  add_tensor_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_48: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_47, [4, -1, 12, 64]);  view_47 = None
    permute_26: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_18: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_19, [4, 512, 12, 64]);  mul_19 = None
    permute_27: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_19: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_50: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_19, [48, -1, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_51: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_17, [48, -1, 64]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_52: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_18, [48, -1, 64]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    bmm_4: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_50, permute_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_4, [-1], True)
    sub_7: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_4, amax_2)
    exp_2: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_2, view_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_53: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [4, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_54: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_21, [4, 512, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_55: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_54, [2048, 768]);  view_54 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    
    # No stacktrace found for following nodes
    mm_default_71: "f32[2048, 768]" = torch.ops.aten.mm.default(view_55, permute_30)
    add_tensor_70: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_71, primals_45);  mm_default_71 = primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_56: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_70, [4, 512, 768]);  add_tensor_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_18: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_17, view_56);  add_17 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
    getitem_10: "f32[4, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_19: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_8: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_11);  add_18 = getitem_11 = None
    mul_20: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_21: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, primals_46)
    add_20: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_47);  mul_21 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_57: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_20, [2048, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_16: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_49, view_57, permute_31);  primals_49 = None
    view_58: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_23: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_2: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_21: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_21);  mul_22 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_59: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_24, [2048, 3072]);  mul_24 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    
    # No stacktrace found for following nodes
    mm_default_70: "f32[2048, 768]" = torch.ops.aten.mm.default(view_59, permute_32)
    add_tensor_69: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_70, primals_51);  mm_default_70 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_60: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_69, [4, 512, 768]);  add_tensor_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_22: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_20, view_60);  add_20 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
    getitem_12: "f32[4, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_23: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_9: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_13);  add_22 = getitem_13 = None
    mul_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_26: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_52)
    add_24: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_53);  mul_26 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_61: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_24, [2048, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    
    # No stacktrace found for following nodes
    mm_default_69: "f32[2048, 768]" = torch.ops.aten.mm.default(view_61, permute_33)
    add_tensor_68: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_69, primals_55);  mm_default_69 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_62: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_68, [4, 512, 768]);  add_tensor_68 = None
    mul_27: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, 0.125);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    
    # No stacktrace found for following nodes
    mm_default_68: "f32[2048, 768]" = torch.ops.aten.mm.default(view_61, permute_34)
    add_tensor_67: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_68, primals_57);  mm_default_68 = primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_64: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_67, [4, 512, 768]);  add_tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_65: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_64, [4, -1, 12, 64]);  view_64 = None
    permute_35: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_25: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    
    # No stacktrace found for following nodes
    mm_default_67: "f32[2048, 768]" = torch.ops.aten.mm.default(view_61, permute_36)
    add_tensor_66: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_67, primals_59);  mm_default_67 = primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_67: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_66, [4, 512, 768]);  add_tensor_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [4, -1, 12, 64]);  view_67 = None
    permute_37: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    clone_26: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_27, [4, 512, 12, 64]);  mul_27 = None
    permute_38: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_27: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_70: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_27, [48, -1, 64]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_71: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_25, [48, -1, 64]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_72: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_26, [48, -1, 64]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
    bmm_6: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_70, permute_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_6, [-1], True)
    sub_10: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_6, amax_3)
    exp_3: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_3, view_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_73: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [4, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_74: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_29, [4, 512, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_75: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_74, [2048, 768]);  view_74 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    
    # No stacktrace found for following nodes
    mm_default_66: "f32[2048, 768]" = torch.ops.aten.mm.default(view_75, permute_41)
    add_tensor_65: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_66, primals_61);  mm_default_66 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_76: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_65, [4, 512, 768]);  add_tensor_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_25: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_24, view_76);  add_24 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_14: "f32[4, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_26: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_11: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_15);  add_25 = getitem_15 = None
    mul_28: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_29: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_62)
    add_27: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_63);  mul_29 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_77: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_27, [2048, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_22: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_65, view_77, permute_42);  primals_65 = None
    view_78: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_31: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_3: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_28: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_30, add_28);  mul_30 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_79: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_32, [2048, 3072]);  mul_32 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    
    # No stacktrace found for following nodes
    mm_default_65: "f32[2048, 768]" = torch.ops.aten.mm.default(view_79, permute_43)
    add_tensor_64: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_65, primals_67);  mm_default_65 = primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_80: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_64, [4, 512, 768]);  add_tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_29: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_27, view_80);  add_27 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_16: "f32[4, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_30: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_12: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_17);  add_29 = getitem_17 = None
    mul_33: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_34: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_68)
    add_31: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_69);  mul_34 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_81: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_31, [2048, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    
    # No stacktrace found for following nodes
    mm_default_64: "f32[2048, 768]" = torch.ops.aten.mm.default(view_81, permute_44)
    add_tensor_63: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_64, primals_71);  mm_default_64 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_82: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_63, [4, 512, 768]);  add_tensor_63 = None
    mul_35: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_82, 0.125);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    
    # No stacktrace found for following nodes
    mm_default_63: "f32[2048, 768]" = torch.ops.aten.mm.default(view_81, permute_45)
    add_tensor_62: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_63, primals_73);  mm_default_63 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_84: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_62, [4, 512, 768]);  add_tensor_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_85: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_84, [4, -1, 12, 64]);  view_84 = None
    permute_46: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
    clone_33: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    
    # No stacktrace found for following nodes
    mm_default_62: "f32[2048, 768]" = torch.ops.aten.mm.default(view_81, permute_47)
    add_tensor_61: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_62, primals_75);  mm_default_62 = primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_87: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_61, [4, 512, 768]);  add_tensor_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_87, [4, -1, 12, 64]);  view_87 = None
    permute_48: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
    clone_34: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_35, [4, 512, 12, 64]);  mul_35 = None
    permute_49: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_35: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_90: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_35, [48, -1, 64]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_91: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_33, [48, -1, 64]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_92: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_34, [48, -1, 64]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    bmm_8: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_90, permute_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_8, [-1], True)
    sub_13: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_8, amax_4)
    exp_4: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_4, view_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_93: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [4, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_94: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_37, [4, 512, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_95: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_94, [2048, 768]);  view_94 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    
    # No stacktrace found for following nodes
    mm_default_61: "f32[2048, 768]" = torch.ops.aten.mm.default(view_95, permute_52)
    add_tensor_60: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_61, primals_77);  mm_default_61 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_96: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_60, [4, 512, 768]);  add_tensor_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_32: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_31, view_96);  add_31 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_18: "f32[4, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_33: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_14: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_19);  add_32 = getitem_19 = None
    mul_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_37: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, primals_78)
    add_34: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, primals_79);  mul_37 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_97: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_34, [2048, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_28: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_81, view_97, permute_53);  primals_81 = None
    view_98: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_39: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_4: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_35: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_35);  mul_38 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_99: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_40, [2048, 3072]);  mul_40 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    
    # No stacktrace found for following nodes
    mm_default_60: "f32[2048, 768]" = torch.ops.aten.mm.default(view_99, permute_54)
    add_tensor_59: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_60, primals_83);  mm_default_60 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_100: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_59, [4, 512, 768]);  add_tensor_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_36: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_34, view_100);  add_34 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_20: "f32[4, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_37: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_15: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_21);  add_36 = getitem_21 = None
    mul_41: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_42: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, primals_84)
    add_38: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, primals_85);  mul_42 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_101: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_38, [2048, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[2048, 768]" = torch.ops.aten.mm.default(view_101, permute_55)
    add_tensor_58: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_59, primals_87);  mm_default_59 = primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_102: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_58, [4, 512, 768]);  add_tensor_58 = None
    mul_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_102, 0.125);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[2048, 768]" = torch.ops.aten.mm.default(view_101, permute_56)
    add_tensor_57: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_58, primals_89);  mm_default_58 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_104: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_57, [4, 512, 768]);  add_tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_104, [4, -1, 12, 64]);  view_104 = None
    permute_57: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    clone_41: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[2048, 768]" = torch.ops.aten.mm.default(view_101, permute_58)
    add_tensor_56: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_57, primals_91);  mm_default_57 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_107: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_56, [4, 512, 768]);  add_tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_108: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_107, [4, -1, 12, 64]);  view_107 = None
    permute_59: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
    clone_42: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_109: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_43, [4, 512, 12, 64]);  mul_43 = None
    permute_60: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    clone_43: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_110: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_43, [48, -1, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_111: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_41, [48, -1, 64]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_112: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_42, [48, -1, 64]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    bmm_10: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_110, permute_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_10, [-1], True)
    sub_16: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_10, amax_5)
    exp_5: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_5, view_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_113: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [4, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_114: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_45, [4, 512, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_115: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_114, [2048, 768]);  view_114 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[2048, 768]" = torch.ops.aten.mm.default(view_115, permute_63)
    add_tensor_55: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_56, primals_93);  mm_default_56 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_116: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_55, [4, 512, 768]);  add_tensor_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_39: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_38, view_116);  add_38 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_22: "f32[4, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_40: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_17: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_23);  add_39 = getitem_23 = None
    mul_44: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_45: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_94)
    add_41: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_95);  mul_45 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_117: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_41, [2048, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_34: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_97, view_117, permute_64);  primals_97 = None
    view_118: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_47: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_5: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_42: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_42);  mul_46 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_119: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_48, [2048, 3072]);  mul_48 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[2048, 768]" = torch.ops.aten.mm.default(view_119, permute_65)
    add_tensor_54: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_55, primals_99);  mm_default_55 = primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_120: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_54, [4, 512, 768]);  add_tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_43: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_41, view_120);  add_41 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_24: "f32[4, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_44: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_18: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_25);  add_43 = getitem_25 = None
    mul_49: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_50: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_100)
    add_45: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_101);  mul_50 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1059, code: inputs_embeds = self.embed_tokens(input) * self.embed_scale
    embedding_2: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_102, primals_264, 1);  primals_102 = None
    mul_51: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:96, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:98, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_46: "i64[512]" = torch.ops.aten.add.Tensor(iota, 1)
    view_122: "i64[512, 1]" = torch.ops.aten.reshape.default(add_46, [512, 1]);  add_46 = None
    lt: "b8[512, 512]" = torch.ops.aten.lt.Tensor(iota, view_122);  iota = view_122 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[512, 512]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_3: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_2, add);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1074, code: hidden_states = inputs_embeds + positions
    add_48: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, embedding_3);  mul_51 = embedding_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1075, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_26: "f32[4, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_49: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_19: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_27);  add_48 = getitem_27 = None
    mul_52: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
    mul_53: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, primals_103)
    add_50: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, primals_104);  mul_53 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_123: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_50, [2048, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[2048, 768]" = torch.ops.aten.mm.default(view_123, permute_66)
    add_tensor_53: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_54, primals_106);  mm_default_54 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_124: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_53, [4, 512, 768]);  add_tensor_53 = None
    mul_54: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_124, 0.125);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[2048, 768]" = torch.ops.aten.mm.default(view_123, permute_67)
    add_tensor_52: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_53, primals_108);  mm_default_53 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_126: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_52, [4, 512, 768]);  add_tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_127: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_126, [4, -1, 12, 64]);  view_126 = None
    permute_68: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
    clone_50: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[2048, 768]" = torch.ops.aten.mm.default(view_123, permute_69)
    add_tensor_51: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_52, primals_110);  mm_default_52 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_129: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_51, [4, 512, 768]);  add_tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_130: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_129, [4, -1, 12, 64]);  view_129 = None
    permute_70: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    clone_51: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_131: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_54, [4, 512, 12, 64]);  mul_54 = None
    permute_71: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    clone_52: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_132: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_52, [48, -1, 64]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_133: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_50, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_134: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_51, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_12: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_132, permute_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_135: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [4, 12, 512, 512]);  bmm_12 = None
    unsqueeze_2: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    expand_3: "f32[4, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_3, [4, 1, 512, 512]);  unsqueeze_3 = None
    add_51: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_135, expand_3);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_136: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_51, [48, 512, 512]);  add_51 = None
    
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
    view_137: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [4, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_54: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_138: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_54, [4, 512, 768]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_139: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_138, [2048, 768]);  view_138 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[2048, 768]" = torch.ops.aten.mm.default(view_139, permute_74)
    add_tensor_50: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_51, primals_112);  mm_default_51 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_140: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_50, [4, 512, 768]);  add_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_52: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_50, view_140);  add_50 = view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_28: "f32[4, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_53: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_21: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_29);  add_52 = getitem_29 = None
    mul_55: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_56: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_55, primals_113)
    add_54: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_56, primals_114);  mul_56 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_141: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_54, [2048, 768])
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[2048, 768]" = torch.ops.aten.mm.default(view_141, permute_75)
    add_tensor_49: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_50, primals_116);  mm_default_50 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_142: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_49, [4, 512, 768]);  add_tensor_49 = None
    mul_57: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_142, 0.125);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_143: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_76: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_76)
    add_tensor_48: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_49, primals_118);  mm_default_49 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_144: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_48, [4, 512, 768]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_145: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_144, [4, -1, 12, 64]);  view_144 = None
    permute_77: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
    clone_56: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_78)
    add_tensor_47: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_48, primals_120);  mm_default_48 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_147: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_47, [4, 512, 768]);  add_tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_148: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_147, [4, -1, 12, 64]);  view_147 = None
    permute_79: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_57: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_57, [4, 512, 12, 64]);  mul_57 = None
    permute_80: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_58: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_150: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_58, [48, -1, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_151: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_56, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_152: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_57, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_81: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
    bmm_14: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_150, permute_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_14, [-1], True)
    sub_22: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_14, amax_7)
    exp_7: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_15: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_7, view_152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_153: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [4, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_82: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_60: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_154: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_60, [4, 512, 768]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_155: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_154, [2048, 768]);  view_154 = None
    permute_83: "f32[768, 768]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[2048, 768]" = torch.ops.aten.mm.default(view_155, permute_83)
    add_tensor_46: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_47, primals_122);  mm_default_47 = primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_156: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_46, [4, 512, 768]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_55: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_54, view_156);  add_54 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_30: "f32[4, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_56: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_23: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_31);  add_55 = getitem_31 = None
    mul_58: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_59: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_123)
    add_57: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_124);  mul_59 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_157: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_57, [2048, 768])
    permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_44: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_126, view_157, permute_84);  primals_126 = None
    view_158: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_44, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_60: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_6: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_58: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_62: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_58);  mul_60 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_159: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_62, [2048, 3072]);  mul_62 = None
    permute_85: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[2048, 768]" = torch.ops.aten.mm.default(view_159, permute_85)
    add_tensor_45: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_46, primals_128);  mm_default_46 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_160: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_45, [4, 512, 768]);  add_tensor_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_59: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_57, view_160);  add_57 = view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_32: "f32[4, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_60: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_24: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_33);  add_59 = getitem_33 = None
    mul_63: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_64: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_129)
    add_61: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_64, primals_130);  mul_64 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_161: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_61, [2048, 768])
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[2048, 768]" = torch.ops.aten.mm.default(view_161, permute_86)
    add_tensor_44: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_45, primals_132);  mm_default_45 = primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_162: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_44, [4, 512, 768]);  add_tensor_44 = None
    mul_65: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_162, 0.125);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[2048, 768]" = torch.ops.aten.mm.default(view_161, permute_87)
    add_tensor_43: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_44, primals_134);  mm_default_44 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_164: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_43, [4, 512, 768]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_165: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_164, [4, -1, 12, 64]);  view_164 = None
    permute_88: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    clone_64: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[2048, 768]" = torch.ops.aten.mm.default(view_161, permute_89)
    add_tensor_42: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_43, primals_136);  mm_default_43 = primals_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_167: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_42, [4, 512, 768]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_168: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_167, [4, -1, 12, 64]);  view_167 = None
    permute_90: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_65: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_169: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_65, [4, 512, 12, 64]);  mul_65 = None
    permute_91: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
    clone_66: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_170: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_66, [48, -1, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_171: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_64, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_172: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_65, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_92: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    bmm_16: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_170, permute_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_173: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [4, 12, 512, 512]);  bmm_16 = None
    add_62: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_173, expand_3);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_174: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_62, [48, 512, 512]);  add_62 = None
    
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
    view_175: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [4, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_93: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_68: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    view_176: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_68, [4, 512, 768]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_177: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_176, [2048, 768]);  view_176 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[2048, 768]" = torch.ops.aten.mm.default(view_177, permute_94)
    add_tensor_41: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_42, primals_138);  mm_default_42 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_178: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_41, [4, 512, 768]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_63: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_61, view_178);  add_61 = view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_34: "f32[4, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_64: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_26: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_35);  add_63 = getitem_35 = None
    mul_66: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_67: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_139)
    add_65: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_140);  mul_67 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_179: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_65, [2048, 768])
    permute_95: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[2048, 768]" = torch.ops.aten.mm.default(view_179, permute_95)
    add_tensor_40: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_41, primals_142);  mm_default_41 = primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_180: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_40, [4, 512, 768]);  add_tensor_40 = None
    mul_68: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_180, 0.125);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_96)
    add_tensor_39: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_40, primals_144);  mm_default_40 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_182: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_39, [4, 512, 768]);  add_tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_183: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_182, [4, -1, 12, 64]);  view_182 = None
    permute_97: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    clone_70: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_98: "f32[768, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_98)
    add_tensor_38: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_39, primals_146);  mm_default_39 = primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_185: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_38, [4, 512, 768]);  add_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_186: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_185, [4, -1, 12, 64]);  view_185 = None
    permute_99: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_71: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_187: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_68, [4, 512, 12, 64]);  mul_68 = None
    permute_100: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    clone_72: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_188: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_72, [48, -1, 64]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_189: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_70, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_190: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_71, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_101: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_18: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_188, permute_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_18, [-1], True)
    sub_27: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_18, amax_9)
    exp_9: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_10: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_19: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_9, view_190)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_191: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [4, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_102: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_74: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_192: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_74, [4, 512, 768]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_193: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_192, [2048, 768]);  view_192 = None
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[2048, 768]" = torch.ops.aten.mm.default(view_193, permute_103)
    add_tensor_37: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_38, primals_148);  mm_default_38 = primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_194: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_37, [4, 512, 768]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_66: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_65, view_194);  add_65 = view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_36: "f32[4, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_67: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_28: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_37);  add_66 = getitem_37 = None
    mul_69: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_70: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_69, primals_149)
    add_68: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_70, primals_150);  mul_70 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_195: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_68, [2048, 768])
    permute_104: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_54: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_152, view_195, permute_104);  primals_152 = None
    view_196: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_54, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.5)
    mul_72: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476);  view_196 = None
    erf_7: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_69: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_73: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_71, add_69);  mul_71 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_197: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_73, [2048, 3072]);  mul_73 = None
    permute_105: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[2048, 768]" = torch.ops.aten.mm.default(view_197, permute_105)
    add_tensor_36: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_37, primals_154);  mm_default_37 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_198: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_36, [4, 512, 768]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_70: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_68, view_198);  add_68 = view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_38: "f32[4, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_71: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_29: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_39);  add_70 = getitem_39 = None
    mul_74: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_75: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_155)
    add_72: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_156);  mul_75 = primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_199: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_72, [2048, 768])
    permute_106: "f32[768, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[2048, 768]" = torch.ops.aten.mm.default(view_199, permute_106)
    add_tensor_35: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_36, primals_158);  mm_default_36 = primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_200: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [4, 512, 768]);  add_tensor_35 = None
    mul_76: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_200, 0.125);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[2048, 768]" = torch.ops.aten.mm.default(view_199, permute_107)
    add_tensor_34: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_35, primals_160);  mm_default_35 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_202: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [4, 512, 768]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_203: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_202, [4, -1, 12, 64]);  view_202 = None
    permute_108: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    clone_78: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_109: "f32[768, 768]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[2048, 768]" = torch.ops.aten.mm.default(view_199, permute_109)
    add_tensor_33: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_34, primals_162);  mm_default_34 = primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_205: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [4, 512, 768]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_206: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_205, [4, -1, 12, 64]);  view_205 = None
    permute_110: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    clone_79: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_76, [4, 512, 12, 64]);  mul_76 = None
    permute_111: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_80: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_208: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_80, [48, -1, 64]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_209: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_78, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_210: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_79, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_112: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_20: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_208, permute_112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_211: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [4, 12, 512, 512]);  bmm_20 = None
    add_73: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_211, expand_3);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_212: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_73, [48, 512, 512]);  add_73 = None
    
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
    view_213: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [4, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_113: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_82: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    view_214: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_82, [4, 512, 768]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_215: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_214, [2048, 768]);  view_214 = None
    permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[2048, 768]" = torch.ops.aten.mm.default(view_215, permute_114)
    add_tensor_32: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_33, primals_164);  mm_default_33 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_216: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [4, 512, 768]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_74: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_72, view_216);  add_72 = view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_40: "f32[4, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_75: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_31: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_41);  add_74 = getitem_41 = None
    mul_77: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_78: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_165)
    add_76: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_166);  mul_78 = primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_217: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_76, [2048, 768])
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[2048, 768]" = torch.ops.aten.mm.default(view_217, permute_115)
    add_tensor_31: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_32, primals_168);  mm_default_32 = primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_218: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_31, [4, 512, 768]);  add_tensor_31 = None
    mul_79: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_218, 0.125);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_116: "f32[768, 768]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_116)
    add_tensor_30: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_31, primals_170);  mm_default_31 = primals_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_220: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [4, 512, 768]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_221: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_220, [4, -1, 12, 64]);  view_220 = None
    permute_117: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None
    clone_84: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_118)
    add_tensor_29: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_30, primals_172);  mm_default_30 = primals_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_223: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [4, 512, 768]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_224: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_223, [4, -1, 12, 64]);  view_223 = None
    permute_119: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    clone_85: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_225: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_79, [4, 512, 12, 64]);  mul_79 = None
    permute_120: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
    clone_86: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_226: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_86, [48, -1, 64]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_227: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_84, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_228: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_85, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_121: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
    bmm_22: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_226, permute_121)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_22, [-1], True)
    sub_32: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_22, amax_11)
    exp_11: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_12: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_23: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_11, view_228)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_229: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [4, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_122: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_88: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_230: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_88, [4, 512, 768]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_231: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_230, [2048, 768]);  view_230 = None
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[2048, 768]" = torch.ops.aten.mm.default(view_231, permute_123)
    add_tensor_28: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_29, primals_174);  mm_default_29 = primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_232: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [4, 512, 768]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_77: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_76, view_232);  add_76 = view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_42: "f32[4, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_78: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_33: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_43);  add_77 = getitem_43 = None
    mul_80: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_81: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_175)
    add_79: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_176);  mul_81 = primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_233: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_79, [2048, 768])
    permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_64: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_178, view_233, permute_124);  primals_178 = None
    view_234: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, 0.5)
    mul_83: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, 0.7071067811865476);  view_234 = None
    erf_8: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_84: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_80);  mul_82 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_235: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_84, [2048, 3072]);  mul_84 = None
    permute_125: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[2048, 768]" = torch.ops.aten.mm.default(view_235, permute_125)
    add_tensor_27: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_28, primals_180);  mm_default_28 = primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_236: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [4, 512, 768]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_81: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_79, view_236);  add_79 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_44: "f32[4, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_82: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_34: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_45);  add_81 = getitem_45 = None
    mul_85: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_86: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_181)
    add_83: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_182);  mul_86 = primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_237: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_83, [2048, 768])
    permute_126: "f32[768, 768]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[2048, 768]" = torch.ops.aten.mm.default(view_237, permute_126)
    add_tensor_26: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_27, primals_184);  mm_default_27 = primals_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_238: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [4, 512, 768]);  add_tensor_26 = None
    mul_87: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_238, 0.125);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_127: "f32[768, 768]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[2048, 768]" = torch.ops.aten.mm.default(view_237, permute_127)
    add_tensor_25: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_26, primals_186);  mm_default_26 = primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_240: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_25, [4, 512, 768]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_241: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_240, [4, -1, 12, 64]);  view_240 = None
    permute_128: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
    clone_92: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[2048, 768]" = torch.ops.aten.mm.default(view_237, permute_129)
    add_tensor_24: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_25, primals_188);  mm_default_25 = primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_243: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [4, 512, 768]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_244: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [4, -1, 12, 64]);  view_243 = None
    permute_130: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    clone_93: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_245: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_87, [4, 512, 12, 64]);  mul_87 = None
    permute_131: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
    clone_94: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_246: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_94, [48, -1, 64]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_247: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_92, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_248: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_93, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_132: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_247, [0, 2, 1]);  view_247 = None
    bmm_24: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_246, permute_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_249: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_24, [4, 12, 512, 512]);  bmm_24 = None
    add_84: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_249, expand_3);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_250: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_84, [48, 512, 512]);  add_84 = None
    
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
    view_251: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_25, [4, 12, 512, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_133: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_96: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_252: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_96, [4, 512, 768]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_253: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_252, [2048, 768]);  view_252 = None
    permute_134: "f32[768, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[2048, 768]" = torch.ops.aten.mm.default(view_253, permute_134)
    add_tensor_23: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_24, primals_190);  mm_default_24 = primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_254: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [4, 512, 768]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_85: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_83, view_254);  add_83 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_46: "f32[4, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_86: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_36: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_47);  add_85 = getitem_47 = None
    mul_88: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_89: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_88, primals_191)
    add_87: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_89, primals_192);  mul_89 = primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_255: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_87, [2048, 768])
    permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[2048, 768]" = torch.ops.aten.mm.default(view_255, permute_135)
    add_tensor_22: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_23, primals_194);  mm_default_23 = primals_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_256: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [4, 512, 768]);  add_tensor_22 = None
    mul_90: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_256, 0.125);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_136)
    add_tensor_21: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_22, primals_196);  mm_default_22 = primals_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_258: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [4, 512, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_259: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_258, [4, -1, 12, 64]);  view_258 = None
    permute_137: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    clone_98: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_138)
    add_tensor_20: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_21, primals_198);  mm_default_21 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_261: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [4, 512, 768]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_262: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_261, [4, -1, 12, 64]);  view_261 = None
    permute_139: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    clone_99: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_263: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_90, [4, 512, 12, 64]);  mul_90 = None
    permute_140: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
    clone_100: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_264: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_100, [48, -1, 64]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_265: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_98, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_266: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_99, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_141: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
    bmm_26: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_264, permute_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_26, [-1], True)
    sub_37: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_26, amax_13)
    exp_13: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_14: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_27: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_13, view_266)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_267: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_27, [4, 12, 512, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_142: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_102: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_268: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_102, [4, 512, 768]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_269: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_268, [2048, 768]);  view_268 = None
    permute_143: "f32[768, 768]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[2048, 768]" = torch.ops.aten.mm.default(view_269, permute_143)
    add_tensor_19: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_20, primals_200);  mm_default_20 = primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_270: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [4, 512, 768]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_88: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_87, view_270);  add_87 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_48: "f32[4, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_89: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_38: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_49);  add_88 = getitem_49 = None
    mul_91: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_24);  sub_38 = None
    mul_92: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, primals_201)
    add_90: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, primals_202);  mul_92 = primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_271: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_90, [2048, 768])
    permute_144: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_74: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_204, view_271, permute_144);  primals_204 = None
    view_272: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_74, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_93: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, 0.5)
    mul_94: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, 0.7071067811865476);  view_272 = None
    erf_9: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_91: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_95: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_93, add_91);  mul_93 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_273: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_95, [2048, 3072]);  mul_95 = None
    permute_145: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[2048, 768]" = torch.ops.aten.mm.default(view_273, permute_145)
    add_tensor_18: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_19, primals_206);  mm_default_19 = primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_274: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [4, 512, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_92: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_90, view_274);  add_90 = view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_50: "f32[4, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_93: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_39: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_51);  add_92 = getitem_51 = None
    mul_96: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = None
    mul_97: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, primals_207)
    add_94: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_208);  mul_97 = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_275: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_94, [2048, 768])
    permute_146: "f32[768, 768]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[2048, 768]" = torch.ops.aten.mm.default(view_275, permute_146)
    add_tensor_17: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_18, primals_210);  mm_default_18 = primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_276: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [4, 512, 768]);  add_tensor_17 = None
    mul_98: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, 0.125);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[2048, 768]" = torch.ops.aten.mm.default(view_275, permute_147)
    add_tensor_16: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_17, primals_212);  mm_default_17 = primals_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_278: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [4, 512, 768]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_279: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_278, [4, -1, 12, 64]);  view_278 = None
    permute_148: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
    clone_106: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_149: "f32[768, 768]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[2048, 768]" = torch.ops.aten.mm.default(view_275, permute_149)
    add_tensor_15: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_16, primals_214);  mm_default_16 = primals_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_281: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [4, 512, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_282: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_281, [4, -1, 12, 64]);  view_281 = None
    permute_150: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    clone_107: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_283: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_98, [4, 512, 12, 64]);  mul_98 = None
    permute_151: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    clone_108: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_284: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_108, [48, -1, 64]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_285: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_106, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_286: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_107, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_152: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
    bmm_28: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_284, permute_152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_287: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_28, [4, 12, 512, 512]);  bmm_28 = None
    add_95: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_287, expand_3);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_288: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_95, [48, 512, 512]);  add_95 = None
    
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
    view_289: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_29, [4, 12, 512, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_153: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_110: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    view_290: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_110, [4, 512, 768]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_291: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_290, [2048, 768]);  view_290 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[2048, 768]" = torch.ops.aten.mm.default(view_291, permute_154)
    add_tensor_14: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_15, primals_216);  mm_default_15 = primals_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_292: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [4, 512, 768]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_96: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_94, view_292);  add_94 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_52: "f32[4, 512, 1]" = var_mean_26[0]
    getitem_53: "f32[4, 512, 1]" = var_mean_26[1];  var_mean_26 = None
    add_97: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_41: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_53);  add_96 = getitem_53 = None
    mul_99: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_26);  sub_41 = None
    mul_100: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, primals_217)
    add_98: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_100, primals_218);  mul_100 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_293: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_98, [2048, 768])
    permute_155: "f32[768, 768]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[2048, 768]" = torch.ops.aten.mm.default(view_293, permute_155)
    add_tensor_13: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_14, primals_220);  mm_default_14 = primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_294: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [4, 512, 768]);  add_tensor_13 = None
    mul_101: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_294, 0.125);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_156: "f32[768, 768]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_156)
    add_tensor_12: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_13, primals_222);  mm_default_13 = primals_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_296: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [4, 512, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_297: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_296, [4, -1, 12, 64]);  view_296 = None
    permute_157: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
    clone_112: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_158)
    add_tensor_11: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_12, primals_224);  mm_default_12 = primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_299: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [4, 512, 768]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_300: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_299, [4, -1, 12, 64]);  view_299 = None
    permute_159: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
    clone_113: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_301: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_101, [4, 512, 12, 64]);  mul_101 = None
    permute_160: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    clone_114: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_302: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_114, [48, -1, 64]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_303: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_112, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_304: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_113, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_161: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    bmm_30: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_302, permute_161)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_30, [-1], True)
    sub_42: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_30, amax_15)
    exp_15: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_16: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_31: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_15, view_304)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_305: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_31, [4, 12, 512, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_162: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_116: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_306: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_116, [4, 512, 768]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_307: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_306, [2048, 768]);  view_306 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[2048, 768]" = torch.ops.aten.mm.default(view_307, permute_163)
    add_tensor_10: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_11, primals_226);  mm_default_11 = primals_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_308: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [4, 512, 768]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_99: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_98, view_308);  add_98 = view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_54: "f32[4, 512, 1]" = var_mean_27[0]
    getitem_55: "f32[4, 512, 1]" = var_mean_27[1];  var_mean_27 = None
    add_100: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_43: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_55);  add_99 = getitem_55 = None
    mul_102: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_27);  sub_43 = None
    mul_103: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_227)
    add_101: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_228);  mul_103 = primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_309: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_101, [2048, 768])
    permute_164: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_84: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_230, view_309, permute_164);  primals_230 = None
    view_310: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_84, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_104: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, 0.5)
    mul_105: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, 0.7071067811865476);  view_310 = None
    erf_10: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_102: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_106: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_104, add_102);  mul_104 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_311: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_106, [2048, 3072]);  mul_106 = None
    permute_165: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_311, permute_165)
    add_tensor_9: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_10, primals_232);  mm_default_10 = primals_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_312: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [4, 512, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_103: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_101, view_312);  add_101 = view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_56: "f32[4, 512, 1]" = var_mean_28[0]
    getitem_57: "f32[4, 512, 1]" = var_mean_28[1];  var_mean_28 = None
    add_104: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_44: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_57);  add_103 = getitem_57 = None
    mul_107: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_28);  sub_44 = None
    mul_108: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, primals_233)
    add_105: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_108, primals_234);  mul_108 = primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_313: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_105, [2048, 768])
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[2048, 768]" = torch.ops.aten.mm.default(view_313, permute_166)
    add_tensor_8: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_9, primals_236);  mm_default_9 = primals_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_314: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [4, 512, 768]);  add_tensor_8 = None
    mul_109: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_314, 0.125);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[2048, 768]" = torch.ops.aten.mm.default(view_313, permute_167)
    add_tensor_7: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_8, primals_238);  mm_default_8 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_316: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [4, 512, 768]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_317: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_316, [4, -1, 12, 64]);  view_316 = None
    permute_168: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    clone_120: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[2048, 768]" = torch.ops.aten.mm.default(view_313, permute_169)
    add_tensor_6: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_7, primals_240);  mm_default_7 = primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_319: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [4, 512, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_320: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_319, [4, -1, 12, 64]);  view_319 = None
    permute_170: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
    clone_121: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_321: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_109, [4, 512, 12, 64]);  mul_109 = None
    permute_171: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
    clone_122: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_322: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_122, [48, -1, 64]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_323: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_120, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_324: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_121, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_172: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    bmm_32: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_322, permute_172)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_325: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_32, [4, 12, 512, 512]);  bmm_32 = None
    add_106: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_325, expand_3);  view_325 = expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_326: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_106, [48, 512, 512]);  add_106 = None
    
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
    view_327: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_33, [4, 12, 512, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_173: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_124: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    view_328: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_124, [4, 512, 768]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_329: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_328, [2048, 768]);  view_328 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_329, permute_174)
    add_tensor_5: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_6, primals_242);  mm_default_6 = primals_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_330: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [4, 512, 768]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_107: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_105, view_330);  add_105 = view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_58: "f32[4, 512, 1]" = var_mean_29[0]
    getitem_59: "f32[4, 512, 1]" = var_mean_29[1];  var_mean_29 = None
    add_108: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_46: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_59);  add_107 = getitem_59 = None
    mul_110: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_29);  sub_46 = None
    mul_111: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, primals_243)
    add_109: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_244);  mul_111 = primals_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_331: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_109, [2048, 768])
    permute_175: "f32[768, 768]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[2048, 768]" = torch.ops.aten.mm.default(view_331, permute_175)
    add_tensor_4: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_5, primals_246);  mm_default_5 = primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_332: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [4, 512, 768]);  add_tensor_4 = None
    mul_112: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_332, 0.125);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_176)
    add_tensor_3: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_4, primals_248);  mm_default_4 = primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_334: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [4, 512, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_335: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_334, [4, -1, 12, 64]);  view_334 = None
    permute_177: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
    clone_126: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[2048, 768]" = torch.ops.aten.mm.default(view_143, permute_178)
    add_tensor_2: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_3, primals_250);  mm_default_3 = primals_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_337: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [4, 512, 768]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_338: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_337, [4, -1, 12, 64]);  view_337 = None
    permute_179: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    clone_127: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_339: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_112, [4, 512, 12, 64]);  mul_112 = None
    permute_180: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    clone_128: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_340: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_128, [48, -1, 64]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_341: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_126, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_342: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_127, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_181: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
    bmm_34: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_340, permute_181)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[48, 512, 1]" = torch.ops.aten.amax.default(bmm_34, [-1], True)
    sub_47: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(bmm_34, amax_17)
    exp_17: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
    sum_18: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_35: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_17, view_342)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_343: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_35, [4, 12, 512, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_182: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_130: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    view_344: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_130, [4, 512, 768]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_345: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_344, [2048, 768]);  view_344 = None
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[2048, 768]" = torch.ops.aten.mm.default(view_345, permute_183)
    add_tensor_1: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_2, primals_252);  mm_default_2 = primals_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_346: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [4, 512, 768]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_110: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_109, view_346);  add_109 = view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_60: "f32[4, 512, 1]" = var_mean_30[0]
    getitem_61: "f32[4, 512, 1]" = var_mean_30[1];  var_mean_30 = None
    add_111: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_48: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_61);  add_110 = getitem_61 = None
    mul_113: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_30);  sub_48 = None
    mul_114: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, primals_253)
    add_112: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, primals_254);  mul_114 = primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_347: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_112, [2048, 768])
    permute_184: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_94: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_256, view_347, permute_184);  primals_256 = None
    view_348: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_94, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_115: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, 0.5)
    mul_116: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, 0.7071067811865476);  view_348 = None
    erf_11: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
    add_113: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_117: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_115, add_113);  mul_115 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_349: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_117, [2048, 3072]);  mul_117 = None
    permute_185: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[2048, 768]" = torch.ops.aten.mm.default(view_349, permute_185)
    add_tensor: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_1, primals_258);  mm_default_1 = primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_350: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor, [4, 512, 768]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_114: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_112, view_350);  add_112 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_62: "f32[4, 512, 1]" = var_mean_31[0]
    getitem_63: "f32[4, 512, 1]" = var_mean_31[1];  var_mean_31 = None
    add_115: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_49: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_63);  add_114 = getitem_63 = None
    mul_118: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_31);  sub_49 = None
    mul_119: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_118, primals_259)
    add_116: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_119, primals_260);  mul_119 = primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1406, code: lm_logits = self.lm_head(outputs[0])
    permute_186: "f32[768, 50265]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    view_351: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_116, [2048, 768]);  add_116 = None
    
    # No stacktrace found for following nodes
    full_default_12: "f32[768, 3]" = torch.ops.aten.full.default([768, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    cat_default: "f32[768, 50268]" = torch.ops.aten.cat.default([permute_186, full_default_12], 1);  full_default_12 = None
    mm_default: "f32[2048, 50268]" = torch.ops.aten.mm.default(view_351, cat_default);  cat_default = None
    slice_tensor_1: "f32[2048, 50265]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1406, code: lm_logits = self.lm_head(outputs[0])
    view_352: "f32[4, 512, 50265]" = torch.ops.aten.reshape.default(slice_tensor_1, [4, 512, 50265]);  slice_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1407, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
    add_117: "f32[4, 512, 50265]" = torch.ops.aten.add.Tensor(view_352, primals_262);  view_352 = primals_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1406, code: lm_logits = self.lm_head(outputs[0])
    permute_189: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    div_18: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 768);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    permute_191: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_195: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_19: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 768);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_204: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_17, [0, 2, 1]);  div_17 = None
    permute_205: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_206: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None
    permute_207: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_181, [0, 2, 1]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_211: "f32[768, 768]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_220: "f32[768, 768]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_20: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 768);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_224: "f32[768, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_229: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_123, [0, 2, 1]);  clone_123 = None
    permute_230: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_19: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_231: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    permute_232: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_172, [0, 2, 1]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_241: "f32[768, 768]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_245: "f32[768, 768]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    div_21: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 768);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    permute_249: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_253: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_22: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 768);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_257: "f32[768, 768]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_262: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_15, [0, 2, 1]);  div_15 = None
    permute_263: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_264: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
    permute_265: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_161, [0, 2, 1]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_278: "f32[768, 768]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_23: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 768);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_287: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_109, [0, 2, 1]);  clone_109 = None
    permute_288: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_21: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_289: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    permute_290: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_152, [0, 2, 1]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_299: "f32[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    div_24: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    permute_307: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_311: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_25: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_320: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_13, [0, 2, 1]);  div_13 = None
    permute_321: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_322: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    permute_323: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_141, [0, 2, 1]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_332: "f32[768, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_26: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_340: "f32[768, 768]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_345: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_95, [0, 2, 1]);  clone_95 = None
    permute_346: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_23: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_347: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_246, [0, 2, 1]);  view_246 = None
    permute_348: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_132, [0, 2, 1]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_352: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_357: "f32[768, 768]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_361: "f32[768, 768]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    div_27: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    permute_365: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_369: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_28: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_373: "f32[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_378: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_11, [0, 2, 1]);  div_11 = None
    permute_379: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_380: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_226, [0, 2, 1]);  view_226 = None
    permute_381: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_121, [0, 2, 1]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_385: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_390: "f32[768, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_394: "f32[768, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_29: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_398: "f32[768, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_403: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_81, [0, 2, 1]);  clone_81 = None
    permute_404: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_25: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_405: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    permute_406: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_112, [0, 2, 1]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_410: "f32[768, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_415: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_419: "f32[768, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    div_30: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    permute_423: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_427: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_31: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_431: "f32[768, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_436: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_9, [0, 2, 1]);  div_9 = None
    permute_437: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_438: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    permute_439: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_101, [0, 2, 1]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_443: "f32[768, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_448: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_452: "f32[768, 768]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_32: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_456: "f32[768, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_461: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_67, [0, 2, 1]);  clone_67 = None
    permute_462: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_27: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_463: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
    permute_464: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_92, [0, 2, 1]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_473: "f32[768, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_477: "f32[768, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    div_33: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    permute_481: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_485: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_34: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_489: "f32[768, 768]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_494: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_7, [0, 2, 1]);  div_7 = None
    permute_495: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_496: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
    permute_497: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_81, [0, 2, 1]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_506: "f32[768, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_510: "f32[768, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_35: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_514: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_519: "f32[48, 512, 512]" = torch.ops.aten.permute.default(clone_53, [0, 2, 1]);  clone_53 = None
    permute_520: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_29: "f32[48, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_521: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    permute_522: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_72, [0, 2, 1]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_526: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_531: "f32[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_535: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1075, code: hidden_states = self.layernorm_embedding(hidden_states)
    div_36: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    div_37: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    permute_539: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_543: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_38: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_547: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_552: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_5, [0, 2, 1]);  div_5 = None
    permute_553: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_554: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    permute_555: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_61, [0, 2, 1]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_559: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_564: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_568: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    div_39: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    permute_572: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_576: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_40: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_580: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_585: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_4, [0, 2, 1]);  div_4 = None
    permute_586: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_587: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
    permute_588: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_50, [0, 2, 1]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_592: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_597: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_601: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    div_41: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    permute_605: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_609: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_42: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_613: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_618: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_3, [0, 2, 1]);  div_3 = None
    permute_619: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_620: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
    permute_621: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_39, [0, 2, 1]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_625: "f32[768, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_630: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_634: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    div_43: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    permute_638: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_642: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_44: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_646: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_651: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_2, [0, 2, 1]);  div_2 = None
    permute_652: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_653: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    permute_654: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_658: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_663: "f32[768, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_667: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    div_45: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    permute_671: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_675: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_46: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_679: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_684: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div_1, [0, 2, 1]);  div_1 = None
    permute_685: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_686: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
    permute_687: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_17, [0, 2, 1]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_691: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_696: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_700: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    div_47: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    permute_704: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_708: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_48: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    permute_712: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_717: "f32[48, 512, 512]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
    permute_718: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_719: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    permute_720: "f32[48, 512, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_724: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_729: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_733: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:824, code: hidden_states = self.layernorm_embedding(hidden_states)
    div_49: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [add_117, clone_50, clone_51, clone_56, clone_57, clone_64, clone_65, clone_70, clone_71, clone_78, clone_79, clone_84, clone_85, clone_92, clone_93, clone_98, clone_99, clone_106, clone_107, clone_112, clone_113, clone_120, clone_121, clone_126, clone_127, add_45, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_103, primals_113, primals_123, primals_129, primals_139, primals_149, primals_155, primals_165, primals_175, primals_181, primals_191, primals_201, primals_207, primals_217, primals_227, primals_233, primals_243, primals_253, primals_259, primals_264, view, add, mul_1, view_1, bmm, amax, sum_1, view_15, mul_4, view_17, addmm_4, view_19, mul_9, view_21, bmm_2, amax_1, sum_2, view_35, mul_12, view_37, addmm_10, view_39, mul_17, view_41, bmm_4, amax_2, sum_3, view_55, mul_20, view_57, addmm_16, view_59, mul_25, view_61, bmm_6, amax_3, sum_4, view_75, mul_28, view_77, addmm_22, view_79, mul_33, view_81, bmm_8, amax_4, sum_5, view_95, mul_36, view_97, addmm_28, view_99, mul_41, view_101, bmm_10, amax_5, sum_6, view_115, mul_44, view_117, addmm_34, view_119, mul_49, mul_52, view_123, view_139, mul_55, view_141, view_143, bmm_14, amax_7, sum_8, view_155, mul_58, view_157, addmm_44, view_159, mul_63, view_161, view_177, mul_66, view_179, bmm_18, amax_9, sum_10, view_193, mul_69, view_195, addmm_54, view_197, mul_74, view_199, view_215, mul_77, view_217, bmm_22, amax_11, sum_12, view_231, mul_80, view_233, addmm_64, view_235, mul_85, view_237, view_253, mul_88, view_255, bmm_26, amax_13, sum_14, view_269, mul_91, view_271, addmm_74, view_273, mul_96, view_275, view_291, mul_99, view_293, bmm_30, amax_15, sum_16, view_307, mul_102, view_309, addmm_84, view_311, mul_107, view_313, view_329, mul_110, view_331, bmm_34, amax_17, sum_18, view_345, mul_113, view_347, addmm_94, view_349, mul_118, view_351, permute_189, div_18, permute_191, permute_195, div_19, permute_199, permute_204, permute_205, permute_206, permute_207, permute_211, permute_216, permute_220, div_20, permute_224, permute_229, permute_230, alias_19, permute_231, permute_232, permute_236, permute_241, permute_245, div_21, permute_249, permute_253, div_22, permute_257, permute_262, permute_263, permute_264, permute_265, permute_269, permute_274, permute_278, div_23, permute_282, permute_287, permute_288, alias_21, permute_289, permute_290, permute_294, permute_299, permute_303, div_24, permute_307, permute_311, div_25, permute_315, permute_320, permute_321, permute_322, permute_323, permute_327, permute_332, permute_336, div_26, permute_340, permute_345, permute_346, alias_23, permute_347, permute_348, permute_352, permute_357, permute_361, div_27, permute_365, permute_369, div_28, permute_373, permute_378, permute_379, permute_380, permute_381, permute_385, permute_390, permute_394, div_29, permute_398, permute_403, permute_404, alias_25, permute_405, permute_406, permute_410, permute_415, permute_419, div_30, permute_423, permute_427, div_31, permute_431, permute_436, permute_437, permute_438, permute_439, permute_443, permute_448, permute_452, div_32, permute_456, permute_461, permute_462, alias_27, permute_463, permute_464, permute_468, permute_473, permute_477, div_33, permute_481, permute_485, div_34, permute_489, permute_494, permute_495, permute_496, permute_497, permute_501, permute_506, permute_510, div_35, permute_514, permute_519, permute_520, alias_29, permute_521, permute_522, permute_526, permute_531, permute_535, div_36, div_37, permute_539, permute_543, div_38, permute_547, permute_552, permute_553, permute_554, permute_555, permute_559, permute_564, permute_568, div_39, permute_572, permute_576, div_40, permute_580, permute_585, permute_586, permute_587, permute_588, permute_592, permute_597, permute_601, div_41, permute_605, permute_609, div_42, permute_613, permute_618, permute_619, permute_620, permute_621, permute_625, permute_630, permute_634, div_43, permute_638, permute_642, div_44, permute_646, permute_651, permute_652, permute_653, permute_654, permute_658, permute_663, permute_667, div_45, permute_671, permute_675, div_46, permute_679, permute_684, permute_685, permute_686, permute_687, permute_691, permute_696, permute_700, div_47, permute_704, permute_708, div_48, permute_712, permute_717, permute_718, permute_719, permute_720, permute_724, permute_729, permute_733, div_49]
    