from __future__ import annotations



def forward(self, primals_1: "f32[2304]", primals_2: "f32[768, 2304]", primals_3: "f32[768]", primals_4: "f32[768, 768]", primals_5: "f32[3072]", primals_6: "f32[768, 3072]", primals_7: "f32[768]", primals_8: "f32[3072, 768]", primals_9: "f32[2304]", primals_10: "f32[768, 2304]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[3072]", primals_14: "f32[768, 3072]", primals_15: "f32[768]", primals_16: "f32[3072, 768]", primals_17: "f32[2304]", primals_18: "f32[768, 2304]", primals_19: "f32[768]", primals_20: "f32[768, 768]", primals_21: "f32[3072]", primals_22: "f32[768, 3072]", primals_23: "f32[768]", primals_24: "f32[3072, 768]", primals_25: "f32[2304]", primals_26: "f32[768, 2304]", primals_27: "f32[768]", primals_28: "f32[768, 768]", primals_29: "f32[3072]", primals_30: "f32[768, 3072]", primals_31: "f32[768]", primals_32: "f32[3072, 768]", primals_33: "f32[2304]", primals_34: "f32[768, 2304]", primals_35: "f32[768]", primals_36: "f32[768, 768]", primals_37: "f32[3072]", primals_38: "f32[768, 3072]", primals_39: "f32[768]", primals_40: "f32[3072, 768]", primals_41: "f32[2304]", primals_42: "f32[768, 2304]", primals_43: "f32[768]", primals_44: "f32[768, 768]", primals_45: "f32[3072]", primals_46: "f32[768, 3072]", primals_47: "f32[768]", primals_48: "f32[3072, 768]", primals_49: "f32[2304]", primals_50: "f32[768, 2304]", primals_51: "f32[768]", primals_52: "f32[768, 768]", primals_53: "f32[3072]", primals_54: "f32[768, 3072]", primals_55: "f32[768]", primals_56: "f32[3072, 768]", primals_57: "f32[2304]", primals_58: "f32[768, 2304]", primals_59: "f32[768]", primals_60: "f32[768, 768]", primals_61: "f32[3072]", primals_62: "f32[768, 3072]", primals_63: "f32[768]", primals_64: "f32[3072, 768]", primals_65: "f32[2304]", primals_66: "f32[768, 2304]", primals_67: "f32[768]", primals_68: "f32[768, 768]", primals_69: "f32[3072]", primals_70: "f32[768, 3072]", primals_71: "f32[768]", primals_72: "f32[3072, 768]", primals_73: "f32[2304]", primals_74: "f32[768, 2304]", primals_75: "f32[768]", primals_76: "f32[768, 768]", primals_77: "f32[3072]", primals_78: "f32[768, 3072]", primals_79: "f32[768]", primals_80: "f32[3072, 768]", primals_81: "f32[2304]", primals_82: "f32[768, 2304]", primals_83: "f32[768]", primals_84: "f32[768, 768]", primals_85: "f32[3072]", primals_86: "f32[768, 3072]", primals_87: "f32[768]", primals_88: "f32[3072, 768]", primals_89: "f32[2304]", primals_90: "f32[768, 2304]", primals_91: "f32[768]", primals_92: "f32[768, 768]", primals_93: "f32[3072]", primals_94: "f32[768, 3072]", primals_95: "f32[768]", primals_96: "f32[3072, 768]", primals_97: "f32[50257, 768]", primals_98: "f32[1024, 768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[768]", primals_108: "f32[768]", primals_109: "f32[768]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[768]", primals_117: "f32[768]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768]", primals_125: "f32[768]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[768]", primals_129: "f32[768]", primals_130: "f32[768]", primals_131: "f32[768]", primals_132: "f32[768]", primals_133: "f32[768]", primals_134: "f32[768]", primals_135: "f32[768]", primals_136: "f32[768]", primals_137: "f32[768]", primals_138: "f32[768]", primals_139: "f32[768]", primals_140: "f32[768]", primals_141: "f32[768]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[768]", primals_145: "f32[768]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[2, 768]", primals_150: "b8[1, 1, 1024, 1024]", primals_151: "b8[1, 1, 1024, 1024]", primals_152: "b8[1, 1, 1024, 1024]", primals_153: "b8[1, 1, 1024, 1024]", primals_154: "b8[1, 1, 1024, 1024]", primals_155: "b8[1, 1, 1024, 1024]", primals_156: "b8[1, 1, 1024, 1024]", primals_157: "b8[1, 1, 1024, 1024]", primals_158: "b8[1, 1, 1024, 1024]", primals_159: "b8[1, 1, 1024, 1024]", primals_160: "b8[1, 1, 1024, 1024]", primals_161: "b8[1, 1, 1024, 1024]", primals_162: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:781, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 1024]" = torch.ops.aten.reshape.default(primals_162, [-1, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:802, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:803, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 1024]" = torch.ops.aten.reshape.default(unsqueeze, [-1, 1024]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_97, view);  primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_98, view_1);  primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:845, code: hidden_states = inputs_embeds + position_embeds
    add: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:851, code: hidden_states = self.drop(hidden_states)
    native_dropout = torch.ops.aten.native_dropout.default(add, 0.1, True);  add = None
    getitem: "f32[1, 1024, 768]" = native_dropout[0]
    getitem_1: "b8[1, 1024, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(getitem, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 1024, 1]" = var_mean[0]
    getitem_3: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(getitem, getitem_3);  getitem_3 = None
    mul: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, primals_99)
    add_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_100);  mul_1 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_2: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_2, [-1, 768]);  add_2 = None
    addmm: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_1, view_2, primals_2);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_3: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm, [1, 1024, 2304]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [768, 768, 768], 2);  view_3 = None
    getitem_4: "f32[1, 1024, 768]" = split_with_sizes[0]
    getitem_5: "f32[1, 1024, 768]" = split_with_sizes[1]
    getitem_6: "f32[1, 1024, 768]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [1, 1024, 12, 64]);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_5: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_5, [1, 1024, 12, 64]);  getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_6: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_6, [1, 1024, 12, 64]);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_3: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
    expand: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute, [1, 12, 1024, 64]);  permute = None
    view_7: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand, [12, 1024, 64]);  expand = None
    expand_1: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_3, [1, 12, 64, 1024]);  permute_3 = None
    view_8: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_1, [12, 64, 1024]);  expand_1 = None
    bmm: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_7, view_8)
    view_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm, [1, 12, 1024, 1024]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_9, full_default);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_150, 0, 0, 9223372036854775807)
    slice_2: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_2, div, full_default_1);  slice_2 = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_1, 0.1, True);  div_1 = None
    getitem_7: "f32[1, 12, 1024, 1024]" = native_dropout_1[0]
    getitem_8: "b8[1, 12, 1024, 1024]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_7, [1, 12, 1024, 1024]);  getitem_7 = None
    view_10: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_2, [12, 1024, 1024]);  expand_2 = None
    expand_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_2, [1, 12, 1024, 64])
    view_11: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_3, [12, 1024, 64]);  expand_3 = None
    bmm_1: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_10, view_11)
    view_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    clone: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_13: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone, [1, 1024, 768]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_14: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_13, [-1, 768]);  view_13 = None
    addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_3, view_14, primals_4);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_15: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 1024, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_15, 0.1, True);  view_15 = None
    getitem_9: "f32[1, 1024, 768]" = native_dropout_2[0]
    getitem_10: "b8[1, 1024, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_3: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_9, getitem);  getitem_9 = getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_11: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_12: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_12);  getitem_12 = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_101)
    add_5: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_102);  mul_3 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_16: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_5, [-1, 768]);  add_5 = None
    addmm_2: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_5, view_16, primals_6);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_17: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_2, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    pow_1: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
    mul_5: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_17, mul_5);  view_17 = mul_5 = None
    mul_6: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    add_7: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    mul_7: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_18: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_7, [-1, 3072]);  mul_7 = None
    addmm_3: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_7, view_18, primals_8);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_19: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 1024, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_19, 0.1, True);  view_19 = None
    getitem_13: "f32[1, 1024, 768]" = native_dropout_3[0]
    getitem_14: "b8[1, 1024, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_8: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_3, getitem_13);  add_3 = getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_15: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_16: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_15, 1e-05);  getitem_15 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_16);  getitem_16 = None
    mul_8: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_103)
    add_10: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_104);  mul_9 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_20: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_10, [-1, 768]);  add_10 = None
    addmm_4: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_9, view_20, primals_10);  primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_21: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_4, [1, 1024, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_21, [768, 768, 768], 2);  view_21 = None
    getitem_17: "f32[1, 1024, 768]" = split_with_sizes_1[0]
    getitem_18: "f32[1, 1024, 768]" = split_with_sizes_1[1]
    getitem_19: "f32[1, 1024, 768]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_22: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_17, [1, 1024, 12, 64]);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_23: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_18, [1, 1024, 12, 64]);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_6: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_24: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_19, [1, 1024, 12, 64]);  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_8: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2])
    expand_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 1024, 64]);  permute_5 = None
    view_25: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_4, [12, 1024, 64]);  expand_4 = None
    expand_5: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_8, [1, 12, 64, 1024]);  permute_8 = None
    view_26: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_5, [12, 64, 1024]);  expand_5 = None
    bmm_2: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_25, view_26)
    view_27: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 1024, 1024]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_27, full_default);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_3: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_151, 0, 0, 9223372036854775807)
    slice_4: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 9223372036854775807);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_4, div_2, full_default_1);  slice_4 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_4 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_20: "f32[1, 12, 1024, 1024]" = native_dropout_4[0]
    getitem_21: "b8[1, 12, 1024, 1024]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_20, [1, 12, 1024, 1024]);  getitem_20 = None
    view_28: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_6, [12, 1024, 1024]);  expand_6 = None
    expand_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_7, [1, 12, 1024, 64])
    view_29: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_7, [12, 1024, 64]);  expand_7 = None
    bmm_3: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_28, view_29)
    view_30: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_9: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_1: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_31: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_1, [1, 1024, 768]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_32: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_31, [-1, 768]);  view_31 = None
    addmm_5: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_11, view_32, primals_12);  primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_33: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 1024, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_33, 0.1, True);  view_33 = None
    getitem_22: "f32[1, 1024, 768]" = native_dropout_5[0]
    getitem_23: "b8[1, 1024, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_11: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_22, add_8);  getitem_22 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_25: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_25);  getitem_25 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_105)
    add_13: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_106);  mul_11 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_34: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
    addmm_6: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_13, view_34, primals_14);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_35: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_6, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_2: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_13: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_13);  view_35 = mul_13 = None
    mul_14: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    add_15: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    mul_15: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_36: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_15, [-1, 3072]);  mul_15 = None
    addmm_7: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_15, view_36, primals_16);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_37: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 1024, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_37, 0.1, True);  view_37 = None
    getitem_26: "f32[1, 1024, 768]" = native_dropout_6[0]
    getitem_27: "b8[1, 1024, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_16: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_11, getitem_26);  add_11 = getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_29: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    add_17: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_6: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_29);  getitem_29 = None
    mul_16: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_17: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_107)
    add_18: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_108);  mul_17 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_38: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_18, [-1, 768]);  add_18 = None
    addmm_8: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_17, view_38, primals_18);  primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_39: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_8, [1, 1024, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_39, [768, 768, 768], 2);  view_39 = None
    getitem_30: "f32[1, 1024, 768]" = split_with_sizes_2[0]
    getitem_31: "f32[1, 1024, 768]" = split_with_sizes_2[1]
    getitem_32: "f32[1, 1024, 768]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_40: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_30, [1, 1024, 12, 64]);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_10: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_41: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_31, [1, 1024, 12, 64]);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_42: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_32, [1, 1024, 12, 64]);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_13: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_11, [0, 1, 3, 2])
    expand_8: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_10, [1, 12, 1024, 64]);  permute_10 = None
    view_43: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_8, [12, 1024, 64]);  expand_8 = None
    expand_9: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_13, [1, 12, 64, 1024]);  permute_13 = None
    view_44: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_9, [12, 64, 1024]);  expand_9 = None
    bmm_4: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_43, view_44)
    view_45: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 1024, 1024]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_45, full_default);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_5: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_152, 0, 0, 9223372036854775807)
    slice_6: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_6, div_4, full_default_1);  slice_6 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_7 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_33: "f32[1, 12, 1024, 1024]" = native_dropout_7[0]
    getitem_34: "b8[1, 12, 1024, 1024]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_33, [1, 12, 1024, 1024]);  getitem_33 = None
    view_46: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_10, [12, 1024, 1024]);  expand_10 = None
    expand_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_12, [1, 12, 1024, 64])
    view_47: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_11, [12, 1024, 64]);  expand_11 = None
    bmm_5: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_46, view_47)
    view_48: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_14: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_2: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_49: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_2, [1, 1024, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_50: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_49, [-1, 768]);  view_49 = None
    addmm_9: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_19, view_50, primals_20);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_51: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 1024, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_51, 0.1, True);  view_51 = None
    getitem_35: "f32[1, 1024, 768]" = native_dropout_8[0]
    getitem_36: "b8[1, 1024, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_19: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_35, add_16);  getitem_35 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_37: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_38: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_38);  getitem_38 = None
    mul_18: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_19: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_109)
    add_21: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_110);  mul_19 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_52: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_21, [-1, 768]);  add_21 = None
    addmm_10: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_21, view_52, primals_22);  primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_53: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
    pow_3: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
    mul_21: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_53, mul_21);  view_53 = mul_21 = None
    mul_22: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    add_23: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    mul_23: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_54: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
    addmm_11: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_23, view_54, primals_24);  primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_55: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 1024, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_55, 0.1, True);  view_55 = None
    getitem_39: "f32[1, 1024, 768]" = native_dropout_9[0]
    getitem_40: "b8[1, 1024, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_24: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_19, getitem_39);  add_19 = getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_41: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_42: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    add_25: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_9: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_42);  getitem_42 = None
    mul_24: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_25: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_111)
    add_26: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_112);  mul_25 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_56: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_26, [-1, 768]);  add_26 = None
    addmm_12: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_25, view_56, primals_26);  primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_57: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_12, [1, 1024, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_57, [768, 768, 768], 2);  view_57 = None
    getitem_43: "f32[1, 1024, 768]" = split_with_sizes_3[0]
    getitem_44: "f32[1, 1024, 768]" = split_with_sizes_3[1]
    getitem_45: "f32[1, 1024, 768]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_58: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_43, [1, 1024, 12, 64]);  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_15: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_59: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_44, [1, 1024, 12, 64]);  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_60: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_45, [1, 1024, 12, 64]);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_17: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_18: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2])
    expand_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_15, [1, 12, 1024, 64]);  permute_15 = None
    view_61: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_12, [12, 1024, 64]);  expand_12 = None
    expand_13: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [1, 12, 64, 1024]);  permute_18 = None
    view_62: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_13, [12, 64, 1024]);  expand_13 = None
    bmm_6: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_61, view_62)
    view_63: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 1024, 1024]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_63, full_default);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_7: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_153, 0, 0, 9223372036854775807)
    slice_8: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_8, div_6, full_default_1);  slice_8 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_10 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_46: "f32[1, 12, 1024, 1024]" = native_dropout_10[0]
    getitem_47: "b8[1, 12, 1024, 1024]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_46, [1, 12, 1024, 1024]);  getitem_46 = None
    view_64: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_14, [12, 1024, 1024]);  expand_14 = None
    expand_15: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [1, 12, 1024, 64])
    view_65: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_15, [12, 1024, 64]);  expand_15 = None
    bmm_7: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_64, view_65)
    view_66: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    clone_3: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_67: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_3, [1, 1024, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_68: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_67, [-1, 768]);  view_67 = None
    addmm_13: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_27, view_68, primals_28);  primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_69: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 1024, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_69, 0.1, True);  view_69 = None
    getitem_48: "f32[1, 1024, 768]" = native_dropout_11[0]
    getitem_49: "b8[1, 1024, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_27: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_48, add_24);  getitem_48 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_51: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_51);  getitem_51 = None
    mul_26: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_27: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_113)
    add_29: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_114);  mul_27 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_70: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_29, [-1, 768]);  add_29 = None
    addmm_14: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_29, view_70, primals_30);  primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_71: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_14, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    pow_4: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
    mul_29: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_71, mul_29);  view_71 = mul_29 = None
    mul_30: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    add_31: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    mul_31: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_72: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_31, [-1, 3072]);  mul_31 = None
    addmm_15: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_31, view_72, primals_32);  primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_73: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 1024, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_73, 0.1, True);  view_73 = None
    getitem_52: "f32[1, 1024, 768]" = native_dropout_12[0]
    getitem_53: "b8[1, 1024, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_32: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_27, getitem_52);  add_27 = getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_55: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    add_33: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_12: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_55);  getitem_55 = None
    mul_32: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_33: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_115)
    add_34: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_116);  mul_33 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_74: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_34, [-1, 768]);  add_34 = None
    addmm_16: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_33, view_74, primals_34);  primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_75: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_16, [1, 1024, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_75, [768, 768, 768], 2);  view_75 = None
    getitem_56: "f32[1, 1024, 768]" = split_with_sizes_4[0]
    getitem_57: "f32[1, 1024, 768]" = split_with_sizes_4[1]
    getitem_58: "f32[1, 1024, 768]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_76: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_56, [1, 1024, 12, 64]);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_20: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_77: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_57, [1, 1024, 12, 64]);  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_21: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_78: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, [1, 1024, 12, 64]);  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_22: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_23: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_21, [0, 1, 3, 2])
    expand_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_20, [1, 12, 1024, 64]);  permute_20 = None
    view_79: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_16, [12, 1024, 64]);  expand_16 = None
    expand_17: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_23, [1, 12, 64, 1024]);  permute_23 = None
    view_80: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_17, [12, 64, 1024]);  expand_17 = None
    bmm_8: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_79, view_80)
    view_81: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 1024, 1024]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_81, full_default);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_9: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_154, 0, 0, 9223372036854775807)
    slice_10: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_10, div_8, full_default_1);  slice_10 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_59: "f32[1, 12, 1024, 1024]" = native_dropout_13[0]
    getitem_60: "b8[1, 12, 1024, 1024]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_59, [1, 12, 1024, 1024]);  getitem_59 = None
    view_82: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_18, [12, 1024, 1024]);  expand_18 = None
    expand_19: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_22, [1, 12, 1024, 64])
    view_83: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_19, [12, 1024, 64]);  expand_19 = None
    bmm_9: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_24: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    clone_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_85: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_4, [1, 1024, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_86: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_85, [-1, 768]);  view_85 = None
    addmm_17: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_35, view_86, primals_36);  primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_87: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 1024, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_87, 0.1, True);  view_87 = None
    getitem_61: "f32[1, 1024, 768]" = native_dropout_14[0]
    getitem_62: "b8[1, 1024, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_35: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_61, add_32);  getitem_61 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_63: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_64: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_64);  getitem_64 = None
    mul_34: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_35: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_117)
    add_37: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_118);  mul_35 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_88: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_37, [-1, 768]);  add_37 = None
    addmm_18: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_37, view_88, primals_38);  primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_89: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_18, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    pow_5: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
    mul_37: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_37);  view_89 = mul_37 = None
    mul_38: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_39: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    mul_39: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_90: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_39, [-1, 3072]);  mul_39 = None
    addmm_19: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_39, view_90, primals_40);  primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_91: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 1024, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_91, 0.1, True);  view_91 = None
    getitem_65: "f32[1, 1024, 768]" = native_dropout_15[0]
    getitem_66: "b8[1, 1024, 768]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_40: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_35, getitem_65);  add_35 = getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_67: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_68: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    add_41: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_15: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_68);  getitem_68 = None
    mul_40: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_41: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_119)
    add_42: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_120);  mul_41 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_92: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_42, [-1, 768]);  add_42 = None
    addmm_20: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_41, view_92, primals_42);  primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_93: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_20, [1, 1024, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_93, [768, 768, 768], 2);  view_93 = None
    getitem_69: "f32[1, 1024, 768]" = split_with_sizes_5[0]
    getitem_70: "f32[1, 1024, 768]" = split_with_sizes_5[1]
    getitem_71: "f32[1, 1024, 768]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_94: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_69, [1, 1024, 12, 64]);  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_95: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_70, [1, 1024, 12, 64]);  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_26: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_96: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_71, [1, 1024, 12, 64]);  getitem_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_20: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_25, [1, 12, 1024, 64]);  permute_25 = None
    view_97: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_20, [12, 1024, 64]);  expand_20 = None
    expand_21: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 1024]);  permute_28 = None
    view_98: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_21, [12, 64, 1024]);  expand_21 = None
    bmm_10: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 1024, 1024]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_99, full_default);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_11: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_155, 0, 0, 9223372036854775807)
    slice_12: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_11, 1, 0, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_12, div_10, full_default_1);  slice_12 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_16 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_72: "f32[1, 12, 1024, 1024]" = native_dropout_16[0]
    getitem_73: "b8[1, 12, 1024, 1024]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_72, [1, 12, 1024, 1024]);  getitem_72 = None
    view_100: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_22, [12, 1024, 1024]);  expand_22 = None
    expand_23: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 1024, 64])
    view_101: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_23, [12, 1024, 64]);  expand_23 = None
    bmm_11: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
    view_102: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_5: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_103: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_5, [1, 1024, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_104: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_103, [-1, 768]);  view_103 = None
    addmm_21: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_43, view_104, primals_44);  primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_105: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 1024, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_105, 0.1, True);  view_105 = None
    getitem_74: "f32[1, 1024, 768]" = native_dropout_17[0]
    getitem_75: "b8[1, 1024, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_43: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_74, add_40);  getitem_74 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_77: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_77);  getitem_77 = None
    mul_42: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_43: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_121)
    add_45: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_122);  mul_43 = primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_106: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [-1, 768]);  add_45 = None
    addmm_22: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_45, view_106, primals_46);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_107: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    pow_6: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
    mul_45: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_107, mul_45);  view_107 = mul_45 = None
    mul_46: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    add_47: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    mul_47: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_108: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
    addmm_23: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_47, view_108, primals_48);  primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_109: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 1024, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_78: "f32[1, 1024, 768]" = native_dropout_18[0]
    getitem_79: "b8[1, 1024, 768]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_48: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_43, getitem_78);  add_43 = getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_81: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    add_49: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_18: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_81);  getitem_81 = None
    mul_48: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_49: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_123)
    add_50: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_49, primals_124);  mul_49 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_110: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_50, [-1, 768]);  add_50 = None
    addmm_24: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_49, view_110, primals_50);  primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_111: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_24, [1, 1024, 2304]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_111, [768, 768, 768], 2);  view_111 = None
    getitem_82: "f32[1, 1024, 768]" = split_with_sizes_6[0]
    getitem_83: "f32[1, 1024, 768]" = split_with_sizes_6[1]
    getitem_84: "f32[1, 1024, 768]" = split_with_sizes_6[2];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_112: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_82, [1, 1024, 12, 64]);  getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_30: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_113: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_83, [1, 1024, 12, 64]);  getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_31: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_114: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_84, [1, 1024, 12, 64]);  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_32: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_33: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_31, [0, 1, 3, 2])
    expand_24: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_30, [1, 12, 1024, 64]);  permute_30 = None
    view_115: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_24, [12, 1024, 64]);  expand_24 = None
    expand_25: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_33, [1, 12, 64, 1024]);  permute_33 = None
    view_116: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_25, [12, 64, 1024]);  expand_25 = None
    bmm_12: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_115, view_116)
    view_117: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_12, [1, 12, 1024, 1024]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_12: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_117, full_default);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_13: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_156, 0, 0, 9223372036854775807)
    slice_14: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_14, div_12, full_default_1);  slice_14 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_19: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
    exp_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_12: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_19 = torch.ops.aten.native_dropout.default(div_13, 0.1, True);  div_13 = None
    getitem_85: "f32[1, 12, 1024, 1024]" = native_dropout_19[0]
    getitem_86: "b8[1, 12, 1024, 1024]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_26: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_85, [1, 12, 1024, 1024]);  getitem_85 = None
    view_118: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_26, [12, 1024, 1024]);  expand_26 = None
    expand_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_32, [1, 12, 1024, 64])
    view_119: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_27, [12, 1024, 64]);  expand_27 = None
    bmm_13: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_118, view_119)
    view_120: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 12, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_34: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    clone_6: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_121: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_6, [1, 1024, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_122: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_121, [-1, 768]);  view_121 = None
    addmm_25: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_51, view_122, primals_52);  primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_123: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 1024, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_123, 0.1, True);  view_123 = None
    getitem_87: "f32[1, 1024, 768]" = native_dropout_20[0]
    getitem_88: "b8[1, 1024, 768]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_51: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_87, add_48);  getitem_87 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_89: "f32[1, 1024, 1]" = var_mean_13[0]
    getitem_90: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    add_52: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_90);  getitem_90 = None
    mul_50: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    mul_51: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_125)
    add_53: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_126);  mul_51 = primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_124: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_53, [-1, 768]);  add_53 = None
    addmm_26: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_53, view_124, primals_54);  primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_125: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_26, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
    pow_7: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_125, 3.0)
    mul_53: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_54: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_125, mul_53);  view_125 = mul_53 = None
    mul_54: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
    tanh_6: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    add_55: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    mul_55: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_126: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_55, [-1, 3072]);  mul_55 = None
    addmm_27: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_55, view_126, primals_56);  primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_127: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 1024, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_127, 0.1, True);  view_127 = None
    getitem_91: "f32[1, 1024, 768]" = native_dropout_21[0]
    getitem_92: "b8[1, 1024, 768]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_56: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_51, getitem_91);  add_51 = getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_93: "f32[1, 1024, 1]" = var_mean_14[0]
    getitem_94: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    add_57: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_93, 1e-05);  getitem_93 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_21: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_94);  getitem_94 = None
    mul_56: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_57: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_127)
    add_58: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_128);  mul_57 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_128: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_58, [-1, 768]);  add_58 = None
    addmm_28: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_57, view_128, primals_58);  primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_129: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_28, [1, 1024, 2304]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_129, [768, 768, 768], 2);  view_129 = None
    getitem_95: "f32[1, 1024, 768]" = split_with_sizes_7[0]
    getitem_96: "f32[1, 1024, 768]" = split_with_sizes_7[1]
    getitem_97: "f32[1, 1024, 768]" = split_with_sizes_7[2];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_130: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_95, [1, 1024, 12, 64]);  getitem_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_131: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_96, [1, 1024, 12, 64]);  getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_36: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_132: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_97, [1, 1024, 12, 64]);  getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_37: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_38: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2])
    expand_28: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [1, 12, 1024, 64]);  permute_35 = None
    view_133: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_28, [12, 1024, 64]);  expand_28 = None
    expand_29: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_38, [1, 12, 64, 1024]);  permute_38 = None
    view_134: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_29, [12, 64, 1024]);  expand_29 = None
    bmm_14: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_133, view_134)
    view_135: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_14, [1, 12, 1024, 1024]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_135, full_default);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_15: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_157, 0, 0, 9223372036854775807)
    slice_16: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_15, 1, 0, 9223372036854775807);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_16, div_14, full_default_1);  slice_16 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
    sub_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
    exp_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_22 = torch.ops.aten.native_dropout.default(div_15, 0.1, True);  div_15 = None
    getitem_98: "f32[1, 12, 1024, 1024]" = native_dropout_22[0]
    getitem_99: "b8[1, 12, 1024, 1024]" = native_dropout_22[1];  native_dropout_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_30: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_98, [1, 12, 1024, 1024]);  getitem_98 = None
    view_136: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_30, [12, 1024, 1024]);  expand_30 = None
    expand_31: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_37, [1, 12, 1024, 64])
    view_137: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_31, [12, 1024, 64]);  expand_31 = None
    bmm_15: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_136, view_137)
    view_138: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 12, 1024, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_39: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_139: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 1024, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_140: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_139, [-1, 768]);  view_139 = None
    addmm_29: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_59, view_140, primals_60);  primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_141: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 1024, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_141, 0.1, True);  view_141 = None
    getitem_100: "f32[1, 1024, 768]" = native_dropout_23[0]
    getitem_101: "b8[1, 1024, 768]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_59: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_100, add_56);  getitem_100 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1024, 1]" = var_mean_15[0]
    getitem_103: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_103);  getitem_103 = None
    mul_58: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_59: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_129)
    add_61: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_130);  mul_59 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_142: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_61, [-1, 768]);  add_61 = None
    addmm_30: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_61, view_142, primals_62);  primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_143: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_30, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
    pow_8: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_143, 3.0)
    mul_61: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_62: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_143, mul_61);  view_143 = mul_61 = None
    mul_62: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_7: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    add_63: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    mul_63: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_144: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_63, [-1, 3072]);  mul_63 = None
    addmm_31: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_63, view_144, primals_64);  primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_145: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_31, [1, 1024, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_145, 0.1, True);  view_145 = None
    getitem_104: "f32[1, 1024, 768]" = native_dropout_24[0]
    getitem_105: "b8[1, 1024, 768]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_64: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_59, getitem_104);  add_59 = getitem_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 1024, 1]" = var_mean_16[0]
    getitem_107: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    add_65: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_24: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_107);  getitem_107 = None
    mul_64: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_65: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_131)
    add_66: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_65, primals_132);  mul_65 = primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_146: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_66, [-1, 768]);  add_66 = None
    addmm_32: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_65, view_146, primals_66);  primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_147: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_32, [1, 1024, 2304]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_147, [768, 768, 768], 2);  view_147 = None
    getitem_108: "f32[1, 1024, 768]" = split_with_sizes_8[0]
    getitem_109: "f32[1, 1024, 768]" = split_with_sizes_8[1]
    getitem_110: "f32[1, 1024, 768]" = split_with_sizes_8[2];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_148: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_108, [1, 1024, 12, 64]);  getitem_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_40: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_149: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_109, [1, 1024, 12, 64]);  getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_41: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_150: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_110, [1, 1024, 12, 64]);  getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_42: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_43: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_41, [0, 1, 3, 2])
    expand_32: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_40, [1, 12, 1024, 64]);  permute_40 = None
    view_151: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_32, [12, 1024, 64]);  expand_32 = None
    expand_33: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_43, [1, 12, 64, 1024]);  permute_43 = None
    view_152: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_33, [12, 64, 1024]);  expand_33 = None
    bmm_16: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_151, view_152)
    view_153: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_16, [1, 12, 1024, 1024]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_153, full_default);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_17: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_158, 0, 0, 9223372036854775807)
    slice_18: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_18, div_16, full_default_1);  slice_18 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_25: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
    exp_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_25 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_111: "f32[1, 12, 1024, 1024]" = native_dropout_25[0]
    getitem_112: "b8[1, 12, 1024, 1024]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_34: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_111, [1, 12, 1024, 1024]);  getitem_111 = None
    view_154: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_34, [12, 1024, 1024]);  expand_34 = None
    expand_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_42, [1, 12, 1024, 64])
    view_155: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_35, [12, 1024, 64]);  expand_35 = None
    bmm_17: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_154, view_155)
    view_156: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 12, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_44: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    clone_8: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_157: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_8, [1, 1024, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_158: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_157, [-1, 768]);  view_157 = None
    addmm_33: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_67, view_158, primals_68);  primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_159: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 1024, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_159, 0.1, True);  view_159 = None
    getitem_113: "f32[1, 1024, 768]" = native_dropout_26[0]
    getitem_114: "b8[1, 1024, 768]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_67: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_113, add_64);  getitem_113 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_115: "f32[1, 1024, 1]" = var_mean_17[0]
    getitem_116: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    add_68: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_116);  getitem_116 = None
    mul_66: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_67: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_133)
    add_69: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_134);  mul_67 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_160: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_69, [-1, 768]);  add_69 = None
    addmm_34: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_69, view_160, primals_70);  primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_161: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    pow_9: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_161, 3.0)
    mul_69: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_70: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_161, mul_69);  view_161 = mul_69 = None
    mul_70: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
    tanh_8: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    add_71: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    mul_71: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_162: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_71, [-1, 3072]);  mul_71 = None
    addmm_35: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_71, view_162, primals_72);  primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_163: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 1024, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_163, 0.1, True);  view_163 = None
    getitem_117: "f32[1, 1024, 768]" = native_dropout_27[0]
    getitem_118: "b8[1, 1024, 768]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_72: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_67, getitem_117);  add_67 = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_119: "f32[1, 1024, 1]" = var_mean_18[0]
    getitem_120: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    add_73: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_27: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_120);  getitem_120 = None
    mul_72: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    mul_73: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_135)
    add_74: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_136);  mul_73 = primals_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_164: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_74, [-1, 768]);  add_74 = None
    addmm_36: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_73, view_164, primals_74);  primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_165: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_36, [1, 1024, 2304]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_165, [768, 768, 768], 2);  view_165 = None
    getitem_121: "f32[1, 1024, 768]" = split_with_sizes_9[0]
    getitem_122: "f32[1, 1024, 768]" = split_with_sizes_9[1]
    getitem_123: "f32[1, 1024, 768]" = split_with_sizes_9[2];  split_with_sizes_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_166: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_121, [1, 1024, 12, 64]);  getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_45: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_167: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_122, [1, 1024, 12, 64]);  getitem_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_46: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_168: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_123, [1, 1024, 12, 64]);  getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_48: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2])
    expand_36: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_45, [1, 12, 1024, 64]);  permute_45 = None
    view_169: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_36, [12, 1024, 64]);  expand_36 = None
    expand_37: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_48, [1, 12, 64, 1024]);  permute_48 = None
    view_170: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_37, [12, 64, 1024]);  expand_37 = None
    bmm_18: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_169, view_170)
    view_171: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_18, [1, 12, 1024, 1024]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_171, full_default);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_19: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_159, 0, 0, 9223372036854775807)
    slice_20: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_19, 1, 0, 9223372036854775807);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_20, div_18, full_default_1);  slice_20 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
    sub_28: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
    exp_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_28 = torch.ops.aten.native_dropout.default(div_19, 0.1, True);  div_19 = None
    getitem_124: "f32[1, 12, 1024, 1024]" = native_dropout_28[0]
    getitem_125: "b8[1, 12, 1024, 1024]" = native_dropout_28[1];  native_dropout_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_38: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_124, [1, 12, 1024, 1024]);  getitem_124 = None
    view_172: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_38, [12, 1024, 1024]);  expand_38 = None
    expand_39: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_47, [1, 12, 1024, 64])
    view_173: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_39, [12, 1024, 64]);  expand_39 = None
    bmm_19: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_172, view_173)
    view_174: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 12, 1024, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_49: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    clone_9: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_175: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_9, [1, 1024, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_176: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_175, [-1, 768]);  view_175 = None
    addmm_37: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_75, view_176, primals_76);  primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_177: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_37, [1, 1024, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_177, 0.1, True);  view_177 = None
    getitem_126: "f32[1, 1024, 768]" = native_dropout_29[0]
    getitem_127: "b8[1, 1024, 768]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_75: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_126, add_72);  getitem_126 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_128: "f32[1, 1024, 1]" = var_mean_19[0]
    getitem_129: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    add_76: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_129);  getitem_129 = None
    mul_74: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_75: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_137)
    add_77: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_138);  mul_75 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_178: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_77, [-1, 768]);  add_77 = None
    addmm_38: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_77, view_178, primals_78);  primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_179: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_38, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_179, 0.5)
    pow_10: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_179, 3.0)
    mul_77: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_78: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_179, mul_77);  view_179 = mul_77 = None
    mul_78: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
    tanh_9: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    add_79: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    mul_79: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_180: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_79, [-1, 3072]);  mul_79 = None
    addmm_39: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_79, view_180, primals_80);  primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_181: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_39, [1, 1024, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_181, 0.1, True);  view_181 = None
    getitem_130: "f32[1, 1024, 768]" = native_dropout_30[0]
    getitem_131: "b8[1, 1024, 768]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_80: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_75, getitem_130);  add_75 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_132: "f32[1, 1024, 1]" = var_mean_20[0]
    getitem_133: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    add_81: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_30: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_133);  getitem_133 = None
    mul_80: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    mul_81: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_139)
    add_82: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_140);  mul_81 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_182: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_82, [-1, 768]);  add_82 = None
    addmm_40: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_81, view_182, primals_82);  primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_183: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_40, [1, 1024, 2304]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_183, [768, 768, 768], 2);  view_183 = None
    getitem_134: "f32[1, 1024, 768]" = split_with_sizes_10[0]
    getitem_135: "f32[1, 1024, 768]" = split_with_sizes_10[1]
    getitem_136: "f32[1, 1024, 768]" = split_with_sizes_10[2];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_184: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_134, [1, 1024, 12, 64]);  getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_50: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_185: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_135, [1, 1024, 12, 64]);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_51: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_186: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_136, [1, 1024, 12, 64]);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_52: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_53: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_51, [0, 1, 3, 2])
    expand_40: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [1, 12, 1024, 64]);  permute_50 = None
    view_187: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_40, [12, 1024, 64]);  expand_40 = None
    expand_41: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_53, [1, 12, 64, 1024]);  permute_53 = None
    view_188: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_41, [12, 64, 1024]);  expand_41 = None
    bmm_20: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_20, [1, 12, 1024, 1024]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_20: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_189, full_default);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_21: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_160, 0, 0, 9223372036854775807)
    slice_22: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_22, div_20, full_default_1);  slice_22 = div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_31: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
    exp_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_20: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_31 = torch.ops.aten.native_dropout.default(div_21, 0.1, True);  div_21 = None
    getitem_137: "f32[1, 12, 1024, 1024]" = native_dropout_31[0]
    getitem_138: "b8[1, 12, 1024, 1024]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_42: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_137, [1, 12, 1024, 1024]);  getitem_137 = None
    view_190: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_42, [12, 1024, 1024]);  expand_42 = None
    expand_43: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_52, [1, 12, 1024, 64])
    view_191: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_43, [12, 1024, 64]);  expand_43 = None
    bmm_21: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 12, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_54: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_10: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_193: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_10, [1, 1024, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_194: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_193, [-1, 768]);  view_193 = None
    addmm_41: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_83, view_194, primals_84);  primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_195: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_41, [1, 1024, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_195, 0.1, True);  view_195 = None
    getitem_139: "f32[1, 1024, 768]" = native_dropout_32[0]
    getitem_140: "b8[1, 1024, 768]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_83: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_139, add_80);  getitem_139 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_141: "f32[1, 1024, 1]" = var_mean_21[0]
    getitem_142: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    add_84: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_142);  getitem_142 = None
    mul_82: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    mul_83: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, primals_141)
    add_85: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_83, primals_142);  mul_83 = primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_196: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_85, [-1, 768]);  add_85 = None
    addmm_42: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_85, view_196, primals_86);  primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_197: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_42, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_11: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_85: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_86: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_197, mul_85);  view_197 = mul_85 = None
    mul_86: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
    tanh_10: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    add_87: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    mul_87: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_198: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_87, [-1, 3072]);  mul_87 = None
    addmm_43: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_87, view_198, primals_88);  primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_199: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_43, [1, 1024, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_199, 0.1, True);  view_199 = None
    getitem_143: "f32[1, 1024, 768]" = native_dropout_33[0]
    getitem_144: "b8[1, 1024, 768]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_88: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_83, getitem_143);  add_83 = getitem_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_145: "f32[1, 1024, 1]" = var_mean_22[0]
    getitem_146: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    add_89: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-05);  getitem_145 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_33: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_146);  getitem_146 = None
    mul_88: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    mul_89: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, primals_143)
    add_90: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, primals_144);  mul_89 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_200: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_90, [-1, 768]);  add_90 = None
    addmm_44: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_89, view_200, primals_90);  primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_201: "f32[1, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_44, [1, 1024, 2304]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_201, [768, 768, 768], 2);  view_201 = None
    getitem_147: "f32[1, 1024, 768]" = split_with_sizes_11[0]
    getitem_148: "f32[1, 1024, 768]" = split_with_sizes_11[1]
    getitem_149: "f32[1, 1024, 768]" = split_with_sizes_11[2];  split_with_sizes_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_202: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_147, [1, 1024, 12, 64]);  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_55: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_203: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_148, [1, 1024, 12, 64]);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_56: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_204: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_149, [1, 1024, 12, 64]);  getitem_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_57: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_58: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_56, [0, 1, 3, 2])
    expand_44: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_55, [1, 12, 1024, 64]);  permute_55 = None
    view_205: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_44, [12, 1024, 64]);  expand_44 = None
    expand_45: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_58, [1, 12, 64, 1024]);  permute_58 = None
    view_206: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_45, [12, 64, 1024]);  expand_45 = None
    bmm_22: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_205, view_206)
    view_207: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_22, [1, 12, 1024, 1024]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_207, full_default);  view_207 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_23: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_161, 0, 0, 9223372036854775807)
    slice_24: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_23, 1, 0, 9223372036854775807);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_24, div_22, full_default_1);  slice_24 = div_22 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_34: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
    exp_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_34 = torch.ops.aten.native_dropout.default(div_23, 0.1, True);  div_23 = None
    getitem_150: "f32[1, 12, 1024, 1024]" = native_dropout_34[0]
    getitem_151: "b8[1, 12, 1024, 1024]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_46: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_150, [1, 12, 1024, 1024]);  getitem_150 = None
    view_208: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_46, [12, 1024, 1024]);  expand_46 = None
    expand_47: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [1, 12, 1024, 64])
    view_209: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_47, [12, 1024, 64]);  expand_47 = None
    bmm_23: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_208, view_209)
    view_210: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 12, 1024, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_59: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_11: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_211: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_11, [1, 1024, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_212: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_211, [-1, 768]);  view_211 = None
    addmm_45: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_91, view_212, primals_92);  primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_213: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_45, [1, 1024, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_213, 0.1, True);  view_213 = None
    getitem_152: "f32[1, 1024, 768]" = native_dropout_35[0]
    getitem_153: "b8[1, 1024, 768]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_91: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_152, add_88);  getitem_152 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_154: "f32[1, 1024, 1]" = var_mean_23[0]
    getitem_155: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    add_92: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_155);  getitem_155 = None
    mul_90: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    mul_91: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_145)
    add_93: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_146);  mul_91 = primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_214: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_93, [-1, 768]);  add_93 = None
    addmm_46: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_93, view_214, primals_94);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_215: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_215, 0.5)
    pow_12: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_215, 3.0)
    mul_93: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_94: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_215, mul_93);  view_215 = mul_93 = None
    mul_94: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
    tanh_11: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    add_95: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    mul_95: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_216: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_95, [-1, 3072]);  mul_95 = None
    addmm_47: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_95, view_216, primals_96);  primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_217: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_47, [1, 1024, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_217, 0.1, True);  view_217 = None
    getitem_156: "f32[1, 1024, 768]" = native_dropout_36[0]
    getitem_157: "b8[1, 1024, 768]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_96: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_91, getitem_156);  add_91 = getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_158: "f32[1, 1024, 1]" = var_mean_24[0]
    getitem_159: "f32[1, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
    add_97: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
    rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_36: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_159);  add_96 = getitem_159 = None
    mul_96: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    mul_97: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, primals_147)
    add_98: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_148);  mul_97 = primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    view_218: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_98, [-1, 1024, 768]);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    permute_60: "f32[768, 2]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    view_219: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_218, [1024, 768])
    mm: "f32[1024, 2]" = torch.ops.aten.mm.default(view_219, permute_60)
    view_220: "f32[1, 1024, 2]" = torch.ops.aten.reshape.default(mm, [1, 1024, 2]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1447, code: sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
    eq: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(primals_162, 0);  primals_162 = None
    convert_element_type: "i64[1, 1024]" = torch.ops.prims.convert_element_type.default(eq, torch.int64);  eq = None
    argmax: "i64[1]" = torch.ops.aten.argmax.default(convert_element_type, -1);  convert_element_type = None
    sub_37: "i64[1]" = torch.ops.aten.sub.Tensor(argmax, 1);  argmax = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1457, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    full_default_24: "i64[1]" = torch.ops.aten.full.default([1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index: "f32[1, 2]" = torch.ops.aten.index.Tensor(view_220, [full_default_24, sub_37]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    permute_63: "f32[2, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    div_24: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    permute_66: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_216, [1, 0]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_67: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    permute_68: "f32[768, 1024]" = torch.ops.aten.permute.default(view_214, [1, 0]);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_25: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    permute_70: "f32[768, 1024]" = torch.ops.aten.permute.default(view_212, [1, 0]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_72: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    permute_73: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_25: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_74: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    permute_75: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_80: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    permute_81: "f32[768, 1024]" = torch.ops.aten.permute.default(view_200, [1, 0]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_27: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_82: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    permute_83: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_198, [1, 0]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_84: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    permute_85: "f32[768, 1024]" = torch.ops.aten.permute.default(view_196, [1, 0]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_28: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    permute_87: "f32[768, 1024]" = torch.ops.aten.permute.default(view_194, [1, 0]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_89: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    permute_90: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_27: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_91: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    permute_92: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_97: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    permute_98: "f32[768, 1024]" = torch.ops.aten.permute.default(view_182, [1, 0]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_30: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_99: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    permute_100: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_180, [1, 0]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_101: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    permute_102: "f32[768, 1024]" = torch.ops.aten.permute.default(view_178, [1, 0]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_31: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    permute_104: "f32[768, 1024]" = torch.ops.aten.permute.default(view_176, [1, 0]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_106: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    permute_107: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_29: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_108: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    permute_109: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_114: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    permute_115: "f32[768, 1024]" = torch.ops.aten.permute.default(view_164, [1, 0]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_33: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    permute_117: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_162, [1, 0]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_118: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    permute_119: "f32[768, 1024]" = torch.ops.aten.permute.default(view_160, [1, 0]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_34: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_120: "f32[768, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    permute_121: "f32[768, 1024]" = torch.ops.aten.permute.default(view_158, [1, 0]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_123: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None
    permute_124: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_31: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_125: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
    permute_126: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_131: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    permute_132: "f32[768, 1024]" = torch.ops.aten.permute.default(view_146, [1, 0]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_36: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_133: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    permute_134: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_144, [1, 0]);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_135: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    permute_136: "f32[768, 1024]" = torch.ops.aten.permute.default(view_142, [1, 0]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_37: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_137: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    permute_138: "f32[768, 1024]" = torch.ops.aten.permute.default(view_140, [1, 0]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_140: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    permute_141: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_137, [0, 2, 1]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_33: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_142: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    permute_143: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_148: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    permute_149: "f32[768, 1024]" = torch.ops.aten.permute.default(view_128, [1, 0]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_39: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    permute_151: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_126, [1, 0]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_152: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    permute_153: "f32[768, 1024]" = torch.ops.aten.permute.default(view_124, [1, 0]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_40: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    permute_155: "f32[768, 1024]" = torch.ops.aten.permute.default(view_122, [1, 0]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_157: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_118, [0, 2, 1]);  view_118 = None
    permute_158: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_35: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_159: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    permute_160: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_165: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    permute_166: "f32[768, 1024]" = torch.ops.aten.permute.default(view_110, [1, 0]);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_42: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_167: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    permute_168: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_108, [1, 0]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_169: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    permute_170: "f32[768, 1024]" = torch.ops.aten.permute.default(view_106, [1, 0]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_43: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    permute_172: "f32[768, 1024]" = torch.ops.aten.permute.default(view_104, [1, 0]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_174: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    permute_175: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_37: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_176: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    permute_177: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_182: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    permute_183: "f32[768, 1024]" = torch.ops.aten.permute.default(view_92, [1, 0]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_45: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_184: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    permute_185: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_90, [1, 0]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_186: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    permute_187: "f32[768, 1024]" = torch.ops.aten.permute.default(view_88, [1, 0]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_46: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_188: "f32[768, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    permute_189: "f32[768, 1024]" = torch.ops.aten.permute.default(view_86, [1, 0]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_191: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    permute_192: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_39: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_193: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    permute_194: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_199: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    permute_200: "f32[768, 1024]" = torch.ops.aten.permute.default(view_74, [1, 0]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_48: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_201: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    permute_202: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_72, [1, 0]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_203: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    permute_204: "f32[768, 1024]" = torch.ops.aten.permute.default(view_70, [1, 0]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_49: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_205: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    permute_206: "f32[768, 1024]" = torch.ops.aten.permute.default(view_68, [1, 0]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_208: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    permute_209: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_41: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_210: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    permute_211: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_216: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    permute_217: "f32[768, 1024]" = torch.ops.aten.permute.default(view_56, [1, 0]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_51: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_218: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    permute_219: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_54, [1, 0]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_220: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    permute_221: "f32[768, 1024]" = torch.ops.aten.permute.default(view_52, [1, 0]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_52: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_222: "f32[768, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    permute_223: "f32[768, 1024]" = torch.ops.aten.permute.default(view_50, [1, 0]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_225: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    permute_226: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_43: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_227: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    permute_228: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_233: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    permute_234: "f32[768, 1024]" = torch.ops.aten.permute.default(view_38, [1, 0]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_54: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_235: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    permute_236: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_36, [1, 0]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_237: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    permute_238: "f32[768, 1024]" = torch.ops.aten.permute.default(view_34, [1, 0]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_55: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    permute_240: "f32[768, 1024]" = torch.ops.aten.permute.default(view_32, [1, 0]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_242: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    permute_243: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_45: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_244: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    permute_245: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_250: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    permute_251: "f32[768, 1024]" = torch.ops.aten.permute.default(view_20, [1, 0]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_57: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_252: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    permute_253: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_18, [1, 0]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_254: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    permute_255: "f32[768, 1024]" = torch.ops.aten.permute.default(view_16, [1, 0]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_58: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
    permute_257: "f32[768, 1024]" = torch.ops.aten.permute.default(view_14, [1, 0]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_259: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    permute_260: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_47: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_261: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    permute_262: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_267: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    permute_268: "f32[768, 1024]" = torch.ops.aten.permute.default(view_2, [1, 0]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_60: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [view_218, permute_1, permute_2, permute_6, permute_7, permute_11, permute_12, permute_16, permute_17, permute_21, permute_22, permute_26, permute_27, permute_31, permute_32, permute_36, permute_37, permute_41, permute_42, permute_46, permute_47, permute_51, permute_52, permute_56, permute_57, index, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, view, view_1, getitem_1, mul, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, getitem_86, getitem_88, mul_50, addmm_26, tanh_6, getitem_92, mul_56, getitem_99, getitem_101, mul_58, addmm_30, tanh_7, getitem_105, mul_64, getitem_112, getitem_114, mul_66, addmm_34, tanh_8, getitem_118, mul_72, getitem_125, getitem_127, mul_74, addmm_38, tanh_9, getitem_131, mul_80, getitem_138, getitem_140, mul_82, addmm_42, tanh_10, getitem_144, mul_88, getitem_151, getitem_153, mul_90, addmm_46, tanh_11, getitem_157, mul_96, view_219, sub_37, full_default_24, permute_63, div_24, permute_65, permute_66, permute_67, permute_68, div_25, permute_69, permute_70, permute_72, permute_73, alias_25, permute_74, permute_75, permute_80, permute_81, div_27, permute_82, permute_83, permute_84, permute_85, div_28, permute_86, permute_87, permute_89, permute_90, alias_27, permute_91, permute_92, permute_97, permute_98, div_30, permute_99, permute_100, permute_101, permute_102, div_31, permute_103, permute_104, permute_106, permute_107, alias_29, permute_108, permute_109, permute_114, permute_115, div_33, permute_116, permute_117, permute_118, permute_119, div_34, permute_120, permute_121, permute_123, permute_124, alias_31, permute_125, permute_126, permute_131, permute_132, div_36, permute_133, permute_134, permute_135, permute_136, div_37, permute_137, permute_138, permute_140, permute_141, alias_33, permute_142, permute_143, permute_148, permute_149, div_39, permute_150, permute_151, permute_152, permute_153, div_40, permute_154, permute_155, permute_157, permute_158, alias_35, permute_159, permute_160, permute_165, permute_166, div_42, permute_167, permute_168, permute_169, permute_170, div_43, permute_171, permute_172, permute_174, permute_175, alias_37, permute_176, permute_177, permute_182, permute_183, div_45, permute_184, permute_185, permute_186, permute_187, div_46, permute_188, permute_189, permute_191, permute_192, alias_39, permute_193, permute_194, permute_199, permute_200, div_48, permute_201, permute_202, permute_203, permute_204, div_49, permute_205, permute_206, permute_208, permute_209, alias_41, permute_210, permute_211, permute_216, permute_217, div_51, permute_218, permute_219, permute_220, permute_221, div_52, permute_222, permute_223, permute_225, permute_226, alias_43, permute_227, permute_228, permute_233, permute_234, div_54, permute_235, permute_236, permute_237, permute_238, div_55, permute_239, permute_240, permute_242, permute_243, alias_45, permute_244, permute_245, permute_250, permute_251, div_57, permute_252, permute_253, permute_254, permute_255, div_58, permute_256, permute_257, permute_259, permute_260, alias_47, permute_261, permute_262, permute_267, permute_268, div_60]
    