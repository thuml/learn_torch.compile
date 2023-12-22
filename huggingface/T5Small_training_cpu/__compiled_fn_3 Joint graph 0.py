from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[512]"; primals_2: "f32[512]"; primals_3: "f32[512]"; primals_4: "f32[512]"; primals_5: "f32[512]"; primals_6: "f32[512]"; primals_7: "f32[512]"; primals_8: "f32[512]"; primals_9: "f32[512]"; primals_10: "f32[512]"; primals_11: "f32[512]"; primals_12: "f32[512]"; primals_13: "f32[512]"; primals_14: "f32[512]"; primals_15: "f32[512]"; primals_16: "f32[512]"; primals_17: "f32[512]"; primals_18: "f32[512]"; primals_19: "f32[512]"; primals_20: "f32[512]"; primals_21: "f32[512]"; primals_22: "f32[512]"; primals_23: "f32[512]"; primals_24: "f32[512]"; primals_25: "f32[512]"; primals_26: "f32[512]"; primals_27: "f32[512]"; primals_28: "f32[512]"; primals_29: "f32[512]"; primals_30: "f32[512]"; primals_31: "f32[512]"; primals_32: "f32[512]"; primals_33: "f32[32128, 512]"; primals_34: "f32[512, 512]"; primals_35: "f32[512, 512]"; primals_36: "f32[512, 512]"; primals_37: "f32[32, 8]"; primals_38: "f32[512, 512]"; primals_39: "f32[2048, 512]"; primals_40: "f32[512, 2048]"; primals_41: "f32[512, 512]"; primals_42: "f32[512, 512]"; primals_43: "f32[512, 512]"; primals_44: "f32[512, 512]"; primals_45: "f32[2048, 512]"; primals_46: "f32[512, 2048]"; primals_47: "f32[512, 512]"; primals_48: "f32[512, 512]"; primals_49: "f32[512, 512]"; primals_50: "f32[512, 512]"; primals_51: "f32[2048, 512]"; primals_52: "f32[512, 2048]"; primals_53: "f32[512, 512]"; primals_54: "f32[512, 512]"; primals_55: "f32[512, 512]"; primals_56: "f32[512, 512]"; primals_57: "f32[2048, 512]"; primals_58: "f32[512, 2048]"; primals_59: "f32[512, 512]"; primals_60: "f32[512, 512]"; primals_61: "f32[512, 512]"; primals_62: "f32[512, 512]"; primals_63: "f32[2048, 512]"; primals_64: "f32[512, 2048]"; primals_65: "f32[512, 512]"; primals_66: "f32[512, 512]"; primals_67: "f32[512, 512]"; primals_68: "f32[512, 512]"; primals_69: "f32[2048, 512]"; primals_70: "f32[512, 2048]"; primals_71: "f32[512, 512]"; primals_72: "f32[512, 512]"; primals_73: "f32[512, 512]"; primals_74: "f32[32, 8]"; primals_75: "f32[512, 512]"; primals_76: "f32[512, 512]"; primals_77: "f32[512, 512]"; primals_78: "f32[512, 512]"; primals_79: "f32[512, 512]"; primals_80: "f32[2048, 512]"; primals_81: "f32[512, 2048]"; primals_82: "f32[512, 512]"; primals_83: "f32[512, 512]"; primals_84: "f32[512, 512]"; primals_85: "f32[512, 512]"; primals_86: "f32[512, 512]"; primals_87: "f32[512, 512]"; primals_88: "f32[512, 512]"; primals_89: "f32[512, 512]"; primals_90: "f32[2048, 512]"; primals_91: "f32[512, 2048]"; primals_92: "f32[512, 512]"; primals_93: "f32[512, 512]"; primals_94: "f32[512, 512]"; primals_95: "f32[512, 512]"; primals_96: "f32[512, 512]"; primals_97: "f32[512, 512]"; primals_98: "f32[512, 512]"; primals_99: "f32[512, 512]"; primals_100: "f32[2048, 512]"; primals_101: "f32[512, 2048]"; primals_102: "f32[512, 512]"; primals_103: "f32[512, 512]"; primals_104: "f32[512, 512]"; primals_105: "f32[512, 512]"; primals_106: "f32[512, 512]"; primals_107: "f32[512, 512]"; primals_108: "f32[512, 512]"; primals_109: "f32[512, 512]"; primals_110: "f32[2048, 512]"; primals_111: "f32[512, 2048]"; primals_112: "f32[512, 512]"; primals_113: "f32[512, 512]"; primals_114: "f32[512, 512]"; primals_115: "f32[512, 512]"; primals_116: "f32[512, 512]"; primals_117: "f32[512, 512]"; primals_118: "f32[512, 512]"; primals_119: "f32[512, 512]"; primals_120: "f32[2048, 512]"; primals_121: "f32[512, 2048]"; primals_122: "f32[512, 512]"; primals_123: "f32[512, 512]"; primals_124: "f32[512, 512]"; primals_125: "f32[512, 512]"; primals_126: "f32[512, 512]"; primals_127: "f32[512, 512]"; primals_128: "f32[512, 512]"; primals_129: "f32[512, 512]"; primals_130: "f32[2048, 512]"; primals_131: "f32[512, 2048]"; primals_132: "f32[32128, 512]"; primals_133: "i64[1, 1024]"; primals_134: "i64[1, 1024]"; primals_135: "i64[1, 1024]"; tangents_1: "f32[]"; tangents_2: "f32[1, 1024, 32128]"; tangents_3: "f32[1, 8, 1024, 64]"; tangents_4: "f32[1, 8, 1024, 64]"; tangents_5: "f32[1, 8, 1024, 64]"; tangents_6: "f32[1, 8, 1024, 64]"; tangents_7: "f32[1, 8, 1024, 64]"; tangents_8: "f32[1, 8, 1024, 64]"; tangents_9: "f32[1, 8, 1024, 64]"; tangents_10: "f32[1, 8, 1024, 64]"; tangents_11: "f32[1, 8, 1024, 64]"; tangents_12: "f32[1, 8, 1024, 64]"; tangents_13: "f32[1, 8, 1024, 64]"; tangents_14: "f32[1, 8, 1024, 64]"; tangents_15: "f32[1, 8, 1024, 64]"; tangents_16: "f32[1, 8, 1024, 64]"; tangents_17: "f32[1, 8, 1024, 64]"; tangents_18: "f32[1, 8, 1024, 64]"; tangents_19: "f32[1, 8, 1024, 64]"; tangents_20: "f32[1, 8, 1024, 64]"; tangents_21: "f32[1, 8, 1024, 64]"; tangents_22: "f32[1, 8, 1024, 64]"; tangents_23: "f32[1, 8, 1024, 64]"; tangents_24: "f32[1, 8, 1024, 64]"; tangents_25: "f32[1, 8, 1024, 64]"; tangents_26: "f32[1, 8, 1024, 64]"; tangents_27: "f32[1, 1024, 512]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1011, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 1024]" = torch.ops.aten.view.default(primals_133, [-1, 1024]);  primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding: "f32[1, 1024, 512]" = torch.ops.aten.embedding.default(primals_33, view)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1033, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    full: "f32[1, 1024]" = torch.ops.aten.full.default([1, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_1: "f32[1, 1024]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
    unsqueeze_1: "f32[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_2: "f32[1, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(1.0, slice_2);  slice_2 = None
    mul: "f32[1, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    native_dropout = torch.ops.aten.native_dropout.default(embedding, 0.1, True);  embedding = None
    getitem: "f32[1, 1024, 512]" = native_dropout[0]
    getitem_1: "b8[1, 1024, 512]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_1: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem, 2)
    mean: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    alias: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt)
    mul_1: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(getitem, rsqrt)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_2: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_1, mul_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute: "f32[512, 512]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    view_1: "f32[1024, 512]" = torch.ops.aten.view.default(mul_2, [1024, 512])
    mm: "f32[1024, 512]" = torch.ops.aten.mm.default(view_1, permute)
    view_2: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm, [1, 1024, 512]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_3: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_2, [1, -1, 8, 64]);  view_2 = None
    permute_1: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    view_4: "f32[1024, 512]" = torch.ops.aten.view.default(mul_2, [1024, 512])
    mm_1: "f32[1024, 512]" = torch.ops.aten.mm.default(view_4, permute_2)
    view_5: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_1, [1, 1024, 512]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_6: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_5, [1, -1, 8, 64]);  view_5 = None
    permute_3: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    view_7: "f32[1024, 512]" = torch.ops.aten.view.default(mul_2, [1024, 512]);  mul_2 = None
    mm_2: "f32[1024, 512]" = torch.ops.aten.mm.default(view_7, permute_4)
    view_8: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_2, [1, 1024, 512]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_9: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_8, [1, -1, 8, 64]);  view_8 = None
    permute_5: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_6: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_1, [1, 8, 1024, 64]);  permute_1 = None
    view_10: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand, [8, 1024, 64]);  expand = None
    expand_1: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_6, [1, 8, 64, 1024]);  permute_6 = None
    view_11: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_1, [8, 64, 1024]);  expand_1 = None
    bmm: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_10, view_11)
    view_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm, [1, 8, 1024, 1024]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:441, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_3: "i64[1024]" = torch.ops.aten.slice.Tensor(iota, 0, 0, 9223372036854775807);  iota = None
    unsqueeze_2: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:442, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_1, 0);  iota_1 = None
    slice_4: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 1, 0, 9223372036854775807);  unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:443, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    sub_1: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(slice_4, unsqueeze_2);  slice_4 = unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:414, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    gt: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(sub_1, 0)
    convert_element_type: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
    mul_3: "i64[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
    add_1: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(mul_3, 0);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:415, code: relative_position = torch.abs(relative_position)
    abs_1: "i64[1024, 1024]" = torch.ops.aten.abs.default(sub_1);  sub_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:422, code: is_small = relative_position < max_exact
    lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(abs_1, 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:426, code: torch.log(relative_position.float() / max_exact)
    convert_element_type_1: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
    div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
    log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
    div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
    mul_4: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:429, code: ).to(torch.long)
    convert_element_type_2: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_4, torch.int64);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:425, code: relative_position_if_large = max_exact + (
    add_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:431, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_1: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:430, code: relative_position_if_large = torch.min(
    minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_2, full_1);  add_2 = full_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:434, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
    add_3: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    embedding_1: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(primals_37, add_3);  primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_7: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
    unsqueeze_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    add_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_4, mul);  unsqueeze_4 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_12, add_4);  view_12 = None
    view_13: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_5, [8, 1024, 1024]);  add_5 = None
    view_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_13, [1, 8, 1024, 1024]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_14, [-1], True)
    sub_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_14, amax);  view_14 = amax = None
    exp: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_2, 0.1, True);  div_2 = None
    getitem_2: "f32[1, 8, 1024, 1024]" = native_dropout_1[0]
    getitem_3: "b8[1, 8, 1024, 1024]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_2, [1, 8, 1024, 1024]);  getitem_2 = None
    view_15: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_2, [8, 1024, 1024]);  expand_2 = None
    expand_3: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [1, 8, 1024, 64]);  permute_5 = None
    view_16: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_3, [8, 1024, 64]);  expand_3 = None
    bmm_1: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_15, view_16)
    view_17: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_1, [1, 8, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_8: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
    clone: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_18: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone, [1, -1, 512]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_9: "f32[512, 512]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    view_19: "f32[1024, 512]" = torch.ops.aten.view.default(view_18, [1024, 512]);  view_18 = None
    mm_3: "f32[1024, 512]" = torch.ops.aten.mm.default(view_19, permute_9)
    view_20: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_3, [1, 1024, 512]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_20, 0.1, True);  view_20 = None
    getitem_4: "f32[1, 1024, 512]" = native_dropout_2[0]
    getitem_5: "b8[1, 1024, 512]" = native_dropout_2[1];  native_dropout_2 = None
    add_6: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(getitem, getitem_4);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_2: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
    mean_1: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_7: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    alias_2: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_1)
    mul_5: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_6: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_2, mul_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_10: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    view_21: "f32[1024, 512]" = torch.ops.aten.view.default(mul_6, [1024, 512]);  mul_6 = None
    mm_4: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_21, permute_10)
    view_22: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_4, [1, 1024, 2048]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_22);  view_22 = None
    alias_3: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(relu, 0.1, True);  relu = None
    getitem_6: "f32[1, 1024, 2048]" = native_dropout_3[0]
    getitem_7: "b8[1, 1024, 2048]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_11: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    view_23: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_6, [1024, 2048]);  getitem_6 = None
    mm_5: "f32[1024, 512]" = torch.ops.aten.mm.default(view_23, permute_11)
    view_24: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_5, [1, 1024, 512]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_4 = torch.ops.aten.native_dropout.default(view_24, 0.1, True);  view_24 = None
    getitem_8: "f32[1, 1024, 512]" = native_dropout_4[0]
    getitem_9: "b8[1, 1024, 512]" = native_dropout_4[1];  native_dropout_4 = None
    add_8: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_6, getitem_8);  getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_3: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_8, 2)
    mean_2: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    alias_4: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_2)
    mul_7: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_8, rsqrt_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_8: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_3, mul_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    view_25: "f32[1024, 512]" = torch.ops.aten.view.default(mul_8, [1024, 512])
    mm_6: "f32[1024, 512]" = torch.ops.aten.mm.default(view_25, permute_12)
    view_26: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_6, [1, 1024, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_27: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_26, [1, -1, 8, 64]);  view_26 = None
    permute_13: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_14: "f32[512, 512]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    view_28: "f32[1024, 512]" = torch.ops.aten.view.default(mul_8, [1024, 512])
    mm_7: "f32[1024, 512]" = torch.ops.aten.mm.default(view_28, permute_14)
    view_29: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_7, [1, 1024, 512]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_30: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_29, [1, -1, 8, 64]);  view_29 = None
    permute_15: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_16: "f32[512, 512]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    view_31: "f32[1024, 512]" = torch.ops.aten.view.default(mul_8, [1024, 512]);  mul_8 = None
    mm_8: "f32[1024, 512]" = torch.ops.aten.mm.default(view_31, permute_16)
    view_32: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_8, [1, 1024, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_33: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_32, [1, -1, 8, 64]);  view_32 = None
    permute_17: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_18: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_4: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_13, [1, 8, 1024, 64]);  permute_13 = None
    view_34: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_4, [8, 1024, 64]);  expand_4 = None
    expand_5: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [1, 8, 64, 1024]);  permute_18 = None
    view_35: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_5, [8, 64, 1024]);  expand_5 = None
    bmm_2: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_34, view_35)
    view_36: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [1, 8, 1024, 1024]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_36, add_4);  view_36 = None
    view_37: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_10, [8, 1024, 1024]);  add_10 = None
    view_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_37, [1, 8, 1024, 1024]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_1: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_38, [-1], True)
    sub_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_38, amax_1);  view_38 = amax_1 = None
    exp_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_5 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_10: "f32[1, 8, 1024, 1024]" = native_dropout_5[0]
    getitem_11: "b8[1, 8, 1024, 1024]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_10, [1, 8, 1024, 1024]);  getitem_10 = None
    view_39: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_6, [8, 1024, 1024]);  expand_6 = None
    expand_7: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [1, 8, 1024, 64]);  permute_17 = None
    view_40: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_7, [8, 1024, 64]);  expand_7 = None
    bmm_3: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_39, view_40)
    view_41: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_3, [1, 8, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_19: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    clone_1: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_42: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_1, [1, -1, 512]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_20: "f32[512, 512]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    view_43: "f32[1024, 512]" = torch.ops.aten.view.default(view_42, [1024, 512]);  view_42 = None
    mm_9: "f32[1024, 512]" = torch.ops.aten.mm.default(view_43, permute_20)
    view_44: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_9, [1, 1024, 512]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_44, 0.1, True);  view_44 = None
    getitem_12: "f32[1, 1024, 512]" = native_dropout_6[0]
    getitem_13: "b8[1, 1024, 512]" = native_dropout_6[1];  native_dropout_6 = None
    add_11: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_8, getitem_12);  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_4: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
    mean_3: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_12: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    alias_6: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_3)
    mul_9: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_10: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_4, mul_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_21: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    view_45: "f32[1024, 512]" = torch.ops.aten.view.default(mul_10, [1024, 512]);  mul_10 = None
    mm_10: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_45, permute_21)
    view_46: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_10, [1, 1024, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_1: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_46);  view_46 = None
    alias_7: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_7 = torch.ops.aten.native_dropout.default(relu_1, 0.1, True);  relu_1 = None
    getitem_14: "f32[1, 1024, 2048]" = native_dropout_7[0]
    getitem_15: "b8[1, 1024, 2048]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_22: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    view_47: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_14, [1024, 2048]);  getitem_14 = None
    mm_11: "f32[1024, 512]" = torch.ops.aten.mm.default(view_47, permute_22)
    view_48: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_11, [1, 1024, 512]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_48, 0.1, True);  view_48 = None
    getitem_16: "f32[1, 1024, 512]" = native_dropout_8[0]
    getitem_17: "b8[1, 1024, 512]" = native_dropout_8[1];  native_dropout_8 = None
    add_13: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_11, getitem_16);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_5: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
    mean_4: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_14: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    alias_8: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_4)
    mul_11: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_12: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_5, mul_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_23: "f32[512, 512]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    view_49: "f32[1024, 512]" = torch.ops.aten.view.default(mul_12, [1024, 512])
    mm_12: "f32[1024, 512]" = torch.ops.aten.mm.default(view_49, permute_23)
    view_50: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_12, [1, 1024, 512]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_51: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_50, [1, -1, 8, 64]);  view_50 = None
    permute_24: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    view_52: "f32[1024, 512]" = torch.ops.aten.view.default(mul_12, [1024, 512])
    mm_13: "f32[1024, 512]" = torch.ops.aten.mm.default(view_52, permute_25)
    view_53: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_13, [1, 1024, 512]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_54: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_53, [1, -1, 8, 64]);  view_53 = None
    permute_26: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_27: "f32[512, 512]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    view_55: "f32[1024, 512]" = torch.ops.aten.view.default(mul_12, [1024, 512]);  mul_12 = None
    mm_14: "f32[1024, 512]" = torch.ops.aten.mm.default(view_55, permute_27)
    view_56: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_14, [1, 1024, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_57: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_56, [1, -1, 8, 64]);  view_56 = None
    permute_28: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_29: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_8: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_24, [1, 8, 1024, 64]);  permute_24 = None
    view_58: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_8, [8, 1024, 64]);  expand_8 = None
    expand_9: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_29, [1, 8, 64, 1024]);  permute_29 = None
    view_59: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_9, [8, 64, 1024]);  expand_9 = None
    bmm_4: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_58, view_59)
    view_60: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_4, [1, 8, 1024, 1024]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_15: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_60, add_4);  view_60 = None
    view_61: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_15, [8, 1024, 1024]);  add_15 = None
    view_62: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_61, [1, 8, 1024, 1024]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_2: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_62, [-1], True)
    sub_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_62, amax_2);  view_62 = amax_2 = None
    exp_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_3: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_9: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_9 = torch.ops.aten.native_dropout.default(div_4, 0.1, True);  div_4 = None
    getitem_18: "f32[1, 8, 1024, 1024]" = native_dropout_9[0]
    getitem_19: "b8[1, 8, 1024, 1024]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_18, [1, 8, 1024, 1024]);  getitem_18 = None
    view_63: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_10, [8, 1024, 1024]);  expand_10 = None
    expand_11: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_28, [1, 8, 1024, 64]);  permute_28 = None
    view_64: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_11, [8, 1024, 64]);  expand_11 = None
    bmm_5: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_63, view_64)
    view_65: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_5, [1, 8, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_30: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_2: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_66: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_2, [1, -1, 512]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_31: "f32[512, 512]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    view_67: "f32[1024, 512]" = torch.ops.aten.view.default(view_66, [1024, 512]);  view_66 = None
    mm_15: "f32[1024, 512]" = torch.ops.aten.mm.default(view_67, permute_31)
    view_68: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_15, [1, 1024, 512]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_10 = torch.ops.aten.native_dropout.default(view_68, 0.1, True);  view_68 = None
    getitem_20: "f32[1, 1024, 512]" = native_dropout_10[0]
    getitem_21: "b8[1, 1024, 512]" = native_dropout_10[1];  native_dropout_10 = None
    add_16: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_13, getitem_20);  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_6: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
    mean_5: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_17: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    alias_10: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_5)
    mul_13: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_14: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_6, mul_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_32: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    view_69: "f32[1024, 512]" = torch.ops.aten.view.default(mul_14, [1024, 512]);  mul_14 = None
    mm_16: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_69, permute_32)
    view_70: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_16, [1, 1024, 2048]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_2: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_70);  view_70 = None
    alias_11: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(relu_2, 0.1, True);  relu_2 = None
    getitem_22: "f32[1, 1024, 2048]" = native_dropout_11[0]
    getitem_23: "b8[1, 1024, 2048]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_33: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    view_71: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_22, [1024, 2048]);  getitem_22 = None
    mm_17: "f32[1024, 512]" = torch.ops.aten.mm.default(view_71, permute_33)
    view_72: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_17, [1, 1024, 512]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_72, 0.1, True);  view_72 = None
    getitem_24: "f32[1, 1024, 512]" = native_dropout_12[0]
    getitem_25: "b8[1, 1024, 512]" = native_dropout_12[1];  native_dropout_12 = None
    add_18: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_16, getitem_24);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_7: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_18, 2)
    mean_6: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_19: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    alias_12: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_6)
    mul_15: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_18, rsqrt_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_16: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_7, mul_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    view_73: "f32[1024, 512]" = torch.ops.aten.view.default(mul_16, [1024, 512])
    mm_18: "f32[1024, 512]" = torch.ops.aten.mm.default(view_73, permute_34)
    view_74: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_18, [1, 1024, 512]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_75: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_74, [1, -1, 8, 64]);  view_74 = None
    permute_35: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_36: "f32[512, 512]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    view_76: "f32[1024, 512]" = torch.ops.aten.view.default(mul_16, [1024, 512])
    mm_19: "f32[1024, 512]" = torch.ops.aten.mm.default(view_76, permute_36)
    view_77: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_19, [1, 1024, 512]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_78: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_77, [1, -1, 8, 64]);  view_77 = None
    permute_37: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_38: "f32[512, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    view_79: "f32[1024, 512]" = torch.ops.aten.view.default(mul_16, [1024, 512]);  mul_16 = None
    mm_20: "f32[1024, 512]" = torch.ops.aten.mm.default(view_79, permute_38)
    view_80: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_20, [1, 1024, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_81: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_80, [1, -1, 8, 64]);  view_80 = None
    permute_39: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_40: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_12: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [1, 8, 1024, 64]);  permute_35 = None
    view_82: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_12, [8, 1024, 64]);  expand_12 = None
    expand_13: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_40, [1, 8, 64, 1024]);  permute_40 = None
    view_83: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_13, [8, 64, 1024]);  expand_13 = None
    bmm_6: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_6, [1, 8, 1024, 1024]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_20: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_84, add_4);  view_84 = None
    view_85: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_20, [8, 1024, 1024]);  add_20 = None
    view_86: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_85, [1, 8, 1024, 1024]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_3: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_86, [-1], True)
    sub_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_86, amax_3);  view_86 = amax_3 = None
    exp_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_4: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_13: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_26: "f32[1, 8, 1024, 1024]" = native_dropout_13[0]
    getitem_27: "b8[1, 8, 1024, 1024]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_26, [1, 8, 1024, 1024]);  getitem_26 = None
    view_87: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_14, [8, 1024, 1024]);  expand_14 = None
    expand_15: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_39, [1, 8, 1024, 64]);  permute_39 = None
    view_88: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_15, [8, 1024, 64]);  expand_15 = None
    bmm_7: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_87, view_88)
    view_89: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_7, [1, 8, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_41: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_3: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_90: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_3, [1, -1, 512]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_42: "f32[512, 512]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    view_91: "f32[1024, 512]" = torch.ops.aten.view.default(view_90, [1024, 512]);  view_90 = None
    mm_21: "f32[1024, 512]" = torch.ops.aten.mm.default(view_91, permute_42)
    view_92: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_21, [1, 1024, 512]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_92, 0.1, True);  view_92 = None
    getitem_28: "f32[1, 1024, 512]" = native_dropout_14[0]
    getitem_29: "b8[1, 1024, 512]" = native_dropout_14[1];  native_dropout_14 = None
    add_21: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_18, getitem_28);  getitem_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_8: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
    mean_7: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_22: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    alias_14: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_7)
    mul_17: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_18: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_8, mul_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_43: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    view_93: "f32[1024, 512]" = torch.ops.aten.view.default(mul_18, [1024, 512]);  mul_18 = None
    mm_22: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_93, permute_43)
    view_94: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_22, [1, 1024, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_3: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_94);  view_94 = None
    alias_15: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(relu_3, 0.1, True);  relu_3 = None
    getitem_30: "f32[1, 1024, 2048]" = native_dropout_15[0]
    getitem_31: "b8[1, 1024, 2048]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_44: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    view_95: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_30, [1024, 2048]);  getitem_30 = None
    mm_23: "f32[1024, 512]" = torch.ops.aten.mm.default(view_95, permute_44)
    view_96: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_23, [1, 1024, 512]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_16 = torch.ops.aten.native_dropout.default(view_96, 0.1, True);  view_96 = None
    getitem_32: "f32[1, 1024, 512]" = native_dropout_16[0]
    getitem_33: "b8[1, 1024, 512]" = native_dropout_16[1];  native_dropout_16 = None
    add_23: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_21, getitem_32);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_9: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 2)
    mean_8: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_24: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    alias_16: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_8)
    mul_19: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_23, rsqrt_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_20: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_9, mul_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_45: "f32[512, 512]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    view_97: "f32[1024, 512]" = torch.ops.aten.view.default(mul_20, [1024, 512])
    mm_24: "f32[1024, 512]" = torch.ops.aten.mm.default(view_97, permute_45)
    view_98: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_24, [1, 1024, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_99: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_98, [1, -1, 8, 64]);  view_98 = None
    permute_46: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_47: "f32[512, 512]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    view_100: "f32[1024, 512]" = torch.ops.aten.view.default(mul_20, [1024, 512])
    mm_25: "f32[1024, 512]" = torch.ops.aten.mm.default(view_100, permute_47)
    view_101: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_25, [1, 1024, 512]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_102: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_101, [1, -1, 8, 64]);  view_101 = None
    permute_48: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_49: "f32[512, 512]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    view_103: "f32[1024, 512]" = torch.ops.aten.view.default(mul_20, [1024, 512]);  mul_20 = None
    mm_26: "f32[1024, 512]" = torch.ops.aten.mm.default(view_103, permute_49)
    view_104: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_26, [1, 1024, 512]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_105: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_104, [1, -1, 8, 64]);  view_104 = None
    permute_50: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_51: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_16: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_46, [1, 8, 1024, 64]);  permute_46 = None
    view_106: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_16, [8, 1024, 64]);  expand_16 = None
    expand_17: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_51, [1, 8, 64, 1024]);  permute_51 = None
    view_107: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_17, [8, 64, 1024]);  expand_17 = None
    bmm_8: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_106, view_107)
    view_108: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_8, [1, 8, 1024, 1024]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_25: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_108, add_4);  view_108 = None
    view_109: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_25, [8, 1024, 1024]);  add_25 = None
    view_110: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_109, [1, 8, 1024, 1024]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_4: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_110, [-1], True)
    sub_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_110, amax_4);  view_110 = amax_4 = None
    exp_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_5: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_17: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_17 = torch.ops.aten.native_dropout.default(div_6, 0.1, True);  div_6 = None
    getitem_34: "f32[1, 8, 1024, 1024]" = native_dropout_17[0]
    getitem_35: "b8[1, 8, 1024, 1024]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_18: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_34, [1, 8, 1024, 1024]);  getitem_34 = None
    view_111: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_18, [8, 1024, 1024]);  expand_18 = None
    expand_19: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [1, 8, 1024, 64]);  permute_50 = None
    view_112: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_19, [8, 1024, 64]);  expand_19 = None
    bmm_9: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_111, view_112)
    view_113: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_9, [1, 8, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_52: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    clone_4: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_114: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_4, [1, -1, 512]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    view_115: "f32[1024, 512]" = torch.ops.aten.view.default(view_114, [1024, 512]);  view_114 = None
    mm_27: "f32[1024, 512]" = torch.ops.aten.mm.default(view_115, permute_53)
    view_116: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_27, [1, 1024, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_116, 0.1, True);  view_116 = None
    getitem_36: "f32[1, 1024, 512]" = native_dropout_18[0]
    getitem_37: "b8[1, 1024, 512]" = native_dropout_18[1];  native_dropout_18 = None
    add_26: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_23, getitem_36);  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_10: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 2)
    mean_9: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_27: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    alias_18: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_9)
    mul_21: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_22: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_10, mul_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_54: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    view_117: "f32[1024, 512]" = torch.ops.aten.view.default(mul_22, [1024, 512]);  mul_22 = None
    mm_28: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_117, permute_54)
    view_118: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_28, [1, 1024, 2048]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_4: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_118);  view_118 = None
    alias_19: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_19 = torch.ops.aten.native_dropout.default(relu_4, 0.1, True);  relu_4 = None
    getitem_38: "f32[1, 1024, 2048]" = native_dropout_19[0]
    getitem_39: "b8[1, 1024, 2048]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_55: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    view_119: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_38, [1024, 2048]);  getitem_38 = None
    mm_29: "f32[1024, 512]" = torch.ops.aten.mm.default(view_119, permute_55)
    view_120: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_29, [1, 1024, 512]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_120, 0.1, True);  view_120 = None
    getitem_40: "f32[1, 1024, 512]" = native_dropout_20[0]
    getitem_41: "b8[1, 1024, 512]" = native_dropout_20[1];  native_dropout_20 = None
    add_28: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_26, getitem_40);  getitem_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_11: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_28, 2)
    mean_10: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_29: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    alias_20: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_10)
    mul_23: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_28, rsqrt_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_24: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_11, mul_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_56: "f32[512, 512]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    view_121: "f32[1024, 512]" = torch.ops.aten.view.default(mul_24, [1024, 512])
    mm_30: "f32[1024, 512]" = torch.ops.aten.mm.default(view_121, permute_56)
    view_122: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_30, [1, 1024, 512]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_123: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_122, [1, -1, 8, 64]);  view_122 = None
    permute_57: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_58: "f32[512, 512]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    view_124: "f32[1024, 512]" = torch.ops.aten.view.default(mul_24, [1024, 512])
    mm_31: "f32[1024, 512]" = torch.ops.aten.mm.default(view_124, permute_58)
    view_125: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_31, [1, 1024, 512]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_126: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_125, [1, -1, 8, 64]);  view_125 = None
    permute_59: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_60: "f32[512, 512]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    view_127: "f32[1024, 512]" = torch.ops.aten.view.default(mul_24, [1024, 512]);  mul_24 = None
    mm_32: "f32[1024, 512]" = torch.ops.aten.mm.default(view_127, permute_60)
    view_128: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_32, [1, 1024, 512]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_129: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_128, [1, -1, 8, 64]);  view_128 = None
    permute_61: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_62: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_20: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [1, 8, 1024, 64]);  permute_57 = None
    view_130: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_20, [8, 1024, 64]);  expand_20 = None
    expand_21: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_62, [1, 8, 64, 1024]);  permute_62 = None
    view_131: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_21, [8, 64, 1024]);  expand_21 = None
    bmm_10: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_130, view_131)
    view_132: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_10, [1, 8, 1024, 1024]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_30: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_132, add_4);  view_132 = add_4 = None
    view_133: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_30, [8, 1024, 1024]);  add_30 = None
    view_134: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_133, [1, 8, 1024, 1024]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_5: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_134, [-1], True)
    sub_7: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_134, amax_5);  view_134 = amax_5 = None
    exp_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_6: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_7: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_21: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_21 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_42: "f32[1, 8, 1024, 1024]" = native_dropout_21[0]
    getitem_43: "b8[1, 8, 1024, 1024]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_22: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_42, [1, 8, 1024, 1024]);  getitem_42 = None
    view_135: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_22, [8, 1024, 1024]);  expand_22 = None
    expand_23: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_61, [1, 8, 1024, 64]);  permute_61 = None
    view_136: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_23, [8, 1024, 64]);  expand_23 = None
    bmm_11: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_135, view_136)
    view_137: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_11, [1, 8, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_63: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    clone_5: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_138: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_5, [1, -1, 512]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_64: "f32[512, 512]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    view_139: "f32[1024, 512]" = torch.ops.aten.view.default(view_138, [1024, 512]);  view_138 = None
    mm_33: "f32[1024, 512]" = torch.ops.aten.mm.default(view_139, permute_64)
    view_140: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_33, [1, 1024, 512]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_22 = torch.ops.aten.native_dropout.default(view_140, 0.1, True);  view_140 = None
    getitem_44: "f32[1, 1024, 512]" = native_dropout_22[0]
    getitem_45: "b8[1, 1024, 512]" = native_dropout_22[1];  native_dropout_22 = None
    add_31: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_28, getitem_44);  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_12: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
    mean_11: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_32: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    alias_22: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_11)
    mul_25: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_26: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_12, mul_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_65: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    view_141: "f32[1024, 512]" = torch.ops.aten.view.default(mul_26, [1024, 512]);  mul_26 = None
    mm_34: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_141, permute_65)
    view_142: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_34, [1, 1024, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_5: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_142);  view_142 = None
    alias_23: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(relu_5, 0.1, True);  relu_5 = None
    getitem_46: "f32[1, 1024, 2048]" = native_dropout_23[0]
    getitem_47: "b8[1, 1024, 2048]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_66: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_143: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_46, [1024, 2048]);  getitem_46 = None
    mm_35: "f32[1024, 512]" = torch.ops.aten.mm.default(view_143, permute_66)
    view_144: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_35, [1, 1024, 512]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_144, 0.1, True);  view_144 = None
    getitem_48: "f32[1, 1024, 512]" = native_dropout_24[0]
    getitem_49: "b8[1, 1024, 512]" = native_dropout_24[1];  native_dropout_24 = None
    add_33: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_31, getitem_48);  getitem_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_13: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_33, 2)
    mean_12: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_34: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    alias_24: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_12)
    mul_27: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_33, rsqrt_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_28: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_13, mul_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    native_dropout_25 = torch.ops.aten.native_dropout.default(mul_28, 0.1, True);  mul_28 = None
    getitem_50: "f32[1, 1024, 512]" = native_dropout_25[0]
    getitem_51: "b8[1, 1024, 512]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1011, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_145: "i64[1, 1024]" = torch.ops.aten.view.default(primals_135, [-1, 1024]);  primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding_2: "f32[1, 1024, 512]" = torch.ops.aten.embedding.default(primals_33, view_145);  primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1033, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    full_2: "f32[1, 1024]" = torch.ops.aten.full.default([1, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1036, code: encoder_attention_mask = torch.ones(
    full_3: "i64[1, 1024]" = torch.ops.aten.full.default([1, 1024], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:860, code: seq_ids = torch.arange(seq_length, device=device)
    iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:861, code: causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    unsqueeze_5: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_2, 0)
    unsqueeze_6: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
    slice_5: "i64[1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_6, 2, 0, 9223372036854775807);  unsqueeze_6 = None
    repeat: "i64[1, 1024, 1024]" = torch.ops.aten.repeat.default(slice_5, [1, 1024, 1]);  slice_5 = None
    unsqueeze_7: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
    slice_6: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_7, 1, 0, 9223372036854775807);  unsqueeze_7 = None
    unsqueeze_8: "i64[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_6, 2);  slice_6 = None
    le: "b8[1, 1024, 1024]" = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:864, code: causal_mask = causal_mask.to(attention_mask.dtype)
    convert_element_type_3: "f32[1, 1024, 1024]" = torch.ops.prims.convert_element_type.default(le, torch.float32);  le = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:876, code: extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    slice_7: "f32[1, 1024, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_3, 0, 0, 9223372036854775807);  convert_element_type_3 = None
    unsqueeze_9: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(slice_7, 1);  slice_7 = None
    slice_8: "f32[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_9, 2, 0, 9223372036854775807);  unsqueeze_9 = None
    slice_9: "f32[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 9223372036854775807);  slice_8 = None
    slice_10: "f32[1, 1024]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807);  full_2 = None
    unsqueeze_10: "f32[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_10, 1);  slice_10 = None
    unsqueeze_11: "f32[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
    slice_11: "f32[1, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_11, 3, 0, 9223372036854775807);  unsqueeze_11 = None
    mul_29: "f32[1, 1, 1024, 1024]" = torch.ops.aten.mul.Tensor(slice_9, slice_11);  slice_9 = slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub_8: "f32[1, 1, 1024, 1024]" = torch.ops.aten.sub.Tensor(1.0, mul_29);  mul_29 = None
    mul_30: "f32[1, 1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_8, -3.4028234663852886e+38);  sub_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:840, code: encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    slice_12: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(full_3, 0, 0, 9223372036854775807);  full_3 = None
    unsqueeze_12: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_12, 1);  slice_12 = None
    unsqueeze_13: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
    slice_13: "i64[1, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 0, 9223372036854775807);  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:846, code: encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    convert_element_type_4: "f32[1, 1, 1, 1024]" = torch.ops.prims.convert_element_type.default(slice_13, torch.float32);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:847, code: encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
    sub_9: "f32[1, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(1.0, convert_element_type_4);  convert_element_type_4 = None
    mul_31: "f32[1, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_9, -3.4028234663852886e+38);  sub_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    native_dropout_26 = torch.ops.aten.native_dropout.default(embedding_2, 0.1, True);  embedding_2 = None
    getitem_52: "f32[1, 1024, 512]" = native_dropout_26[0]
    getitem_53: "b8[1, 1024, 512]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_14: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem_52, 2)
    mean_13: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_35: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    alias_25: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_13)
    mul_32: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(getitem_52, rsqrt_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_33: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_14, mul_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    view_146: "f32[1024, 512]" = torch.ops.aten.view.default(mul_33, [1024, 512])
    mm_36: "f32[1024, 512]" = torch.ops.aten.mm.default(view_146, permute_67)
    view_147: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_36, [1, 1024, 512]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_148: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_147, [1, -1, 8, 64]);  view_147 = None
    permute_68: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_69: "f32[512, 512]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    view_149: "f32[1024, 512]" = torch.ops.aten.view.default(mul_33, [1024, 512])
    mm_37: "f32[1024, 512]" = torch.ops.aten.mm.default(view_149, permute_69)
    view_150: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_37, [1, 1024, 512]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_151: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_150, [1, -1, 8, 64]);  view_150 = None
    permute_70: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_71: "f32[512, 512]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    view_152: "f32[1024, 512]" = torch.ops.aten.view.default(mul_33, [1024, 512]);  mul_33 = None
    mm_38: "f32[1024, 512]" = torch.ops.aten.mm.default(view_152, permute_71)
    view_153: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_38, [1, 1024, 512]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_154: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_153, [1, -1, 8, 64]);  view_153 = None
    permute_72: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_73: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_24: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_68, [1, 8, 1024, 64]);  permute_68 = None
    view_155: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_24, [8, 1024, 64]);  expand_24 = None
    expand_25: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_73, [1, 8, 64, 1024]);  permute_73 = None
    view_156: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_25, [8, 64, 1024]);  expand_25 = None
    bmm_12: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_155, view_156)
    view_157: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_12, [1, 8, 1024, 1024]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:441, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    iota_3: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_14: "i64[1024]" = torch.ops.aten.slice.Tensor(iota_3, 0, 0, 9223372036854775807);  iota_3 = None
    unsqueeze_14: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(slice_14, 1);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:442, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    iota_4: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_15: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
    slice_15: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_15, 1, 0, 9223372036854775807);  unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:443, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    sub_10: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(slice_15, unsqueeze_14);  slice_15 = unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:417, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    full_4: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub_10, full_4);  sub_10 = full_4 = None
    neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:422, code: is_small = relative_position < max_exact
    lt_1: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:426, code: torch.log(relative_position.float() / max_exact)
    convert_element_type_5: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
    div_8: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_5, 16);  convert_element_type_5 = None
    log_1: "f32[1024, 1024]" = torch.ops.aten.log.default(div_8);  div_8 = None
    div_9: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
    mul_34: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_9, 16);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:429, code: ).to(torch.long)
    convert_element_type_6: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_34, torch.int64);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:425, code: relative_position_if_large = max_exact + (
    add_36: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_6, 16);  convert_element_type_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:431, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_5: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:430, code: relative_position_if_large = torch.min(
    minimum_2: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_36, full_5);  add_36 = full_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:434, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where_1: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
    add_37: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_1, 0);  where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    embedding_3: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(primals_74, add_37);  primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_74: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
    unsqueeze_16: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_74, 0);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    add_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_16, mul_30);  unsqueeze_16 = mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_39: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_157, add_38);  view_157 = None
    view_158: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_39, [8, 1024, 1024]);  add_39 = None
    view_159: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_158, [1, 8, 1024, 1024]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_6: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_159, [-1], True)
    sub_11: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_159, amax_6);  view_159 = amax_6 = None
    exp_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_7: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_26: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_27 = torch.ops.aten.native_dropout.default(div_10, 0.1, True);  div_10 = None
    getitem_54: "f32[1, 8, 1024, 1024]" = native_dropout_27[0]
    getitem_55: "b8[1, 8, 1024, 1024]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_26: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_54, [1, 8, 1024, 1024]);  getitem_54 = None
    view_160: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_26, [8, 1024, 1024]);  expand_26 = None
    expand_27: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_72, [1, 8, 1024, 64])
    view_161: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_27, [8, 1024, 64]);  expand_27 = None
    bmm_13: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_160, view_161)
    view_162: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_13, [1, 8, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_75: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_6: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    view_163: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_6, [1, -1, 512]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_76: "f32[512, 512]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    view_164: "f32[1024, 512]" = torch.ops.aten.view.default(view_163, [1024, 512]);  view_163 = None
    mm_39: "f32[1024, 512]" = torch.ops.aten.mm.default(view_164, permute_76)
    view_165: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_39, [1, 1024, 512]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_28 = torch.ops.aten.native_dropout.default(view_165, 0.1, True);  view_165 = None
    getitem_56: "f32[1, 1024, 512]" = native_dropout_28[0]
    getitem_57: "b8[1, 1024, 512]" = native_dropout_28[1];  native_dropout_28 = None
    add_40: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(getitem_52, getitem_56);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_15: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_40, 2)
    mean_14: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_41: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    alias_27: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_14)
    mul_35: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_40, rsqrt_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_36: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_15, mul_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_77: "f32[512, 512]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    view_166: "f32[1024, 512]" = torch.ops.aten.view.default(mul_36, [1024, 512]);  mul_36 = None
    mm_40: "f32[1024, 512]" = torch.ops.aten.mm.default(view_166, permute_77)
    view_167: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_40, [1, 1024, 512]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_168: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_167, [1, -1, 8, 64]);  view_167 = None
    permute_78: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_79: "f32[512, 512]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    view_169: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_41: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_79)
    view_170: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_41, [1, 1024, 512]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_171: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_170, [1, -1, 8, 64]);  view_170 = None
    permute_80: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    view_172: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_42: "f32[1024, 512]" = torch.ops.aten.mm.default(view_172, permute_81)
    view_173: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_42, [1, 1024, 512]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_174: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_173, [1, -1, 8, 64]);  view_173 = None
    permute_82: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_83: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_28: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_78, [1, 8, 1024, 64]);  permute_78 = None
    view_175: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_28, [8, 1024, 64]);  expand_28 = None
    expand_29: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_83, [1, 8, 64, 1024]);  permute_83 = None
    view_176: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_29, [8, 64, 1024]);  expand_29 = None
    bmm_14: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_175, view_176)
    view_177: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, [1, 8, 1024, 1024]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: position_bias = torch.zeros(
    full_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.full.default([1, 8, 1024, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    add_42: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(full_6, mul_31);  full_6 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_43: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_177, add_42);  view_177 = None
    view_178: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_43, [8, 1024, 1024]);  add_43 = None
    view_179: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_178, [1, 8, 1024, 1024]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_7: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_179, [-1], True)
    sub_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_179, amax_7);  view_179 = amax_7 = None
    exp_7: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_8: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_11: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_28: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_29 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_58: "f32[1, 8, 1024, 1024]" = native_dropout_29[0]
    getitem_59: "b8[1, 8, 1024, 1024]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_30: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_58, [1, 8, 1024, 1024]);  getitem_58 = None
    view_180: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_30, [8, 1024, 1024]);  expand_30 = None
    expand_31: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_82, [1, 8, 1024, 64])
    view_181: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_31, [8, 1024, 64]);  expand_31 = None
    bmm_15: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_180, view_181)
    view_182: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_15, [1, 8, 1024, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_84: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    clone_7: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_183: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_7, [1, -1, 512]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_85: "f32[512, 512]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    view_184: "f32[1024, 512]" = torch.ops.aten.view.default(view_183, [1024, 512]);  view_183 = None
    mm_43: "f32[1024, 512]" = torch.ops.aten.mm.default(view_184, permute_85)
    view_185: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_43, [1, 1024, 512]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_185, 0.1, True);  view_185 = None
    getitem_60: "f32[1, 1024, 512]" = native_dropout_30[0]
    getitem_61: "b8[1, 1024, 512]" = native_dropout_30[1];  native_dropout_30 = None
    add_44: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_40, getitem_60);  getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_16: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 2)
    mean_15: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_45: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    alias_29: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_15)
    mul_37: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_38: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_16, mul_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_86: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    view_186: "f32[1024, 512]" = torch.ops.aten.view.default(mul_38, [1024, 512]);  mul_38 = None
    mm_44: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_186, permute_86)
    view_187: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_44, [1, 1024, 2048]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_6: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_187);  view_187 = None
    alias_30: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_31 = torch.ops.aten.native_dropout.default(relu_6, 0.1, True);  relu_6 = None
    getitem_62: "f32[1, 1024, 2048]" = native_dropout_31[0]
    getitem_63: "b8[1, 1024, 2048]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_87: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    view_188: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_62, [1024, 2048]);  getitem_62 = None
    mm_45: "f32[1024, 512]" = torch.ops.aten.mm.default(view_188, permute_87)
    view_189: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_45, [1, 1024, 512]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_189, 0.1, True);  view_189 = None
    getitem_64: "f32[1, 1024, 512]" = native_dropout_32[0]
    getitem_65: "b8[1, 1024, 512]" = native_dropout_32[1];  native_dropout_32 = None
    add_46: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_44, getitem_64);  getitem_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_17: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 2)
    mean_16: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_47: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    alias_31: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_16)
    mul_39: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_40: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_17, mul_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    view_190: "f32[1024, 512]" = torch.ops.aten.view.default(mul_40, [1024, 512])
    mm_46: "f32[1024, 512]" = torch.ops.aten.mm.default(view_190, permute_88)
    view_191: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_46, [1, 1024, 512]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_192: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_191, [1, -1, 8, 64]);  view_191 = None
    permute_89: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_90: "f32[512, 512]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    view_193: "f32[1024, 512]" = torch.ops.aten.view.default(mul_40, [1024, 512])
    mm_47: "f32[1024, 512]" = torch.ops.aten.mm.default(view_193, permute_90)
    view_194: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_47, [1, 1024, 512]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_195: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_194, [1, -1, 8, 64]);  view_194 = None
    permute_91: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_92: "f32[512, 512]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    view_196: "f32[1024, 512]" = torch.ops.aten.view.default(mul_40, [1024, 512]);  mul_40 = None
    mm_48: "f32[1024, 512]" = torch.ops.aten.mm.default(view_196, permute_92)
    view_197: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_48, [1, 1024, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_198: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_197, [1, -1, 8, 64]);  view_197 = None
    permute_93: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_94: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_32: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_89, [1, 8, 1024, 64]);  permute_89 = None
    view_199: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_32, [8, 1024, 64]);  expand_32 = None
    expand_33: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_94, [1, 8, 64, 1024]);  permute_94 = None
    view_200: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_33, [8, 64, 1024]);  expand_33 = None
    bmm_16: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_199, view_200)
    view_201: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_16, [1, 8, 1024, 1024]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_48: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_201, add_38);  view_201 = None
    view_202: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_48, [8, 1024, 1024]);  add_48 = None
    view_203: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_202, [1, 8, 1024, 1024]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_8: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_203, [-1], True)
    sub_13: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_203, amax_8);  view_203 = amax_8 = None
    exp_8: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_9: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_32: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_33 = torch.ops.aten.native_dropout.default(div_12, 0.1, True);  div_12 = None
    getitem_66: "f32[1, 8, 1024, 1024]" = native_dropout_33[0]
    getitem_67: "b8[1, 8, 1024, 1024]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_34: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_66, [1, 8, 1024, 1024]);  getitem_66 = None
    view_204: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_34, [8, 1024, 1024]);  expand_34 = None
    expand_35: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_93, [1, 8, 1024, 64])
    view_205: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_35, [8, 1024, 64]);  expand_35 = None
    bmm_17: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_204, view_205)
    view_206: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_17, [1, 8, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_95: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    clone_8: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_207: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_8, [1, -1, 512]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_96: "f32[512, 512]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    view_208: "f32[1024, 512]" = torch.ops.aten.view.default(view_207, [1024, 512]);  view_207 = None
    mm_49: "f32[1024, 512]" = torch.ops.aten.mm.default(view_208, permute_96)
    view_209: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_49, [1, 1024, 512]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_34 = torch.ops.aten.native_dropout.default(view_209, 0.1, True);  view_209 = None
    getitem_68: "f32[1, 1024, 512]" = native_dropout_34[0]
    getitem_69: "b8[1, 1024, 512]" = native_dropout_34[1];  native_dropout_34 = None
    add_49: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_46, getitem_68);  getitem_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_18: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 2)
    mean_17: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_50: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    alias_33: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_17)
    mul_41: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_42: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_18, mul_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_97: "f32[512, 512]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    view_210: "f32[1024, 512]" = torch.ops.aten.view.default(mul_42, [1024, 512]);  mul_42 = None
    mm_50: "f32[1024, 512]" = torch.ops.aten.mm.default(view_210, permute_97)
    view_211: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_50, [1, 1024, 512]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_212: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_211, [1, -1, 8, 64]);  view_211 = None
    permute_98: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_99: "f32[512, 512]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    view_213: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_51: "f32[1024, 512]" = torch.ops.aten.mm.default(view_213, permute_99)
    view_214: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_51, [1, 1024, 512]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_215: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_214, [1, -1, 8, 64]);  view_214 = None
    permute_100: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_101: "f32[512, 512]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    view_216: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_52: "f32[1024, 512]" = torch.ops.aten.mm.default(view_216, permute_101)
    view_217: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_52, [1, 1024, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_218: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_217, [1, -1, 8, 64]);  view_217 = None
    permute_102: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_103: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_36: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_98, [1, 8, 1024, 64]);  permute_98 = None
    view_219: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_36, [8, 1024, 64]);  expand_36 = None
    expand_37: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_103, [1, 8, 64, 1024]);  permute_103 = None
    view_220: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_37, [8, 64, 1024]);  expand_37 = None
    bmm_18: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_219, view_220)
    view_221: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, [1, 8, 1024, 1024]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_51: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_221, add_42);  view_221 = None
    view_222: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_51, [8, 1024, 1024]);  add_51 = None
    view_223: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_222, [1, 8, 1024, 1024]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_9: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_223, [-1], True)
    sub_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_223, amax_9);  view_223 = amax_9 = None
    exp_9: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_10: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_13: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_34: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_35 = torch.ops.aten.native_dropout.default(div_13, 0.1, True);  div_13 = None
    getitem_70: "f32[1, 8, 1024, 1024]" = native_dropout_35[0]
    getitem_71: "b8[1, 8, 1024, 1024]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_70, [1, 8, 1024, 1024]);  getitem_70 = None
    view_224: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_38, [8, 1024, 1024]);  expand_38 = None
    expand_39: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_102, [1, 8, 1024, 64])
    view_225: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_39, [8, 1024, 64]);  expand_39 = None
    bmm_19: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_224, view_225)
    view_226: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_19, [1, 8, 1024, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_104: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_9: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_227: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_9, [1, -1, 512]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_105: "f32[512, 512]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    view_228: "f32[1024, 512]" = torch.ops.aten.view.default(view_227, [1024, 512]);  view_227 = None
    mm_53: "f32[1024, 512]" = torch.ops.aten.mm.default(view_228, permute_105)
    view_229: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_53, [1, 1024, 512]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_229, 0.1, True);  view_229 = None
    getitem_72: "f32[1, 1024, 512]" = native_dropout_36[0]
    getitem_73: "b8[1, 1024, 512]" = native_dropout_36[1];  native_dropout_36 = None
    add_52: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_49, getitem_72);  getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_19: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 2)
    mean_18: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_53: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    alias_35: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_18)
    mul_43: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_44: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_19, mul_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_106: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    view_230: "f32[1024, 512]" = torch.ops.aten.view.default(mul_44, [1024, 512]);  mul_44 = None
    mm_54: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_230, permute_106)
    view_231: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_54, [1, 1024, 2048]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_7: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_231);  view_231 = None
    alias_36: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_37 = torch.ops.aten.native_dropout.default(relu_7, 0.1, True);  relu_7 = None
    getitem_74: "f32[1, 1024, 2048]" = native_dropout_37[0]
    getitem_75: "b8[1, 1024, 2048]" = native_dropout_37[1];  native_dropout_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_107: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    view_232: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_74, [1024, 2048]);  getitem_74 = None
    mm_55: "f32[1024, 512]" = torch.ops.aten.mm.default(view_232, permute_107)
    view_233: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_55, [1, 1024, 512]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_38 = torch.ops.aten.native_dropout.default(view_233, 0.1, True);  view_233 = None
    getitem_76: "f32[1, 1024, 512]" = native_dropout_38[0]
    getitem_77: "b8[1, 1024, 512]" = native_dropout_38[1];  native_dropout_38 = None
    add_54: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_52, getitem_76);  getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_20: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 2)
    mean_19: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_55: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    alias_37: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_19)
    mul_45: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_46: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_20, mul_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_108: "f32[512, 512]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    view_234: "f32[1024, 512]" = torch.ops.aten.view.default(mul_46, [1024, 512])
    mm_56: "f32[1024, 512]" = torch.ops.aten.mm.default(view_234, permute_108)
    view_235: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_56, [1, 1024, 512]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_236: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_235, [1, -1, 8, 64]);  view_235 = None
    permute_109: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_110: "f32[512, 512]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    view_237: "f32[1024, 512]" = torch.ops.aten.view.default(mul_46, [1024, 512])
    mm_57: "f32[1024, 512]" = torch.ops.aten.mm.default(view_237, permute_110)
    view_238: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_57, [1, 1024, 512]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_239: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_238, [1, -1, 8, 64]);  view_238 = None
    permute_111: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_112: "f32[512, 512]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    view_240: "f32[1024, 512]" = torch.ops.aten.view.default(mul_46, [1024, 512]);  mul_46 = None
    mm_58: "f32[1024, 512]" = torch.ops.aten.mm.default(view_240, permute_112)
    view_241: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_58, [1, 1024, 512]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_242: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_241, [1, -1, 8, 64]);  view_241 = None
    permute_113: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_114: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_40: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_109, [1, 8, 1024, 64]);  permute_109 = None
    view_243: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_40, [8, 1024, 64]);  expand_40 = None
    expand_41: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_114, [1, 8, 64, 1024]);  permute_114 = None
    view_244: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_41, [8, 64, 1024]);  expand_41 = None
    bmm_20: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_243, view_244)
    view_245: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_20, [1, 8, 1024, 1024]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_56: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_245, add_38);  view_245 = None
    view_246: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_56, [8, 1024, 1024]);  add_56 = None
    view_247: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_246, [1, 8, 1024, 1024]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_10: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_247, [-1], True)
    sub_15: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_247, amax_10);  view_247 = amax_10 = None
    exp_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_11: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_39 = torch.ops.aten.native_dropout.default(div_14, 0.1, True);  div_14 = None
    getitem_78: "f32[1, 8, 1024, 1024]" = native_dropout_39[0]
    getitem_79: "b8[1, 8, 1024, 1024]" = native_dropout_39[1];  native_dropout_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_42: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_78, [1, 8, 1024, 1024]);  getitem_78 = None
    view_248: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_42, [8, 1024, 1024]);  expand_42 = None
    expand_43: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_113, [1, 8, 1024, 64])
    view_249: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_43, [8, 1024, 64]);  expand_43 = None
    bmm_21: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_248, view_249)
    view_250: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_21, [1, 8, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_115: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    clone_10: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_251: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_10, [1, -1, 512]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    view_252: "f32[1024, 512]" = torch.ops.aten.view.default(view_251, [1024, 512]);  view_251 = None
    mm_59: "f32[1024, 512]" = torch.ops.aten.mm.default(view_252, permute_116)
    view_253: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_59, [1, 1024, 512]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_40 = torch.ops.aten.native_dropout.default(view_253, 0.1, True);  view_253 = None
    getitem_80: "f32[1, 1024, 512]" = native_dropout_40[0]
    getitem_81: "b8[1, 1024, 512]" = native_dropout_40[1];  native_dropout_40 = None
    add_57: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_54, getitem_80);  getitem_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_21: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_57, 2)
    mean_20: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_58: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    alias_39: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_20)
    mul_47: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_57, rsqrt_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_48: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_21, mul_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_117: "f32[512, 512]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    view_254: "f32[1024, 512]" = torch.ops.aten.view.default(mul_48, [1024, 512]);  mul_48 = None
    mm_60: "f32[1024, 512]" = torch.ops.aten.mm.default(view_254, permute_117)
    view_255: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_60, [1, 1024, 512]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_256: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_255, [1, -1, 8, 64]);  view_255 = None
    permute_118: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_119: "f32[512, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    view_257: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_61: "f32[1024, 512]" = torch.ops.aten.mm.default(view_257, permute_119)
    view_258: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_61, [1, 1024, 512]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_259: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_258, [1, -1, 8, 64]);  view_258 = None
    permute_120: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_121: "f32[512, 512]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    view_260: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_62: "f32[1024, 512]" = torch.ops.aten.mm.default(view_260, permute_121)
    view_261: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_62, [1, 1024, 512]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_262: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_261, [1, -1, 8, 64]);  view_261 = None
    permute_122: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_123: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_44: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_118, [1, 8, 1024, 64]);  permute_118 = None
    view_263: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_44, [8, 1024, 64]);  expand_44 = None
    expand_45: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_123, [1, 8, 64, 1024]);  permute_123 = None
    view_264: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_45, [8, 64, 1024]);  expand_45 = None
    bmm_22: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_263, view_264)
    view_265: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, [1, 8, 1024, 1024]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_59: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_265, add_42);  view_265 = None
    view_266: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_59, [8, 1024, 1024]);  add_59 = None
    view_267: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_266, [1, 8, 1024, 1024]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_11: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_267, [-1], True)
    sub_16: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_267, amax_11);  view_267 = amax_11 = None
    exp_11: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_12: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_15: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_40: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_41 = torch.ops.aten.native_dropout.default(div_15, 0.1, True);  div_15 = None
    getitem_82: "f32[1, 8, 1024, 1024]" = native_dropout_41[0]
    getitem_83: "b8[1, 8, 1024, 1024]" = native_dropout_41[1];  native_dropout_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_46: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_82, [1, 8, 1024, 1024]);  getitem_82 = None
    view_268: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_46, [8, 1024, 1024]);  expand_46 = None
    expand_47: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_122, [1, 8, 1024, 64])
    view_269: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_47, [8, 1024, 64]);  expand_47 = None
    bmm_23: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_268, view_269)
    view_270: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_23, [1, 8, 1024, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_124: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
    clone_11: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    view_271: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_11, [1, -1, 512]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_125: "f32[512, 512]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    view_272: "f32[1024, 512]" = torch.ops.aten.view.default(view_271, [1024, 512]);  view_271 = None
    mm_63: "f32[1024, 512]" = torch.ops.aten.mm.default(view_272, permute_125)
    view_273: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_63, [1, 1024, 512]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_42 = torch.ops.aten.native_dropout.default(view_273, 0.1, True);  view_273 = None
    getitem_84: "f32[1, 1024, 512]" = native_dropout_42[0]
    getitem_85: "b8[1, 1024, 512]" = native_dropout_42[1];  native_dropout_42 = None
    add_60: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_57, getitem_84);  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_22: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 2)
    mean_21: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_61: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    alias_41: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_21)
    mul_49: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_50: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_22, mul_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_126: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    view_274: "f32[1024, 512]" = torch.ops.aten.view.default(mul_50, [1024, 512]);  mul_50 = None
    mm_64: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_274, permute_126)
    view_275: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_64, [1, 1024, 2048]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_8: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_275);  view_275 = None
    alias_42: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_43 = torch.ops.aten.native_dropout.default(relu_8, 0.1, True);  relu_8 = None
    getitem_86: "f32[1, 1024, 2048]" = native_dropout_43[0]
    getitem_87: "b8[1, 1024, 2048]" = native_dropout_43[1];  native_dropout_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_127: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    view_276: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_86, [1024, 2048]);  getitem_86 = None
    mm_65: "f32[1024, 512]" = torch.ops.aten.mm.default(view_276, permute_127)
    view_277: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_65, [1, 1024, 512]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_44 = torch.ops.aten.native_dropout.default(view_277, 0.1, True);  view_277 = None
    getitem_88: "f32[1, 1024, 512]" = native_dropout_44[0]
    getitem_89: "b8[1, 1024, 512]" = native_dropout_44[1];  native_dropout_44 = None
    add_62: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_60, getitem_88);  getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_23: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
    mean_22: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_63: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    alias_43: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_22)
    mul_51: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_52: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_23, mul_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_128: "f32[512, 512]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    view_278: "f32[1024, 512]" = torch.ops.aten.view.default(mul_52, [1024, 512])
    mm_66: "f32[1024, 512]" = torch.ops.aten.mm.default(view_278, permute_128)
    view_279: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_66, [1, 1024, 512]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_280: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_279, [1, -1, 8, 64]);  view_279 = None
    permute_129: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    view_281: "f32[1024, 512]" = torch.ops.aten.view.default(mul_52, [1024, 512])
    mm_67: "f32[1024, 512]" = torch.ops.aten.mm.default(view_281, permute_130)
    view_282: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_67, [1, 1024, 512]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_283: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_282, [1, -1, 8, 64]);  view_282 = None
    permute_131: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_132: "f32[512, 512]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    view_284: "f32[1024, 512]" = torch.ops.aten.view.default(mul_52, [1024, 512]);  mul_52 = None
    mm_68: "f32[1024, 512]" = torch.ops.aten.mm.default(view_284, permute_132)
    view_285: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_68, [1, 1024, 512]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_286: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_285, [1, -1, 8, 64]);  view_285 = None
    permute_133: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_134: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_48: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_129, [1, 8, 1024, 64]);  permute_129 = None
    view_287: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_48, [8, 1024, 64]);  expand_48 = None
    expand_49: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_134, [1, 8, 64, 1024]);  permute_134 = None
    view_288: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_49, [8, 64, 1024]);  expand_49 = None
    bmm_24: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_287, view_288)
    view_289: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_24, [1, 8, 1024, 1024]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_64: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_289, add_38);  view_289 = None
    view_290: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_64, [8, 1024, 1024]);  add_64 = None
    view_291: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_290, [1, 8, 1024, 1024]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_12: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_291, [-1], True)
    sub_17: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_291, amax_12);  view_291 = amax_12 = None
    exp_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_13: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_16: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_44: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_45 = torch.ops.aten.native_dropout.default(div_16, 0.1, True);  div_16 = None
    getitem_90: "f32[1, 8, 1024, 1024]" = native_dropout_45[0]
    getitem_91: "b8[1, 8, 1024, 1024]" = native_dropout_45[1];  native_dropout_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_50: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_90, [1, 8, 1024, 1024]);  getitem_90 = None
    view_292: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_50, [8, 1024, 1024]);  expand_50 = None
    expand_51: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_133, [1, 8, 1024, 64])
    view_293: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_51, [8, 1024, 64]);  expand_51 = None
    bmm_25: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_292, view_293)
    view_294: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_25, [1, 8, 1024, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_135: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    clone_12: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    view_295: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_12, [1, -1, 512]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_136: "f32[512, 512]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    view_296: "f32[1024, 512]" = torch.ops.aten.view.default(view_295, [1024, 512]);  view_295 = None
    mm_69: "f32[1024, 512]" = torch.ops.aten.mm.default(view_296, permute_136)
    view_297: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_69, [1, 1024, 512]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_46 = torch.ops.aten.native_dropout.default(view_297, 0.1, True);  view_297 = None
    getitem_92: "f32[1, 1024, 512]" = native_dropout_46[0]
    getitem_93: "b8[1, 1024, 512]" = native_dropout_46[1];  native_dropout_46 = None
    add_65: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_62, getitem_92);  getitem_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_24: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_65, 2)
    mean_23: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_66: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    alias_45: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_23)
    mul_53: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_54: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_24, mul_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_137: "f32[512, 512]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    view_298: "f32[1024, 512]" = torch.ops.aten.view.default(mul_54, [1024, 512]);  mul_54 = None
    mm_70: "f32[1024, 512]" = torch.ops.aten.mm.default(view_298, permute_137)
    view_299: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_70, [1, 1024, 512]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_300: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_299, [1, -1, 8, 64]);  view_299 = None
    permute_138: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_139: "f32[512, 512]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    view_301: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_71: "f32[1024, 512]" = torch.ops.aten.mm.default(view_301, permute_139)
    view_302: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_71, [1, 1024, 512]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_303: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_302, [1, -1, 8, 64]);  view_302 = None
    permute_140: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_141: "f32[512, 512]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    view_304: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_72: "f32[1024, 512]" = torch.ops.aten.mm.default(view_304, permute_141)
    view_305: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_72, [1, 1024, 512]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_306: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_305, [1, -1, 8, 64]);  view_305 = None
    permute_142: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_143: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_52: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_138, [1, 8, 1024, 64]);  permute_138 = None
    view_307: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_52, [8, 1024, 64]);  expand_52 = None
    expand_53: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_143, [1, 8, 64, 1024]);  permute_143 = None
    view_308: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_53, [8, 64, 1024]);  expand_53 = None
    bmm_26: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_307, view_308)
    view_309: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_26, [1, 8, 1024, 1024]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_67: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_309, add_42);  view_309 = None
    view_310: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_67, [8, 1024, 1024]);  add_67 = None
    view_311: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_310, [1, 8, 1024, 1024]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_13: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_311, [-1], True)
    sub_18: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_311, amax_13);  view_311 = amax_13 = None
    exp_13: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_14: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_17: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_46: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_47 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_94: "f32[1, 8, 1024, 1024]" = native_dropout_47[0]
    getitem_95: "b8[1, 8, 1024, 1024]" = native_dropout_47[1];  native_dropout_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_54: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_94, [1, 8, 1024, 1024]);  getitem_94 = None
    view_312: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_54, [8, 1024, 1024]);  expand_54 = None
    expand_55: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_142, [1, 8, 1024, 64])
    view_313: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_55, [8, 1024, 64]);  expand_55 = None
    bmm_27: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_312, view_313)
    view_314: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_27, [1, 8, 1024, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_144: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    clone_13: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    view_315: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_13, [1, -1, 512]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_145: "f32[512, 512]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    view_316: "f32[1024, 512]" = torch.ops.aten.view.default(view_315, [1024, 512]);  view_315 = None
    mm_73: "f32[1024, 512]" = torch.ops.aten.mm.default(view_316, permute_145)
    view_317: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_73, [1, 1024, 512]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_48 = torch.ops.aten.native_dropout.default(view_317, 0.1, True);  view_317 = None
    getitem_96: "f32[1, 1024, 512]" = native_dropout_48[0]
    getitem_97: "b8[1, 1024, 512]" = native_dropout_48[1];  native_dropout_48 = None
    add_68: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_65, getitem_96);  getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_25: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_68, 2)
    mean_24: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_69: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
    rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    alias_47: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_24)
    mul_55: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_68, rsqrt_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_56: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_25, mul_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_146: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    view_318: "f32[1024, 512]" = torch.ops.aten.view.default(mul_56, [1024, 512]);  mul_56 = None
    mm_74: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_318, permute_146)
    view_319: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_74, [1, 1024, 2048]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_9: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_319);  view_319 = None
    alias_48: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_49 = torch.ops.aten.native_dropout.default(relu_9, 0.1, True);  relu_9 = None
    getitem_98: "f32[1, 1024, 2048]" = native_dropout_49[0]
    getitem_99: "b8[1, 1024, 2048]" = native_dropout_49[1];  native_dropout_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_147: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    view_320: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_98, [1024, 2048]);  getitem_98 = None
    mm_75: "f32[1024, 512]" = torch.ops.aten.mm.default(view_320, permute_147)
    view_321: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_75, [1, 1024, 512]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_50 = torch.ops.aten.native_dropout.default(view_321, 0.1, True);  view_321 = None
    getitem_100: "f32[1, 1024, 512]" = native_dropout_50[0]
    getitem_101: "b8[1, 1024, 512]" = native_dropout_50[1];  native_dropout_50 = None
    add_70: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_68, getitem_100);  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_26: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
    mean_25: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_71: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
    rsqrt_25: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    alias_49: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_25)
    mul_57: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_58: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_26, mul_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_148: "f32[512, 512]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    view_322: "f32[1024, 512]" = torch.ops.aten.view.default(mul_58, [1024, 512])
    mm_76: "f32[1024, 512]" = torch.ops.aten.mm.default(view_322, permute_148)
    view_323: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_76, [1, 1024, 512]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_324: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_323, [1, -1, 8, 64]);  view_323 = None
    permute_149: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_150: "f32[512, 512]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    view_325: "f32[1024, 512]" = torch.ops.aten.view.default(mul_58, [1024, 512])
    mm_77: "f32[1024, 512]" = torch.ops.aten.mm.default(view_325, permute_150)
    view_326: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_77, [1, 1024, 512]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_327: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_326, [1, -1, 8, 64]);  view_326 = None
    permute_151: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_152: "f32[512, 512]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_328: "f32[1024, 512]" = torch.ops.aten.view.default(mul_58, [1024, 512]);  mul_58 = None
    mm_78: "f32[1024, 512]" = torch.ops.aten.mm.default(view_328, permute_152)
    view_329: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_78, [1, 1024, 512]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_330: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_329, [1, -1, 8, 64]);  view_329 = None
    permute_153: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_154: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_56: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_149, [1, 8, 1024, 64]);  permute_149 = None
    view_331: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_56, [8, 1024, 64]);  expand_56 = None
    expand_57: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_154, [1, 8, 64, 1024]);  permute_154 = None
    view_332: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_57, [8, 64, 1024]);  expand_57 = None
    bmm_28: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_331, view_332)
    view_333: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_28, [1, 8, 1024, 1024]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_72: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_333, add_38);  view_333 = None
    view_334: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_72, [8, 1024, 1024]);  add_72 = None
    view_335: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_334, [1, 8, 1024, 1024]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_14: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_335, [-1], True)
    sub_19: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_335, amax_14);  view_335 = amax_14 = None
    exp_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_15: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_18: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_50: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_51 = torch.ops.aten.native_dropout.default(div_18, 0.1, True);  div_18 = None
    getitem_102: "f32[1, 8, 1024, 1024]" = native_dropout_51[0]
    getitem_103: "b8[1, 8, 1024, 1024]" = native_dropout_51[1];  native_dropout_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_58: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_102, [1, 8, 1024, 1024]);  getitem_102 = None
    view_336: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_58, [8, 1024, 1024]);  expand_58 = None
    expand_59: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_153, [1, 8, 1024, 64])
    view_337: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_59, [8, 1024, 64]);  expand_59 = None
    bmm_29: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_336, view_337)
    view_338: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_29, [1, 8, 1024, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_155: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    clone_14: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    view_339: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_14, [1, -1, 512]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_156: "f32[512, 512]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    view_340: "f32[1024, 512]" = torch.ops.aten.view.default(view_339, [1024, 512]);  view_339 = None
    mm_79: "f32[1024, 512]" = torch.ops.aten.mm.default(view_340, permute_156)
    view_341: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_79, [1, 1024, 512]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_52 = torch.ops.aten.native_dropout.default(view_341, 0.1, True);  view_341 = None
    getitem_104: "f32[1, 1024, 512]" = native_dropout_52[0]
    getitem_105: "b8[1, 1024, 512]" = native_dropout_52[1];  native_dropout_52 = None
    add_73: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_70, getitem_104);  getitem_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_27: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_73, 2)
    mean_26: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_74: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
    rsqrt_26: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    alias_51: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_26)
    mul_59: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_73, rsqrt_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_60: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_27, mul_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_157: "f32[512, 512]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    view_342: "f32[1024, 512]" = torch.ops.aten.view.default(mul_60, [1024, 512]);  mul_60 = None
    mm_80: "f32[1024, 512]" = torch.ops.aten.mm.default(view_342, permute_157)
    view_343: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_80, [1, 1024, 512]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_344: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_343, [1, -1, 8, 64]);  view_343 = None
    permute_158: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_159: "f32[512, 512]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    view_345: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_81: "f32[1024, 512]" = torch.ops.aten.mm.default(view_345, permute_159)
    view_346: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_81, [1, 1024, 512]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_347: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_346, [1, -1, 8, 64]);  view_346 = None
    permute_160: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_161: "f32[512, 512]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    view_348: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_82: "f32[1024, 512]" = torch.ops.aten.mm.default(view_348, permute_161)
    view_349: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_82, [1, 1024, 512]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_350: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_349, [1, -1, 8, 64]);  view_349 = None
    permute_162: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_163: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_60: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_158, [1, 8, 1024, 64]);  permute_158 = None
    view_351: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_60, [8, 1024, 64]);  expand_60 = None
    expand_61: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_163, [1, 8, 64, 1024]);  permute_163 = None
    view_352: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_61, [8, 64, 1024]);  expand_61 = None
    bmm_30: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_351, view_352)
    view_353: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_30, [1, 8, 1024, 1024]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_75: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_353, add_42);  view_353 = None
    view_354: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_75, [8, 1024, 1024]);  add_75 = None
    view_355: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_354, [1, 8, 1024, 1024]);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_15: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_355, [-1], True)
    sub_20: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_355, amax_15);  view_355 = amax_15 = None
    exp_15: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_16: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_19: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_52: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_53 = torch.ops.aten.native_dropout.default(div_19, 0.1, True);  div_19 = None
    getitem_106: "f32[1, 8, 1024, 1024]" = native_dropout_53[0]
    getitem_107: "b8[1, 8, 1024, 1024]" = native_dropout_53[1];  native_dropout_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_62: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_106, [1, 8, 1024, 1024]);  getitem_106 = None
    view_356: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_62, [8, 1024, 1024]);  expand_62 = None
    expand_63: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_162, [1, 8, 1024, 64])
    view_357: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_63, [8, 1024, 64]);  expand_63 = None
    bmm_31: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_356, view_357)
    view_358: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_31, [1, 8, 1024, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_164: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
    clone_15: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_359: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_15, [1, -1, 512]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_165: "f32[512, 512]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    view_360: "f32[1024, 512]" = torch.ops.aten.view.default(view_359, [1024, 512]);  view_359 = None
    mm_83: "f32[1024, 512]" = torch.ops.aten.mm.default(view_360, permute_165)
    view_361: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_83, [1, 1024, 512]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_54 = torch.ops.aten.native_dropout.default(view_361, 0.1, True);  view_361 = None
    getitem_108: "f32[1, 1024, 512]" = native_dropout_54[0]
    getitem_109: "b8[1, 1024, 512]" = native_dropout_54[1];  native_dropout_54 = None
    add_76: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_73, getitem_108);  getitem_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_28: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_76, 2)
    mean_27: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_77: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
    rsqrt_27: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    alias_53: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_27)
    mul_61: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_62: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_28, mul_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_166: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    view_362: "f32[1024, 512]" = torch.ops.aten.view.default(mul_62, [1024, 512]);  mul_62 = None
    mm_84: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_362, permute_166)
    view_363: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_84, [1, 1024, 2048]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_10: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_363);  view_363 = None
    alias_54: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_55 = torch.ops.aten.native_dropout.default(relu_10, 0.1, True);  relu_10 = None
    getitem_110: "f32[1, 1024, 2048]" = native_dropout_55[0]
    getitem_111: "b8[1, 1024, 2048]" = native_dropout_55[1];  native_dropout_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_167: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    view_364: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_110, [1024, 2048]);  getitem_110 = None
    mm_85: "f32[1024, 512]" = torch.ops.aten.mm.default(view_364, permute_167)
    view_365: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_85, [1, 1024, 512]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_56 = torch.ops.aten.native_dropout.default(view_365, 0.1, True);  view_365 = None
    getitem_112: "f32[1, 1024, 512]" = native_dropout_56[0]
    getitem_113: "b8[1, 1024, 512]" = native_dropout_56[1];  native_dropout_56 = None
    add_78: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_76, getitem_112);  getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_29: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
    mean_28: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_79: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
    rsqrt_28: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    alias_55: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_28)
    mul_63: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_64: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_29, mul_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_168: "f32[512, 512]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    view_366: "f32[1024, 512]" = torch.ops.aten.view.default(mul_64, [1024, 512])
    mm_86: "f32[1024, 512]" = torch.ops.aten.mm.default(view_366, permute_168)
    view_367: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_86, [1, 1024, 512]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_368: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_367, [1, -1, 8, 64]);  view_367 = None
    permute_169: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_170: "f32[512, 512]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    view_369: "f32[1024, 512]" = torch.ops.aten.view.default(mul_64, [1024, 512])
    mm_87: "f32[1024, 512]" = torch.ops.aten.mm.default(view_369, permute_170)
    view_370: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_87, [1, 1024, 512]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_371: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_370, [1, -1, 8, 64]);  view_370 = None
    permute_171: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_172: "f32[512, 512]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    view_372: "f32[1024, 512]" = torch.ops.aten.view.default(mul_64, [1024, 512]);  mul_64 = None
    mm_88: "f32[1024, 512]" = torch.ops.aten.mm.default(view_372, permute_172)
    view_373: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_88, [1, 1024, 512]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_374: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_373, [1, -1, 8, 64]);  view_373 = None
    permute_173: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_174: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_64: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_169, [1, 8, 1024, 64]);  permute_169 = None
    view_375: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_64, [8, 1024, 64]);  expand_64 = None
    expand_65: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_174, [1, 8, 64, 1024]);  permute_174 = None
    view_376: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_65, [8, 64, 1024]);  expand_65 = None
    bmm_32: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_375, view_376)
    view_377: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_32, [1, 8, 1024, 1024]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_80: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_377, add_38);  view_377 = add_38 = None
    view_378: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_80, [8, 1024, 1024]);  add_80 = None
    view_379: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_378, [1, 8, 1024, 1024]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_16: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_379, [-1], True)
    sub_21: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_379, amax_16);  view_379 = amax_16 = None
    exp_16: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_17: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_20: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_56: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_57 = torch.ops.aten.native_dropout.default(div_20, 0.1, True);  div_20 = None
    getitem_114: "f32[1, 8, 1024, 1024]" = native_dropout_57[0]
    getitem_115: "b8[1, 8, 1024, 1024]" = native_dropout_57[1];  native_dropout_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_66: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_114, [1, 8, 1024, 1024]);  getitem_114 = None
    view_380: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_66, [8, 1024, 1024]);  expand_66 = None
    expand_67: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_173, [1, 8, 1024, 64])
    view_381: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_67, [8, 1024, 64]);  expand_67 = None
    bmm_33: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_380, view_381)
    view_382: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_33, [1, 8, 1024, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_175: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
    clone_16: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    view_383: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_16, [1, -1, 512]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_176: "f32[512, 512]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    view_384: "f32[1024, 512]" = torch.ops.aten.view.default(view_383, [1024, 512]);  view_383 = None
    mm_89: "f32[1024, 512]" = torch.ops.aten.mm.default(view_384, permute_176)
    view_385: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_89, [1, 1024, 512]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_58 = torch.ops.aten.native_dropout.default(view_385, 0.1, True);  view_385 = None
    getitem_116: "f32[1, 1024, 512]" = native_dropout_58[0]
    getitem_117: "b8[1, 1024, 512]" = native_dropout_58[1];  native_dropout_58 = None
    add_81: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_78, getitem_116);  getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_30: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_81, 2)
    mean_29: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_82: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
    rsqrt_29: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    alias_57: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_29)
    mul_65: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_66: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_30, mul_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_177: "f32[512, 512]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    view_386: "f32[1024, 512]" = torch.ops.aten.view.default(mul_66, [1024, 512]);  mul_66 = None
    mm_90: "f32[1024, 512]" = torch.ops.aten.mm.default(view_386, permute_177)
    view_387: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_90, [1, 1024, 512]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_388: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_387, [1, -1, 8, 64]);  view_387 = None
    permute_178: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_179: "f32[512, 512]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    view_389: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_91: "f32[1024, 512]" = torch.ops.aten.mm.default(view_389, permute_179)
    view_390: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_91, [1, 1024, 512]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_391: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_390, [1, -1, 8, 64]);  view_390 = None
    permute_180: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_181: "f32[512, 512]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    view_392: "f32[1024, 512]" = torch.ops.aten.view.default(getitem_50, [1024, 512])
    mm_92: "f32[1024, 512]" = torch.ops.aten.mm.default(view_392, permute_181)
    view_393: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_92, [1, 1024, 512]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_394: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_393, [1, -1, 8, 64]);  view_393 = None
    permute_182: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_183: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_68: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_178, [1, 8, 1024, 64]);  permute_178 = None
    view_395: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_68, [8, 1024, 64]);  expand_68 = None
    expand_69: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_183, [1, 8, 64, 1024]);  permute_183 = None
    view_396: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_69, [8, 64, 1024]);  expand_69 = None
    bmm_34: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_395, view_396)
    view_397: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_34, [1, 8, 1024, 1024]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_83: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_397, add_42);  view_397 = add_42 = None
    view_398: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_83, [8, 1024, 1024]);  add_83 = None
    view_399: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_398, [1, 8, 1024, 1024]);  view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_17: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_399, [-1], True)
    sub_22: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_399, amax_17);  view_399 = amax_17 = None
    exp_17: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_18: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_21: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_58: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    native_dropout_59 = torch.ops.aten.native_dropout.default(div_21, 0.1, True);  div_21 = None
    getitem_118: "f32[1, 8, 1024, 1024]" = native_dropout_59[0]
    getitem_119: "b8[1, 8, 1024, 1024]" = native_dropout_59[1];  native_dropout_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_70: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(getitem_118, [1, 8, 1024, 1024]);  getitem_118 = None
    view_400: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_70, [8, 1024, 1024]);  expand_70 = None
    expand_71: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_182, [1, 8, 1024, 64])
    view_401: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_71, [8, 1024, 64]);  expand_71 = None
    bmm_35: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_400, view_401)
    view_402: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_35, [1, 8, 1024, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_184: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    clone_17: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    view_403: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_17, [1, -1, 512]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_185: "f32[512, 512]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    view_404: "f32[1024, 512]" = torch.ops.aten.view.default(view_403, [1024, 512]);  view_403 = None
    mm_93: "f32[1024, 512]" = torch.ops.aten.mm.default(view_404, permute_185)
    view_405: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_93, [1, 1024, 512]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_60 = torch.ops.aten.native_dropout.default(view_405, 0.1, True);  view_405 = None
    getitem_120: "f32[1, 1024, 512]" = native_dropout_60[0]
    getitem_121: "b8[1, 1024, 512]" = native_dropout_60[1];  native_dropout_60 = None
    add_84: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_81, getitem_120);  getitem_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_31: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
    mean_30: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_85: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
    rsqrt_30: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    alias_59: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_30)
    mul_67: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_68: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_31, mul_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_186: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    view_406: "f32[1024, 512]" = torch.ops.aten.view.default(mul_68, [1024, 512]);  mul_68 = None
    mm_94: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_406, permute_186)
    view_407: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_94, [1, 1024, 2048]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_11: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_407);  view_407 = None
    alias_60: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(relu_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    native_dropout_61 = torch.ops.aten.native_dropout.default(relu_11, 0.1, True);  relu_11 = None
    getitem_122: "f32[1, 1024, 2048]" = native_dropout_61[0]
    getitem_123: "b8[1, 1024, 2048]" = native_dropout_61[1];  native_dropout_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_187: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    view_408: "f32[1024, 2048]" = torch.ops.aten.view.default(getitem_122, [1024, 2048]);  getitem_122 = None
    mm_95: "f32[1024, 512]" = torch.ops.aten.mm.default(view_408, permute_187)
    view_409: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_95, [1, 1024, 512]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_62 = torch.ops.aten.native_dropout.default(view_409, 0.1, True);  view_409 = None
    getitem_124: "f32[1, 1024, 512]" = native_dropout_62[0]
    getitem_125: "b8[1, 1024, 512]" = native_dropout_62[1];  native_dropout_62 = None
    add_86: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_84, getitem_124);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_32: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_86, 2)
    mean_31: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_87: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
    rsqrt_31: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    alias_61: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_31)
    mul_69: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_70: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_32, mul_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    native_dropout_63 = torch.ops.aten.native_dropout.default(mul_70, 0.1, True);  mul_70 = None
    getitem_126: "f32[1, 1024, 512]" = native_dropout_63[0]
    getitem_127: "b8[1, 1024, 512]" = native_dropout_63[1];  native_dropout_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1772, code: sequence_output = sequence_output * (self.model_dim**-0.5)
    mul_71: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(getitem_126, 0.04419417382415922);  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1774, code: lm_logits = self.lm_head(sequence_output)
    permute_188: "f32[512, 32128]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    view_410: "f32[1024, 512]" = torch.ops.aten.view.default(mul_71, [1024, 512]);  mul_71 = None
    mm_96: "f32[1024, 32128]" = torch.ops.aten.mm.default(view_410, permute_188)
    view_411: "f32[1, 1024, 32128]" = torch.ops.aten.view.default(mm_96, [1, 1024, 32128]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1781, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    view_412: "f32[1024, 32128]" = torch.ops.aten.view.default(view_411, [-1, 32128])
    view_413: "i64[1024]" = torch.ops.aten.view.default(primals_134, [-1]);  primals_134 = None
    amax_18: "f32[1024, 1]" = torch.ops.aten.amax.default(view_412, [1], True)
    sub_23: "f32[1024, 32128]" = torch.ops.aten.sub.Tensor(view_412, amax_18);  view_412 = amax_18 = None
    exp_18: "f32[1024, 32128]" = torch.ops.aten.exp.default(sub_23)
    sum_19: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
    log_2: "f32[1024, 1]" = torch.ops.aten.log.default(sum_19);  sum_19 = None
    sub_24: "f32[1024, 32128]" = torch.ops.aten.sub.Tensor(sub_23, log_2);  sub_23 = log_2 = None
    alias_62: "f32[1024, 32128]" = torch.ops.aten.alias.default(sub_24)
    ne: "b8[1024]" = torch.ops.aten.ne.Scalar(view_413, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_2: "i64[1024]" = torch.ops.aten.where.self(ne, view_413, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_17: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_24, 1, unsqueeze_17);  sub_24 = unsqueeze_17 = None
    squeeze: "f32[1024]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg_1: "f32[1024]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[1024]" = torch.ops.aten.ne.Scalar(view_413, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[1024]" = torch.ops.aten.where.self(ne_1, neg_1, scalar_tensor_1);  ne_1 = neg_1 = scalar_tensor_1 = None
    ne_2: "b8[1024]" = torch.ops.aten.ne.Scalar(view_413, -100)
    sum_20: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_7: "f32[]" = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
    sum_21: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_22: "f32[]" = torch.ops.aten.div.Tensor(sum_21, convert_element_type_7);  sum_21 = None
    div_23: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_7);  tangents_1 = convert_element_type_7 = None
    unsqueeze_18: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(view_413, 1);  view_413 = None
    ne_3: "b8[1024, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_18, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_4: "i64[1024, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_18, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full_7: "f32[1024, 32128]" = torch.ops.aten.full.default([1024, 32128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[1024, 32128]" = torch.ops.aten.scatter.value(full_7, 1, where_4, -1.0);  full_7 = where_4 = None
    ne_4: "b8[1024, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_18, -100);  unsqueeze_18 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[1024, 1]" = torch.ops.aten.where.self(ne_4, div_23, scalar_tensor_3);  ne_4 = div_23 = scalar_tensor_3 = None
    mul_72: "f32[1024, 32128]" = torch.ops.aten.mul.Tensor(scatter, where_5);  scatter = where_5 = None
    alias_63: "f32[1024, 32128]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    exp_19: "f32[1024, 32128]" = torch.ops.aten.exp.default(alias_63);  alias_63 = None
    sum_22: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_72, [1], True)
    mul_73: "f32[1024, 32128]" = torch.ops.aten.mul.Tensor(exp_19, sum_22);  exp_19 = sum_22 = None
    sub_25: "f32[1024, 32128]" = torch.ops.aten.sub.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    view_414: "f32[1, 1024, 32128]" = torch.ops.aten.view.default(sub_25, [1, 1024, 32128]);  sub_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1781, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    add_88: "f32[1, 1024, 32128]" = torch.ops.aten.add.Tensor(tangents_2, view_414);  tangents_2 = view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1774, code: lm_logits = self.lm_head(sequence_output)
    view_415: "f32[1024, 32128]" = torch.ops.aten.view.default(add_88, [1024, 32128]);  add_88 = None
    permute_189: "f32[32128, 1024]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_97: "f32[32128, 512]" = torch.ops.aten.mm.default(permute_189, view_410);  permute_189 = view_410 = None
    permute_190: "f32[512, 32128]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    permute_191: "f32[32128, 512]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    mm_98: "f32[1024, 512]" = torch.ops.aten.mm.default(view_415, permute_191);  view_415 = permute_191 = None
    view_416: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_98, [1, 1024, 512]);  mm_98 = None
    permute_192: "f32[32128, 512]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1772, code: sequence_output = sequence_output * (self.model_dim**-0.5)
    mul_74: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_416, 0.04419417382415922);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_75: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_76: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    clone_18: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_76, memory_format = torch.contiguous_format);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_77: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(clone_18, primals_32);  primals_32 = None
    mul_78: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(clone_18, mul_69);  clone_18 = mul_69 = None
    sum_23: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_78, [0, 1], True);  mul_78 = None
    view_417: "f32[512]" = torch.ops.aten.view.default(sum_23, [512]);  sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_79: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_77, add_86)
    mul_80: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_77, rsqrt_31);  mul_77 = rsqrt_31 = None
    sum_24: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_79, [2], True);  mul_79 = None
    alias_64: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    pow_33: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_64, 3);  alias_64 = None
    mul_81: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_24, -0.5);  sum_24 = None
    mul_82: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_81, pow_33);  mul_81 = pow_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_72: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_82, [1, 1024, 512]);  mul_82 = None
    div_24: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_72, 512);  expand_72 = None
    pow_34: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_86, 1.0);  add_86 = None
    mul_83: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
    mul_84: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_24, mul_83);  div_24 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_89: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(mul_80, mul_84);  mul_80 = mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_9: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
    mul_85: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_86: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_89, mul_85);  mul_85 = None
    clone_19: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_86, memory_format = torch.contiguous_format);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_418: "f32[1024, 512]" = torch.ops.aten.view.default(clone_19, [1024, 512]);  clone_19 = None
    permute_193: "f32[512, 1024]" = torch.ops.aten.permute.default(view_418, [1, 0])
    mm_99: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_193, view_408);  permute_193 = view_408 = None
    permute_194: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    permute_195: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    mm_100: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_418, permute_195);  view_418 = permute_195 = None
    view_419: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_100, [1, 1024, 2048]);  mm_100 = None
    permute_196: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None
    mul_87: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_88: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_419, mul_87);  view_419 = mul_87 = None
    clone_20: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_88, memory_format = torch.contiguous_format);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_65: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_1: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_65, 0);  alias_65 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_1, scalar_tensor_4, clone_20);  le_1 = scalar_tensor_4 = clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_420: "f32[1024, 2048]" = torch.ops.aten.view.default(where_6, [1024, 2048]);  where_6 = None
    permute_197: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_101: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_197, view_406);  permute_197 = view_406 = None
    permute_198: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    permute_199: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    mm_102: "f32[1024, 512]" = torch.ops.aten.mm.default(view_420, permute_199);  view_420 = permute_199 = None
    view_421: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_102, [1, 1024, 512]);  mm_102 = None
    permute_200: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_89: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_421, primals_31);  primals_31 = None
    mul_90: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_421, mul_67);  view_421 = mul_67 = None
    sum_25: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_90, [0, 1], True);  mul_90 = None
    view_422: "f32[512]" = torch.ops.aten.view.default(sum_25, [512]);  sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_91: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_89, add_84)
    mul_92: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_89, rsqrt_30);  mul_89 = rsqrt_30 = None
    sum_26: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_91, [2], True);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_90: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_89, mul_92);  add_89 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_66: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    pow_35: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_66, 3);  alias_66 = None
    mul_93: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_26, -0.5);  sum_26 = None
    mul_94: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_93, pow_35);  mul_93 = pow_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_73: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_94, [1, 1024, 512]);  mul_94 = None
    div_25: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_73, 512);  expand_73 = None
    pow_36: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 1.0);  add_84 = None
    mul_95: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_36, 2.0);  pow_36 = None
    mul_96: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_25, mul_95);  div_25 = mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_91: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_90, mul_96);  add_90 = mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_11: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_97: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_98: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_91, mul_97);  mul_97 = None
    clone_21: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_98, memory_format = torch.contiguous_format);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_423: "f32[1024, 512]" = torch.ops.aten.view.default(clone_21, [1024, 512]);  clone_21 = None
    permute_201: "f32[512, 1024]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_103: "f32[512, 512]" = torch.ops.aten.mm.default(permute_201, view_404);  permute_201 = view_404 = None
    permute_202: "f32[512, 512]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    permute_203: "f32[512, 512]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    mm_104: "f32[1024, 512]" = torch.ops.aten.mm.default(view_423, permute_203);  view_423 = permute_203 = None
    view_424: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_104, [1, 1024, 512]);  mm_104 = None
    permute_204: "f32[512, 512]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_425: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_424, [1, 1024, 8, 64]);  view_424 = None
    permute_205: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_426: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_205, [8, 1024, 64]);  permute_205 = None
    permute_206: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_400, [0, 2, 1]);  view_400 = None
    bmm_36: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_206, view_426);  permute_206 = None
    permute_207: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_401, [0, 2, 1]);  view_401 = None
    bmm_37: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_426, permute_207);  view_426 = permute_207 = None
    view_427: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_36, [1, 8, 1024, 64]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_92: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_26, view_427);  tangents_26 = view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_428: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_37, [1, 8, 1024, 1024]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_12: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_119, torch.float32);  getitem_119 = None
    mul_99: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_100: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_428, mul_99);  view_428 = mul_99 = None
    clone_22: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_100, memory_format = torch.contiguous_format);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_67: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    mul_101: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_22, alias_67);  clone_22 = None
    sum_27: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [-1], True)
    mul_102: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_67, sum_27);  alias_67 = sum_27 = None
    sub_26: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_26, 0);  sub_26 = None
    full_8: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_8, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided, squeeze_1);  as_strided = squeeze_1 = None
    as_strided_scatter: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_8, copy, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_8 = copy = None
    as_strided_3: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter = None
    new_empty_strided: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_3, [8, 1024, 1024], [1048576, 1024, 1])
    copy_1: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided, as_strided_3);  new_empty_strided = as_strided_3 = None
    as_strided_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_1, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_23: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_5, memory_format = torch.contiguous_format)
    copy_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_5, clone_23);  as_strided_5 = clone_23 = None
    as_strided_scatter_1: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_1, copy_2, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_1 = copy_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_208: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_395, [0, 2, 1]);  view_395 = None
    bmm_38: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_208, as_strided_scatter_1);  permute_208 = None
    permute_209: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_396, [0, 2, 1]);  view_396 = None
    bmm_39: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_1, permute_209);  as_strided_scatter_1 = permute_209 = None
    view_429: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_38, [1, 8, 64, 1024]);  bmm_38 = None
    view_430: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_39, [1, 8, 1024, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_210: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_429, [0, 1, 3, 2]);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_93: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_25, permute_210);  tangents_25 = permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_211: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_92, [0, 2, 1, 3]);  add_92 = None
    clone_24: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
    view_431: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_24, [1, 1024, 512]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_432: "f32[1024, 512]" = torch.ops.aten.view.default(view_431, [1024, 512]);  view_431 = None
    permute_212: "f32[512, 1024]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_105: "f32[512, 512]" = torch.ops.aten.mm.default(permute_212, view_392);  permute_212 = view_392 = None
    permute_213: "f32[512, 512]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    permute_214: "f32[512, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    mm_106: "f32[1024, 512]" = torch.ops.aten.mm.default(view_432, permute_214);  view_432 = permute_214 = None
    view_433: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_106, [1, 1024, 512]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_94: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(tangents_27, view_433);  tangents_27 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_215: "f32[512, 512]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_216: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_93, [0, 2, 1, 3]);  add_93 = None
    clone_25: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    view_434: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_25, [1, 1024, 512]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_435: "f32[1024, 512]" = torch.ops.aten.view.default(view_434, [1024, 512]);  view_434 = None
    permute_217: "f32[512, 1024]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_107: "f32[512, 512]" = torch.ops.aten.mm.default(permute_217, view_389);  permute_217 = view_389 = None
    permute_218: "f32[512, 512]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    permute_219: "f32[512, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    mm_108: "f32[1024, 512]" = torch.ops.aten.mm.default(view_435, permute_219);  view_435 = permute_219 = None
    view_436: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_108, [1, 1024, 512]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_95: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_94, view_436);  add_94 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_220: "f32[512, 512]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_221: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
    clone_26: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    view_437: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_26, [1, 1024, 512]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_438: "f32[1024, 512]" = torch.ops.aten.view.default(view_437, [1024, 512]);  view_437 = None
    permute_222: "f32[512, 1024]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_109: "f32[512, 512]" = torch.ops.aten.mm.default(permute_222, view_386);  permute_222 = view_386 = None
    permute_223: "f32[512, 512]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    permute_224: "f32[512, 512]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    mm_110: "f32[1024, 512]" = torch.ops.aten.mm.default(view_438, permute_224);  view_438 = permute_224 = None
    view_439: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_110, [1, 1024, 512]);  mm_110 = None
    permute_225: "f32[512, 512]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_103: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_439, primals_30);  primals_30 = None
    mul_104: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_439, mul_65);  view_439 = mul_65 = None
    sum_28: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_104, [0, 1], True);  mul_104 = None
    view_440: "f32[512]" = torch.ops.aten.view.default(sum_28, [512]);  sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_105: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_103, add_81)
    mul_106: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_103, rsqrt_29);  mul_103 = rsqrt_29 = None
    sum_29: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_105, [2], True);  mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_96: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_91, mul_106);  add_91 = mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_68: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    pow_37: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_68, 3);  alias_68 = None
    mul_107: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_29, -0.5);  sum_29 = None
    mul_108: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_107, pow_37);  mul_107 = pow_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_74: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_108, [1, 1024, 512]);  mul_108 = None
    div_26: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_74, 512);  expand_74 = None
    pow_38: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_81, 1.0);  add_81 = None
    mul_109: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
    mul_110: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_26, mul_109);  div_26 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_97: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_96, mul_110);  add_96 = mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_13: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_111: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_112: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_97, mul_111);  mul_111 = None
    clone_27: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_112, memory_format = torch.contiguous_format);  mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_441: "f32[1024, 512]" = torch.ops.aten.view.default(clone_27, [1024, 512]);  clone_27 = None
    permute_226: "f32[512, 1024]" = torch.ops.aten.permute.default(view_441, [1, 0])
    mm_111: "f32[512, 512]" = torch.ops.aten.mm.default(permute_226, view_384);  permute_226 = view_384 = None
    permute_227: "f32[512, 512]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    permute_228: "f32[512, 512]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    mm_112: "f32[1024, 512]" = torch.ops.aten.mm.default(view_441, permute_228);  view_441 = permute_228 = None
    view_442: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_112, [1, 1024, 512]);  mm_112 = None
    permute_229: "f32[512, 512]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_443: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_442, [1, 1024, 8, 64]);  view_442 = None
    permute_230: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_444: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_230, [8, 1024, 64]);  permute_230 = None
    permute_231: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None
    bmm_40: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_231, view_444);  permute_231 = None
    permute_232: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
    bmm_41: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_444, permute_232);  view_444 = permute_232 = None
    view_445: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_40, [1, 8, 1024, 64]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_98: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_24, view_445);  tangents_24 = view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_446: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_41, [1, 8, 1024, 1024]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_14: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_113: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_114: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_446, mul_113);  view_446 = mul_113 = None
    clone_28: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_114, memory_format = torch.contiguous_format);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_69: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    mul_115: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_28, alias_69);  clone_28 = None
    sum_30: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_115, [-1], True)
    mul_116: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_69, sum_30);  alias_69 = sum_30 = None
    sub_27: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_2: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_27, 0);  sub_27 = None
    full_9: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_7: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_9, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_3: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_7, squeeze_2);  as_strided_7 = squeeze_2 = None
    as_strided_scatter_2: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_9, copy_3, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_9 = copy_3 = None
    as_strided_10: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_2, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_2 = None
    new_empty_strided_1: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_10, [8, 1024, 1024], [1048576, 1024, 1])
    copy_4: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_1, as_strided_10);  new_empty_strided_1 = as_strided_10 = None
    as_strided_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_4, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_29: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_12, memory_format = torch.contiguous_format)
    copy_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_12, clone_29);  as_strided_12 = None
    as_strided_scatter_3: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_4, copy_5, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_4 = copy_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_233: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
    bmm_42: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_233, as_strided_scatter_3);  permute_233 = None
    permute_234: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1]);  view_376 = None
    bmm_43: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_3, permute_234);  as_strided_scatter_3 = permute_234 = None
    view_447: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_42, [1, 8, 64, 1024]);  bmm_42 = None
    view_448: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_43, [1, 8, 1024, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_235: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_447, [0, 1, 3, 2]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_99: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_23, permute_235);  tangents_23 = permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_236: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_98, [0, 2, 1, 3]);  add_98 = None
    clone_30: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    view_449: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_30, [1, 1024, 512]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_450: "f32[1024, 512]" = torch.ops.aten.view.default(view_449, [1024, 512]);  view_449 = None
    permute_237: "f32[512, 1024]" = torch.ops.aten.permute.default(view_450, [1, 0])
    mm_113: "f32[512, 512]" = torch.ops.aten.mm.default(permute_237, view_372);  permute_237 = view_372 = None
    permute_238: "f32[512, 512]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    permute_239: "f32[512, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    mm_114: "f32[1024, 512]" = torch.ops.aten.mm.default(view_450, permute_239);  view_450 = permute_239 = None
    view_451: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_114, [1, 1024, 512]);  mm_114 = None
    permute_240: "f32[512, 512]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_241: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_99, [0, 2, 1, 3]);  add_99 = None
    clone_31: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    view_452: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_31, [1, 1024, 512]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_453: "f32[1024, 512]" = torch.ops.aten.view.default(view_452, [1024, 512]);  view_452 = None
    permute_242: "f32[512, 1024]" = torch.ops.aten.permute.default(view_453, [1, 0])
    mm_115: "f32[512, 512]" = torch.ops.aten.mm.default(permute_242, view_369);  permute_242 = view_369 = None
    permute_243: "f32[512, 512]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    permute_244: "f32[512, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    mm_116: "f32[1024, 512]" = torch.ops.aten.mm.default(view_453, permute_244);  view_453 = permute_244 = None
    view_454: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_116, [1, 1024, 512]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_100: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_451, view_454);  view_451 = view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_245: "f32[512, 512]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_246: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    clone_32: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    view_455: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_32, [1, 1024, 512]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_456: "f32[1024, 512]" = torch.ops.aten.view.default(view_455, [1024, 512]);  view_455 = None
    permute_247: "f32[512, 1024]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_117: "f32[512, 512]" = torch.ops.aten.mm.default(permute_247, view_366);  permute_247 = view_366 = None
    permute_248: "f32[512, 512]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    permute_249: "f32[512, 512]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    mm_118: "f32[1024, 512]" = torch.ops.aten.mm.default(view_456, permute_249);  view_456 = permute_249 = None
    view_457: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_118, [1, 1024, 512]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_101: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_100, view_457);  add_100 = view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_250: "f32[512, 512]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_117: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_101, primals_29);  primals_29 = None
    mul_118: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_101, mul_63);  add_101 = mul_63 = None
    sum_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_118, [0, 1], True);  mul_118 = None
    view_458: "f32[512]" = torch.ops.aten.view.default(sum_31, [512]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_119: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_117, add_78)
    mul_120: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_117, rsqrt_28);  mul_117 = rsqrt_28 = None
    sum_32: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_102: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_97, mul_120);  add_97 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_70: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    pow_39: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_70, 3);  alias_70 = None
    mul_121: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
    mul_122: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_121, pow_39);  mul_121 = pow_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_75: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_122, [1, 1024, 512]);  mul_122 = None
    div_27: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_75, 512);  expand_75 = None
    pow_40: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 1.0);  add_78 = None
    mul_123: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
    mul_124: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_27, mul_123);  div_27 = mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_103: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_102, mul_124);  add_102 = mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_15: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None
    mul_125: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_126: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_103, mul_125);  mul_125 = None
    clone_33: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_126, memory_format = torch.contiguous_format);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_459: "f32[1024, 512]" = torch.ops.aten.view.default(clone_33, [1024, 512]);  clone_33 = None
    permute_251: "f32[512, 1024]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_119: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_251, view_364);  permute_251 = view_364 = None
    permute_252: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    permute_253: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    mm_120: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_459, permute_253);  view_459 = permute_253 = None
    view_460: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_120, [1, 1024, 2048]);  mm_120 = None
    permute_254: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_127: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_128: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_460, mul_127);  view_460 = mul_127 = None
    clone_34: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_128, memory_format = torch.contiguous_format);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_71: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_2: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_71, 0);  alias_71 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_2, scalar_tensor_5, clone_34);  le_2 = scalar_tensor_5 = clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_461: "f32[1024, 2048]" = torch.ops.aten.view.default(where_7, [1024, 2048]);  where_7 = None
    permute_255: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_461, [1, 0])
    mm_121: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_255, view_362);  permute_255 = view_362 = None
    permute_256: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    permute_257: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    mm_122: "f32[1024, 512]" = torch.ops.aten.mm.default(view_461, permute_257);  view_461 = permute_257 = None
    view_462: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_122, [1, 1024, 512]);  mm_122 = None
    permute_258: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_129: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_462, primals_28);  primals_28 = None
    mul_130: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_462, mul_61);  view_462 = mul_61 = None
    sum_33: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_130, [0, 1], True);  mul_130 = None
    view_463: "f32[512]" = torch.ops.aten.view.default(sum_33, [512]);  sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_131: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_129, add_76)
    mul_132: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_129, rsqrt_27);  mul_129 = rsqrt_27 = None
    sum_34: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_104: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_103, mul_132);  add_103 = mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_72: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    pow_41: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_72, 3);  alias_72 = None
    mul_133: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_34, -0.5);  sum_34 = None
    mul_134: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_133, pow_41);  mul_133 = pow_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_76: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_134, [1, 1024, 512]);  mul_134 = None
    div_28: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_76, 512);  expand_76 = None
    pow_42: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_76, 1.0);  add_76 = None
    mul_135: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
    mul_136: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_28, mul_135);  div_28 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_105: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_104, mul_136);  add_104 = mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_17: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_109, torch.float32);  getitem_109 = None
    mul_137: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_138: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_105, mul_137);  mul_137 = None
    clone_35: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_138, memory_format = torch.contiguous_format);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_464: "f32[1024, 512]" = torch.ops.aten.view.default(clone_35, [1024, 512]);  clone_35 = None
    permute_259: "f32[512, 1024]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_123: "f32[512, 512]" = torch.ops.aten.mm.default(permute_259, view_360);  permute_259 = view_360 = None
    permute_260: "f32[512, 512]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    permute_261: "f32[512, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    mm_124: "f32[1024, 512]" = torch.ops.aten.mm.default(view_464, permute_261);  view_464 = permute_261 = None
    view_465: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_124, [1, 1024, 512]);  mm_124 = None
    permute_262: "f32[512, 512]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_466: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_465, [1, 1024, 8, 64]);  view_465 = None
    permute_263: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_467: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_263, [8, 1024, 64]);  permute_263 = None
    permute_264: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_356, [0, 2, 1]);  view_356 = None
    bmm_44: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_264, view_467);  permute_264 = None
    permute_265: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_357, [0, 2, 1]);  view_357 = None
    bmm_45: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_467, permute_265);  view_467 = permute_265 = None
    view_468: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_44, [1, 8, 1024, 64]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_106: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_22, view_468);  tangents_22 = view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_469: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_45, [1, 8, 1024, 1024]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_18: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_139: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_140: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_469, mul_139);  view_469 = mul_139 = None
    clone_36: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_140, memory_format = torch.contiguous_format);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_73: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    mul_141: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_36, alias_73);  clone_36 = None
    sum_35: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [-1], True)
    mul_142: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_73, sum_35);  alias_73 = sum_35 = None
    sub_28: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_3: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_28, 0);  sub_28 = None
    full_10: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_14: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_10, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_6: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_14, squeeze_3);  as_strided_14 = squeeze_3 = None
    as_strided_scatter_4: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_10, copy_6, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_10 = copy_6 = None
    as_strided_17: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_4, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_4 = None
    new_empty_strided_2: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_17, [8, 1024, 1024], [1048576, 1024, 1])
    copy_7: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_2, as_strided_17);  new_empty_strided_2 = as_strided_17 = None
    as_strided_19: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_7, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_37: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_19, memory_format = torch.contiguous_format)
    copy_8: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_19, clone_37);  as_strided_19 = clone_37 = None
    as_strided_scatter_5: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_7, copy_8, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_7 = copy_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_266: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
    bmm_46: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_266, as_strided_scatter_5);  permute_266 = None
    permute_267: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
    bmm_47: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_5, permute_267);  as_strided_scatter_5 = permute_267 = None
    view_470: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_46, [1, 8, 64, 1024]);  bmm_46 = None
    view_471: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_47, [1, 8, 1024, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_268: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_470, [0, 1, 3, 2]);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_107: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_21, permute_268);  tangents_21 = permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_269: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_106, [0, 2, 1, 3]);  add_106 = None
    clone_38: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
    view_472: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_38, [1, 1024, 512]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_473: "f32[1024, 512]" = torch.ops.aten.view.default(view_472, [1024, 512]);  view_472 = None
    permute_270: "f32[512, 1024]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_125: "f32[512, 512]" = torch.ops.aten.mm.default(permute_270, view_348);  permute_270 = view_348 = None
    permute_271: "f32[512, 512]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    permute_272: "f32[512, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    mm_126: "f32[1024, 512]" = torch.ops.aten.mm.default(view_473, permute_272);  view_473 = permute_272 = None
    view_474: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_126, [1, 1024, 512]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_108: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_95, view_474);  add_95 = view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_273: "f32[512, 512]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_274: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_107, [0, 2, 1, 3]);  add_107 = None
    clone_39: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
    view_475: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_39, [1, 1024, 512]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_476: "f32[1024, 512]" = torch.ops.aten.view.default(view_475, [1024, 512]);  view_475 = None
    permute_275: "f32[512, 1024]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_127: "f32[512, 512]" = torch.ops.aten.mm.default(permute_275, view_345);  permute_275 = view_345 = None
    permute_276: "f32[512, 512]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    permute_277: "f32[512, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    mm_128: "f32[1024, 512]" = torch.ops.aten.mm.default(view_476, permute_277);  view_476 = permute_277 = None
    view_477: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_128, [1, 1024, 512]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_109: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_108, view_477);  add_108 = view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_278: "f32[512, 512]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_279: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
    clone_40: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    view_478: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_40, [1, 1024, 512]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_479: "f32[1024, 512]" = torch.ops.aten.view.default(view_478, [1024, 512]);  view_478 = None
    permute_280: "f32[512, 1024]" = torch.ops.aten.permute.default(view_479, [1, 0])
    mm_129: "f32[512, 512]" = torch.ops.aten.mm.default(permute_280, view_342);  permute_280 = view_342 = None
    permute_281: "f32[512, 512]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    permute_282: "f32[512, 512]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    mm_130: "f32[1024, 512]" = torch.ops.aten.mm.default(view_479, permute_282);  view_479 = permute_282 = None
    view_480: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_130, [1, 1024, 512]);  mm_130 = None
    permute_283: "f32[512, 512]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_143: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_480, primals_27);  primals_27 = None
    mul_144: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_480, mul_59);  view_480 = mul_59 = None
    sum_36: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_144, [0, 1], True);  mul_144 = None
    view_481: "f32[512]" = torch.ops.aten.view.default(sum_36, [512]);  sum_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_145: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_143, add_73)
    mul_146: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_143, rsqrt_26);  mul_143 = rsqrt_26 = None
    sum_37: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_110: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_105, mul_146);  add_105 = mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_74: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    pow_43: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_74, 3);  alias_74 = None
    mul_147: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_37, -0.5);  sum_37 = None
    mul_148: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_147, pow_43);  mul_147 = pow_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_77: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_148, [1, 1024, 512]);  mul_148 = None
    div_29: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_77, 512);  expand_77 = None
    pow_44: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_73, 1.0);  add_73 = None
    mul_149: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
    mul_150: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_29, mul_149);  div_29 = mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_111: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_110, mul_150);  add_110 = mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_19: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_151: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_152: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_111, mul_151);  mul_151 = None
    clone_41: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_152, memory_format = torch.contiguous_format);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_482: "f32[1024, 512]" = torch.ops.aten.view.default(clone_41, [1024, 512]);  clone_41 = None
    permute_284: "f32[512, 1024]" = torch.ops.aten.permute.default(view_482, [1, 0])
    mm_131: "f32[512, 512]" = torch.ops.aten.mm.default(permute_284, view_340);  permute_284 = view_340 = None
    permute_285: "f32[512, 512]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    permute_286: "f32[512, 512]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    mm_132: "f32[1024, 512]" = torch.ops.aten.mm.default(view_482, permute_286);  view_482 = permute_286 = None
    view_483: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_132, [1, 1024, 512]);  mm_132 = None
    permute_287: "f32[512, 512]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_484: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_483, [1, 1024, 8, 64]);  view_483 = None
    permute_288: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_485: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_288, [8, 1024, 64]);  permute_288 = None
    permute_289: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None
    bmm_48: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_289, view_485);  permute_289 = None
    permute_290: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
    bmm_49: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_485, permute_290);  view_485 = permute_290 = None
    view_486: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_48, [1, 8, 1024, 64]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_112: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_20, view_486);  tangents_20 = view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_487: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_49, [1, 8, 1024, 1024]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_20: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_103, torch.float32);  getitem_103 = None
    mul_153: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_154: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_487, mul_153);  view_487 = mul_153 = None
    clone_42: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_154, memory_format = torch.contiguous_format);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_75: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    mul_155: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_42, alias_75);  clone_42 = None
    sum_38: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [-1], True)
    mul_156: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_75, sum_38);  alias_75 = sum_38 = None
    sub_29: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_4: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_29, 0);  sub_29 = None
    full_11: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_21: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_11, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_9: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_21, squeeze_4);  as_strided_21 = squeeze_4 = None
    as_strided_scatter_6: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_11, copy_9, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_11 = copy_9 = None
    as_strided_24: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_6, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_6 = None
    new_empty_strided_3: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_24, [8, 1024, 1024], [1048576, 1024, 1])
    copy_10: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_3, as_strided_24);  new_empty_strided_3 = as_strided_24 = None
    as_strided_26: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_10, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_43: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_26, memory_format = torch.contiguous_format)
    copy_11: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_26, clone_43);  as_strided_26 = None
    as_strided_scatter_7: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_10, copy_11, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_10 = copy_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_113: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(clone_29, clone_43);  clone_29 = clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_291: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
    bmm_50: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_291, as_strided_scatter_7);  permute_291 = None
    permute_292: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None
    bmm_51: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_7, permute_292);  as_strided_scatter_7 = permute_292 = None
    view_488: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_50, [1, 8, 64, 1024]);  bmm_50 = None
    view_489: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_51, [1, 8, 1024, 64]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_293: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_488, [0, 1, 3, 2]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_114: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_19, permute_293);  tangents_19 = permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_294: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_112, [0, 2, 1, 3]);  add_112 = None
    clone_44: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    view_490: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_44, [1, 1024, 512]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_491: "f32[1024, 512]" = torch.ops.aten.view.default(view_490, [1024, 512]);  view_490 = None
    permute_295: "f32[512, 1024]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_133: "f32[512, 512]" = torch.ops.aten.mm.default(permute_295, view_328);  permute_295 = view_328 = None
    permute_296: "f32[512, 512]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    permute_297: "f32[512, 512]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    mm_134: "f32[1024, 512]" = torch.ops.aten.mm.default(view_491, permute_297);  view_491 = permute_297 = None
    view_492: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_134, [1, 1024, 512]);  mm_134 = None
    permute_298: "f32[512, 512]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_299: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_114, [0, 2, 1, 3]);  add_114 = None
    clone_45: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_493: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_45, [1, 1024, 512]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_494: "f32[1024, 512]" = torch.ops.aten.view.default(view_493, [1024, 512]);  view_493 = None
    permute_300: "f32[512, 1024]" = torch.ops.aten.permute.default(view_494, [1, 0])
    mm_135: "f32[512, 512]" = torch.ops.aten.mm.default(permute_300, view_325);  permute_300 = view_325 = None
    permute_301: "f32[512, 512]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    permute_302: "f32[512, 512]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    mm_136: "f32[1024, 512]" = torch.ops.aten.mm.default(view_494, permute_302);  view_494 = permute_302 = None
    view_495: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_136, [1, 1024, 512]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_115: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_492, view_495);  view_492 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_303: "f32[512, 512]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_304: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
    clone_46: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
    view_496: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_46, [1, 1024, 512]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_497: "f32[1024, 512]" = torch.ops.aten.view.default(view_496, [1024, 512]);  view_496 = None
    permute_305: "f32[512, 1024]" = torch.ops.aten.permute.default(view_497, [1, 0])
    mm_137: "f32[512, 512]" = torch.ops.aten.mm.default(permute_305, view_322);  permute_305 = view_322 = None
    permute_306: "f32[512, 512]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    permute_307: "f32[512, 512]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    mm_138: "f32[1024, 512]" = torch.ops.aten.mm.default(view_497, permute_307);  view_497 = permute_307 = None
    view_498: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_138, [1, 1024, 512]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_116: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_115, view_498);  add_115 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_308: "f32[512, 512]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_157: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_116, primals_26);  primals_26 = None
    mul_158: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_116, mul_57);  add_116 = mul_57 = None
    sum_39: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True);  mul_158 = None
    view_499: "f32[512]" = torch.ops.aten.view.default(sum_39, [512]);  sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_159: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_157, add_70)
    mul_160: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_157, rsqrt_25);  mul_157 = rsqrt_25 = None
    sum_40: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_117: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_111, mul_160);  add_111 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_76: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    pow_45: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_76, 3);  alias_76 = None
    mul_161: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_40, -0.5);  sum_40 = None
    mul_162: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_161, pow_45);  mul_161 = pow_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_78: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_162, [1, 1024, 512]);  mul_162 = None
    div_30: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_78, 512);  expand_78 = None
    pow_46: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 1.0);  add_70 = None
    mul_163: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
    mul_164: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_30, mul_163);  div_30 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_118: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_117, mul_164);  add_117 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_21: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_165: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_166: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_118, mul_165);  mul_165 = None
    clone_47: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_166, memory_format = torch.contiguous_format);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_500: "f32[1024, 512]" = torch.ops.aten.view.default(clone_47, [1024, 512]);  clone_47 = None
    permute_309: "f32[512, 1024]" = torch.ops.aten.permute.default(view_500, [1, 0])
    mm_139: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_309, view_320);  permute_309 = view_320 = None
    permute_310: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    permute_311: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    mm_140: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_500, permute_311);  view_500 = permute_311 = None
    view_501: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_140, [1, 1024, 2048]);  mm_140 = None
    permute_312: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_99, torch.float32);  getitem_99 = None
    mul_167: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_168: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_501, mul_167);  view_501 = mul_167 = None
    clone_48: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_168, memory_format = torch.contiguous_format);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_77: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_3: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_77, 0);  alias_77 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_3, scalar_tensor_6, clone_48);  le_3 = scalar_tensor_6 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_502: "f32[1024, 2048]" = torch.ops.aten.view.default(where_8, [1024, 2048]);  where_8 = None
    permute_313: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_502, [1, 0])
    mm_141: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_313, view_318);  permute_313 = view_318 = None
    permute_314: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    permute_315: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    mm_142: "f32[1024, 512]" = torch.ops.aten.mm.default(view_502, permute_315);  view_502 = permute_315 = None
    view_503: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_142, [1, 1024, 512]);  mm_142 = None
    permute_316: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_169: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_503, primals_25);  primals_25 = None
    mul_170: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_503, mul_55);  view_503 = mul_55 = None
    sum_41: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_170, [0, 1], True);  mul_170 = None
    view_504: "f32[512]" = torch.ops.aten.view.default(sum_41, [512]);  sum_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_171: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_169, add_68)
    mul_172: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_169, rsqrt_24);  mul_169 = rsqrt_24 = None
    sum_42: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_119: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_118, mul_172);  add_118 = mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_78: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    pow_47: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_78, 3);  alias_78 = None
    mul_173: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_42, -0.5);  sum_42 = None
    mul_174: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_173, pow_47);  mul_173 = pow_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_79: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_174, [1, 1024, 512]);  mul_174 = None
    div_31: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_79, 512);  expand_79 = None
    pow_48: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_68, 1.0);  add_68 = None
    mul_175: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
    mul_176: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_31, mul_175);  div_31 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_120: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_119, mul_176);  add_119 = mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_23: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_177: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_178: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_120, mul_177);  mul_177 = None
    clone_49: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_178, memory_format = torch.contiguous_format);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_505: "f32[1024, 512]" = torch.ops.aten.view.default(clone_49, [1024, 512]);  clone_49 = None
    permute_317: "f32[512, 1024]" = torch.ops.aten.permute.default(view_505, [1, 0])
    mm_143: "f32[512, 512]" = torch.ops.aten.mm.default(permute_317, view_316);  permute_317 = view_316 = None
    permute_318: "f32[512, 512]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    permute_319: "f32[512, 512]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_144: "f32[1024, 512]" = torch.ops.aten.mm.default(view_505, permute_319);  view_505 = permute_319 = None
    view_506: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_144, [1, 1024, 512]);  mm_144 = None
    permute_320: "f32[512, 512]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_507: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_506, [1, 1024, 8, 64]);  view_506 = None
    permute_321: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_508: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_321, [8, 1024, 64]);  permute_321 = None
    permute_322: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
    bmm_52: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_322, view_508);  permute_322 = None
    permute_323: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    bmm_53: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_508, permute_323);  view_508 = permute_323 = None
    view_509: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_52, [1, 8, 1024, 64]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_121: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_18, view_509);  tangents_18 = view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_510: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_53, [1, 8, 1024, 1024]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_24: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_179: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_180: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_510, mul_179);  view_510 = mul_179 = None
    clone_50: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_180, memory_format = torch.contiguous_format);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_79: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    mul_181: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_50, alias_79);  clone_50 = None
    sum_43: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [-1], True)
    mul_182: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_79, sum_43);  alias_79 = sum_43 = None
    sub_30: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_5: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_30, 0);  sub_30 = None
    full_12: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_28: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_12, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_12: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_28, squeeze_5);  as_strided_28 = squeeze_5 = None
    as_strided_scatter_8: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_12, copy_12, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_12 = copy_12 = None
    as_strided_31: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_8, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_8 = None
    new_empty_strided_4: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_31, [8, 1024, 1024], [1048576, 1024, 1])
    copy_13: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_4, as_strided_31);  new_empty_strided_4 = as_strided_31 = None
    as_strided_33: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_13, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_51: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_33, memory_format = torch.contiguous_format)
    copy_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_33, clone_51);  as_strided_33 = clone_51 = None
    as_strided_scatter_9: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_13, copy_14, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_13 = copy_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_324: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
    bmm_54: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_324, as_strided_scatter_9);  permute_324 = None
    permute_325: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
    bmm_55: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_9, permute_325);  as_strided_scatter_9 = permute_325 = None
    view_511: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_54, [1, 8, 64, 1024]);  bmm_54 = None
    view_512: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_55, [1, 8, 1024, 64]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_326: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_511, [0, 1, 3, 2]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_122: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_17, permute_326);  tangents_17 = permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_327: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_121, [0, 2, 1, 3]);  add_121 = None
    clone_52: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    view_513: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_52, [1, 1024, 512]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_514: "f32[1024, 512]" = torch.ops.aten.view.default(view_513, [1024, 512]);  view_513 = None
    permute_328: "f32[512, 1024]" = torch.ops.aten.permute.default(view_514, [1, 0])
    mm_145: "f32[512, 512]" = torch.ops.aten.mm.default(permute_328, view_304);  permute_328 = view_304 = None
    permute_329: "f32[512, 512]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    permute_330: "f32[512, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    mm_146: "f32[1024, 512]" = torch.ops.aten.mm.default(view_514, permute_330);  view_514 = permute_330 = None
    view_515: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_146, [1, 1024, 512]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_123: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_109, view_515);  add_109 = view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_331: "f32[512, 512]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_332: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_122, [0, 2, 1, 3]);  add_122 = None
    clone_53: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_332, memory_format = torch.contiguous_format);  permute_332 = None
    view_516: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_53, [1, 1024, 512]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_517: "f32[1024, 512]" = torch.ops.aten.view.default(view_516, [1024, 512]);  view_516 = None
    permute_333: "f32[512, 1024]" = torch.ops.aten.permute.default(view_517, [1, 0])
    mm_147: "f32[512, 512]" = torch.ops.aten.mm.default(permute_333, view_301);  permute_333 = view_301 = None
    permute_334: "f32[512, 512]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    permute_335: "f32[512, 512]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    mm_148: "f32[1024, 512]" = torch.ops.aten.mm.default(view_517, permute_335);  view_517 = permute_335 = None
    view_518: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_148, [1, 1024, 512]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_124: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_123, view_518);  add_123 = view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_336: "f32[512, 512]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_337: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
    clone_54: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
    view_519: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_54, [1, 1024, 512]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_520: "f32[1024, 512]" = torch.ops.aten.view.default(view_519, [1024, 512]);  view_519 = None
    permute_338: "f32[512, 1024]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_149: "f32[512, 512]" = torch.ops.aten.mm.default(permute_338, view_298);  permute_338 = view_298 = None
    permute_339: "f32[512, 512]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    permute_340: "f32[512, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    mm_150: "f32[1024, 512]" = torch.ops.aten.mm.default(view_520, permute_340);  view_520 = permute_340 = None
    view_521: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_150, [1, 1024, 512]);  mm_150 = None
    permute_341: "f32[512, 512]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_183: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_521, primals_24);  primals_24 = None
    mul_184: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_521, mul_53);  view_521 = mul_53 = None
    sum_44: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1], True);  mul_184 = None
    view_522: "f32[512]" = torch.ops.aten.view.default(sum_44, [512]);  sum_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_185: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_183, add_65)
    mul_186: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_183, rsqrt_23);  mul_183 = rsqrt_23 = None
    sum_45: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_125: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_120, mul_186);  add_120 = mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_80: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    pow_49: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_80, 3);  alias_80 = None
    mul_187: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_45, -0.5);  sum_45 = None
    mul_188: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_187, pow_49);  mul_187 = pow_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_80: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_188, [1, 1024, 512]);  mul_188 = None
    div_32: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_80, 512);  expand_80 = None
    pow_50: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_65, 1.0);  add_65 = None
    mul_189: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
    mul_190: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_32, mul_189);  div_32 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_126: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_125, mul_190);  add_125 = mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_25: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_93, torch.float32);  getitem_93 = None
    mul_191: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_192: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_126, mul_191);  mul_191 = None
    clone_55: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_192, memory_format = torch.contiguous_format);  mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_523: "f32[1024, 512]" = torch.ops.aten.view.default(clone_55, [1024, 512]);  clone_55 = None
    permute_342: "f32[512, 1024]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_151: "f32[512, 512]" = torch.ops.aten.mm.default(permute_342, view_296);  permute_342 = view_296 = None
    permute_343: "f32[512, 512]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    permute_344: "f32[512, 512]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    mm_152: "f32[1024, 512]" = torch.ops.aten.mm.default(view_523, permute_344);  view_523 = permute_344 = None
    view_524: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_152, [1, 1024, 512]);  mm_152 = None
    permute_345: "f32[512, 512]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_525: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_524, [1, 1024, 8, 64]);  view_524 = None
    permute_346: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_525, [0, 2, 1, 3]);  view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_526: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_346, [8, 1024, 64]);  permute_346 = None
    permute_347: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
    bmm_56: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_347, view_526);  permute_347 = None
    permute_348: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
    bmm_57: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_526, permute_348);  view_526 = permute_348 = None
    view_527: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_56, [1, 8, 1024, 64]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_127: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_16, view_527);  tangents_16 = view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_528: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_57, [1, 8, 1024, 1024]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_26: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_193: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_194: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_528, mul_193);  view_528 = mul_193 = None
    clone_56: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_194, memory_format = torch.contiguous_format);  mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_81: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    mul_195: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_56, alias_81);  clone_56 = None
    sum_46: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [-1], True)
    mul_196: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_81, sum_46);  alias_81 = sum_46 = None
    sub_31: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_6: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_31, 0);  sub_31 = None
    full_13: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_35: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_13, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_15: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_35, squeeze_6);  as_strided_35 = squeeze_6 = None
    as_strided_scatter_10: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_13, copy_15, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_13 = copy_15 = None
    as_strided_38: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_10, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_10 = None
    new_empty_strided_5: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_38, [8, 1024, 1024], [1048576, 1024, 1])
    copy_16: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_5, as_strided_38);  new_empty_strided_5 = as_strided_38 = None
    as_strided_40: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_16, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_57: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_40, memory_format = torch.contiguous_format)
    copy_17: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_40, clone_57);  as_strided_40 = None
    as_strided_scatter_11: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_16, copy_17, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_16 = copy_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_128: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_113, clone_57);  add_113 = clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_349: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    bmm_58: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_349, as_strided_scatter_11);  permute_349 = None
    permute_350: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    bmm_59: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_11, permute_350);  as_strided_scatter_11 = permute_350 = None
    view_529: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_58, [1, 8, 64, 1024]);  bmm_58 = None
    view_530: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_59, [1, 8, 1024, 64]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_351: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_529, [0, 1, 3, 2]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_129: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_15, permute_351);  tangents_15 = permute_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_352: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_127, [0, 2, 1, 3]);  add_127 = None
    clone_58: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
    view_531: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_58, [1, 1024, 512]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_532: "f32[1024, 512]" = torch.ops.aten.view.default(view_531, [1024, 512]);  view_531 = None
    permute_353: "f32[512, 1024]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_153: "f32[512, 512]" = torch.ops.aten.mm.default(permute_353, view_284);  permute_353 = view_284 = None
    permute_354: "f32[512, 512]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    permute_355: "f32[512, 512]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_154: "f32[1024, 512]" = torch.ops.aten.mm.default(view_532, permute_355);  view_532 = permute_355 = None
    view_533: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_154, [1, 1024, 512]);  mm_154 = None
    permute_356: "f32[512, 512]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_357: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_129, [0, 2, 1, 3]);  add_129 = None
    clone_59: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    view_534: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_59, [1, 1024, 512]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_535: "f32[1024, 512]" = torch.ops.aten.view.default(view_534, [1024, 512]);  view_534 = None
    permute_358: "f32[512, 1024]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_155: "f32[512, 512]" = torch.ops.aten.mm.default(permute_358, view_281);  permute_358 = view_281 = None
    permute_359: "f32[512, 512]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    permute_360: "f32[512, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_156: "f32[1024, 512]" = torch.ops.aten.mm.default(view_535, permute_360);  view_535 = permute_360 = None
    view_536: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_156, [1, 1024, 512]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_130: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_533, view_536);  view_533 = view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_361: "f32[512, 512]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_362: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_530, [0, 2, 1, 3]);  view_530 = None
    clone_60: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_362, memory_format = torch.contiguous_format);  permute_362 = None
    view_537: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_60, [1, 1024, 512]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_538: "f32[1024, 512]" = torch.ops.aten.view.default(view_537, [1024, 512]);  view_537 = None
    permute_363: "f32[512, 1024]" = torch.ops.aten.permute.default(view_538, [1, 0])
    mm_157: "f32[512, 512]" = torch.ops.aten.mm.default(permute_363, view_278);  permute_363 = view_278 = None
    permute_364: "f32[512, 512]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    permute_365: "f32[512, 512]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_158: "f32[1024, 512]" = torch.ops.aten.mm.default(view_538, permute_365);  view_538 = permute_365 = None
    view_539: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_158, [1, 1024, 512]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_131: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_130, view_539);  add_130 = view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_366: "f32[512, 512]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_197: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_131, primals_23);  primals_23 = None
    mul_198: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_131, mul_51);  add_131 = mul_51 = None
    sum_47: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1], True);  mul_198 = None
    view_540: "f32[512]" = torch.ops.aten.view.default(sum_47, [512]);  sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_199: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_197, add_62)
    mul_200: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_197, rsqrt_22);  mul_197 = rsqrt_22 = None
    sum_48: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_132: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_126, mul_200);  add_126 = mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_82: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    pow_51: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_82, 3);  alias_82 = None
    mul_201: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
    mul_202: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_201, pow_51);  mul_201 = pow_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_81: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_202, [1, 1024, 512]);  mul_202 = None
    div_33: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_81, 512);  expand_81 = None
    pow_52: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 1.0);  add_62 = None
    mul_203: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_52, 2.0);  pow_52 = None
    mul_204: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_33, mul_203);  div_33 = mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_133: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_132, mul_204);  add_132 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_27: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_89, torch.float32);  getitem_89 = None
    mul_205: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_206: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_133, mul_205);  mul_205 = None
    clone_61: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_206, memory_format = torch.contiguous_format);  mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_541: "f32[1024, 512]" = torch.ops.aten.view.default(clone_61, [1024, 512]);  clone_61 = None
    permute_367: "f32[512, 1024]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_159: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_367, view_276);  permute_367 = view_276 = None
    permute_368: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    permute_369: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    mm_160: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_541, permute_369);  view_541 = permute_369 = None
    view_542: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_160, [1, 1024, 2048]);  mm_160 = None
    permute_370: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_207: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_208: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_542, mul_207);  view_542 = mul_207 = None
    clone_62: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_208, memory_format = torch.contiguous_format);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_83: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_4: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_83, 0);  alias_83 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_4, scalar_tensor_7, clone_62);  le_4 = scalar_tensor_7 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_543: "f32[1024, 2048]" = torch.ops.aten.view.default(where_9, [1024, 2048]);  where_9 = None
    permute_371: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_161: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_371, view_274);  permute_371 = view_274 = None
    permute_372: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    permute_373: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    mm_162: "f32[1024, 512]" = torch.ops.aten.mm.default(view_543, permute_373);  view_543 = permute_373 = None
    view_544: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_162, [1, 1024, 512]);  mm_162 = None
    permute_374: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_209: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_544, primals_22);  primals_22 = None
    mul_210: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_544, mul_49);  view_544 = mul_49 = None
    sum_49: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1], True);  mul_210 = None
    view_545: "f32[512]" = torch.ops.aten.view.default(sum_49, [512]);  sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_211: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_209, add_60)
    mul_212: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_209, rsqrt_21);  mul_209 = rsqrt_21 = None
    sum_50: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_134: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_133, mul_212);  add_133 = mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_84: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    pow_53: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_84, 3);  alias_84 = None
    mul_213: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_50, -0.5);  sum_50 = None
    mul_214: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_213, pow_53);  mul_213 = pow_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_82: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_214, [1, 1024, 512]);  mul_214 = None
    div_34: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_82, 512);  expand_82 = None
    pow_54: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 1.0);  add_60 = None
    mul_215: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_54, 2.0);  pow_54 = None
    mul_216: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_34, mul_215);  div_34 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_135: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_134, mul_216);  add_134 = mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_29: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_217: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_218: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_135, mul_217);  mul_217 = None
    clone_63: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_218, memory_format = torch.contiguous_format);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_546: "f32[1024, 512]" = torch.ops.aten.view.default(clone_63, [1024, 512]);  clone_63 = None
    permute_375: "f32[512, 1024]" = torch.ops.aten.permute.default(view_546, [1, 0])
    mm_163: "f32[512, 512]" = torch.ops.aten.mm.default(permute_375, view_272);  permute_375 = view_272 = None
    permute_376: "f32[512, 512]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    permute_377: "f32[512, 512]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_164: "f32[1024, 512]" = torch.ops.aten.mm.default(view_546, permute_377);  view_546 = permute_377 = None
    view_547: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_164, [1, 1024, 512]);  mm_164 = None
    permute_378: "f32[512, 512]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_548: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_547, [1, 1024, 8, 64]);  view_547 = None
    permute_379: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_549: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_379, [8, 1024, 64]);  permute_379 = None
    permute_380: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
    bmm_60: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_380, view_549);  permute_380 = None
    permute_381: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
    bmm_61: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_549, permute_381);  view_549 = permute_381 = None
    view_550: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_60, [1, 8, 1024, 64]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_136: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_14, view_550);  tangents_14 = view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_551: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_61, [1, 8, 1024, 1024]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_30: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_83, torch.float32);  getitem_83 = None
    mul_219: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_220: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_551, mul_219);  view_551 = mul_219 = None
    clone_64: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_220, memory_format = torch.contiguous_format);  mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_85: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    mul_221: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_64, alias_85);  clone_64 = None
    sum_51: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [-1], True)
    mul_222: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_85, sum_51);  alias_85 = sum_51 = None
    sub_32: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_7: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_32, 0);  sub_32 = None
    full_14: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_42: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_14, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_18: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_42, squeeze_7);  as_strided_42 = squeeze_7 = None
    as_strided_scatter_12: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_14, copy_18, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_14 = copy_18 = None
    as_strided_45: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_12, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_12 = None
    new_empty_strided_6: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_45, [8, 1024, 1024], [1048576, 1024, 1])
    copy_19: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_6, as_strided_45);  new_empty_strided_6 = as_strided_45 = None
    as_strided_47: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_19, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_65: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_47, memory_format = torch.contiguous_format)
    copy_20: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_47, clone_65);  as_strided_47 = clone_65 = None
    as_strided_scatter_13: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_19, copy_20, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_19 = copy_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_382: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
    bmm_62: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_382, as_strided_scatter_13);  permute_382 = None
    permute_383: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    bmm_63: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_13, permute_383);  as_strided_scatter_13 = permute_383 = None
    view_552: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_62, [1, 8, 64, 1024]);  bmm_62 = None
    view_553: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_63, [1, 8, 1024, 64]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_384: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_552, [0, 1, 3, 2]);  view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_137: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_13, permute_384);  tangents_13 = permute_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_385: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_136, [0, 2, 1, 3]);  add_136 = None
    clone_66: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_554: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_66, [1, 1024, 512]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_555: "f32[1024, 512]" = torch.ops.aten.view.default(view_554, [1024, 512]);  view_554 = None
    permute_386: "f32[512, 1024]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_165: "f32[512, 512]" = torch.ops.aten.mm.default(permute_386, view_260);  permute_386 = view_260 = None
    permute_387: "f32[512, 512]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    permute_388: "f32[512, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_166: "f32[1024, 512]" = torch.ops.aten.mm.default(view_555, permute_388);  view_555 = permute_388 = None
    view_556: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_166, [1, 1024, 512]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_138: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_124, view_556);  add_124 = view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_389: "f32[512, 512]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_390: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_137, [0, 2, 1, 3]);  add_137 = None
    clone_67: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
    view_557: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_67, [1, 1024, 512]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_558: "f32[1024, 512]" = torch.ops.aten.view.default(view_557, [1024, 512]);  view_557 = None
    permute_391: "f32[512, 1024]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_167: "f32[512, 512]" = torch.ops.aten.mm.default(permute_391, view_257);  permute_391 = view_257 = None
    permute_392: "f32[512, 512]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    permute_393: "f32[512, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_168: "f32[1024, 512]" = torch.ops.aten.mm.default(view_558, permute_393);  view_558 = permute_393 = None
    view_559: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_168, [1, 1024, 512]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_139: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_138, view_559);  add_138 = view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_394: "f32[512, 512]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_395: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_553, [0, 2, 1, 3]);  view_553 = None
    clone_68: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    view_560: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_68, [1, 1024, 512]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_561: "f32[1024, 512]" = torch.ops.aten.view.default(view_560, [1024, 512]);  view_560 = None
    permute_396: "f32[512, 1024]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_169: "f32[512, 512]" = torch.ops.aten.mm.default(permute_396, view_254);  permute_396 = view_254 = None
    permute_397: "f32[512, 512]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    permute_398: "f32[512, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    mm_170: "f32[1024, 512]" = torch.ops.aten.mm.default(view_561, permute_398);  view_561 = permute_398 = None
    view_562: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_170, [1, 1024, 512]);  mm_170 = None
    permute_399: "f32[512, 512]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_223: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_562, primals_21);  primals_21 = None
    mul_224: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_562, mul_47);  view_562 = mul_47 = None
    sum_52: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 1], True);  mul_224 = None
    view_563: "f32[512]" = torch.ops.aten.view.default(sum_52, [512]);  sum_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_225: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_223, add_57)
    mul_226: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_223, rsqrt_20);  mul_223 = rsqrt_20 = None
    sum_53: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_140: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_135, mul_226);  add_135 = mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_86: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    pow_55: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_86, 3);  alias_86 = None
    mul_227: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_53, -0.5);  sum_53 = None
    mul_228: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_227, pow_55);  mul_227 = pow_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_83: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_228, [1, 1024, 512]);  mul_228 = None
    div_35: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_83, 512);  expand_83 = None
    pow_56: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_57, 1.0);  add_57 = None
    mul_229: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_56, 2.0);  pow_56 = None
    mul_230: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_35, mul_229);  div_35 = mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_141: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_140, mul_230);  add_140 = mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_31: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_231: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_232: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_141, mul_231);  mul_231 = None
    clone_69: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_232, memory_format = torch.contiguous_format);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_564: "f32[1024, 512]" = torch.ops.aten.view.default(clone_69, [1024, 512]);  clone_69 = None
    permute_400: "f32[512, 1024]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_171: "f32[512, 512]" = torch.ops.aten.mm.default(permute_400, view_252);  permute_400 = view_252 = None
    permute_401: "f32[512, 512]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    permute_402: "f32[512, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_172: "f32[1024, 512]" = torch.ops.aten.mm.default(view_564, permute_402);  view_564 = permute_402 = None
    view_565: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_172, [1, 1024, 512]);  mm_172 = None
    permute_403: "f32[512, 512]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_566: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_565, [1, 1024, 8, 64]);  view_565 = None
    permute_404: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_567: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_404, [8, 1024, 64]);  permute_404 = None
    permute_405: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
    bmm_64: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_405, view_567);  permute_405 = None
    permute_406: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
    bmm_65: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_567, permute_406);  view_567 = permute_406 = None
    view_568: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_64, [1, 8, 1024, 64]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_142: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_12, view_568);  tangents_12 = view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_569: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_65, [1, 8, 1024, 1024]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_32: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
    mul_233: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_234: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_569, mul_233);  view_569 = mul_233 = None
    clone_70: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_234, memory_format = torch.contiguous_format);  mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_87: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    mul_235: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_70, alias_87);  clone_70 = None
    sum_54: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_235, [-1], True)
    mul_236: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_87, sum_54);  alias_87 = sum_54 = None
    sub_33: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_8: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_33, 0);  sub_33 = None
    full_15: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_49: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_15, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_21: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_49, squeeze_8);  as_strided_49 = squeeze_8 = None
    as_strided_scatter_14: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_15, copy_21, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_15 = copy_21 = None
    as_strided_52: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_14, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_14 = None
    new_empty_strided_7: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_52, [8, 1024, 1024], [1048576, 1024, 1])
    copy_22: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_7, as_strided_52);  new_empty_strided_7 = as_strided_52 = None
    as_strided_54: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_22, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_71: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_54, memory_format = torch.contiguous_format)
    copy_23: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_54, clone_71);  as_strided_54 = None
    as_strided_scatter_15: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_22, copy_23, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_22 = copy_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_143: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_128, clone_71);  add_128 = clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_407: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
    bmm_66: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_407, as_strided_scatter_15);  permute_407 = None
    permute_408: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    bmm_67: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_15, permute_408);  as_strided_scatter_15 = permute_408 = None
    view_570: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_66, [1, 8, 64, 1024]);  bmm_66 = None
    view_571: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_67, [1, 8, 1024, 64]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_409: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_570, [0, 1, 3, 2]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_144: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_11, permute_409);  tangents_11 = permute_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_410: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_142, [0, 2, 1, 3]);  add_142 = None
    clone_72: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
    view_572: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_72, [1, 1024, 512]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_573: "f32[1024, 512]" = torch.ops.aten.view.default(view_572, [1024, 512]);  view_572 = None
    permute_411: "f32[512, 1024]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_173: "f32[512, 512]" = torch.ops.aten.mm.default(permute_411, view_240);  permute_411 = view_240 = None
    permute_412: "f32[512, 512]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    permute_413: "f32[512, 512]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_174: "f32[1024, 512]" = torch.ops.aten.mm.default(view_573, permute_413);  view_573 = permute_413 = None
    view_574: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_174, [1, 1024, 512]);  mm_174 = None
    permute_414: "f32[512, 512]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_415: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_144, [0, 2, 1, 3]);  add_144 = None
    clone_73: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_415, memory_format = torch.contiguous_format);  permute_415 = None
    view_575: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_73, [1, 1024, 512]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_576: "f32[1024, 512]" = torch.ops.aten.view.default(view_575, [1024, 512]);  view_575 = None
    permute_416: "f32[512, 1024]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_175: "f32[512, 512]" = torch.ops.aten.mm.default(permute_416, view_237);  permute_416 = view_237 = None
    permute_417: "f32[512, 512]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    permute_418: "f32[512, 512]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_176: "f32[1024, 512]" = torch.ops.aten.mm.default(view_576, permute_418);  view_576 = permute_418 = None
    view_577: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_176, [1, 1024, 512]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_145: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_574, view_577);  view_574 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_419: "f32[512, 512]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_420: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None
    clone_74: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_578: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_74, [1, 1024, 512]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_579: "f32[1024, 512]" = torch.ops.aten.view.default(view_578, [1024, 512]);  view_578 = None
    permute_421: "f32[512, 1024]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_177: "f32[512, 512]" = torch.ops.aten.mm.default(permute_421, view_234);  permute_421 = view_234 = None
    permute_422: "f32[512, 512]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    permute_423: "f32[512, 512]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_178: "f32[1024, 512]" = torch.ops.aten.mm.default(view_579, permute_423);  view_579 = permute_423 = None
    view_580: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_178, [1, 1024, 512]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_146: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_145, view_580);  add_145 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_424: "f32[512, 512]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_237: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_146, primals_20);  primals_20 = None
    mul_238: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_146, mul_45);  add_146 = mul_45 = None
    sum_55: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 1], True);  mul_238 = None
    view_581: "f32[512]" = torch.ops.aten.view.default(sum_55, [512]);  sum_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_239: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_237, add_54)
    mul_240: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_237, rsqrt_19);  mul_237 = rsqrt_19 = None
    sum_56: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_147: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_141, mul_240);  add_141 = mul_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_88: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    pow_57: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_88, 3);  alias_88 = None
    mul_241: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_56, -0.5);  sum_56 = None
    mul_242: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_241, pow_57);  mul_241 = pow_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_84: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_242, [1, 1024, 512]);  mul_242 = None
    div_36: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_84, 512);  expand_84 = None
    pow_58: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 1.0);  add_54 = None
    mul_243: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_58, 2.0);  pow_58 = None
    mul_244: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_36, mul_243);  div_36 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_148: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_147, mul_244);  add_147 = mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_33: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_245: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_246: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_148, mul_245);  mul_245 = None
    clone_75: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_246, memory_format = torch.contiguous_format);  mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_582: "f32[1024, 512]" = torch.ops.aten.view.default(clone_75, [1024, 512]);  clone_75 = None
    permute_425: "f32[512, 1024]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_179: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_425, view_232);  permute_425 = view_232 = None
    permute_426: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    permute_427: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_180: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_582, permute_427);  view_582 = permute_427 = None
    view_583: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_180, [1, 1024, 2048]);  mm_180 = None
    permute_428: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_426, [1, 0]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_247: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_248: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_583, mul_247);  view_583 = mul_247 = None
    clone_76: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_248, memory_format = torch.contiguous_format);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_89: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le_5: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_89, 0);  alias_89 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_5, scalar_tensor_8, clone_76);  le_5 = scalar_tensor_8 = clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_584: "f32[1024, 2048]" = torch.ops.aten.view.default(where_10, [1024, 2048]);  where_10 = None
    permute_429: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_584, [1, 0])
    mm_181: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_429, view_230);  permute_429 = view_230 = None
    permute_430: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    permute_431: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    mm_182: "f32[1024, 512]" = torch.ops.aten.mm.default(view_584, permute_431);  view_584 = permute_431 = None
    view_585: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_182, [1, 1024, 512]);  mm_182 = None
    permute_432: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_249: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_585, primals_19);  primals_19 = None
    mul_250: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_585, mul_43);  view_585 = mul_43 = None
    sum_57: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1], True);  mul_250 = None
    view_586: "f32[512]" = torch.ops.aten.view.default(sum_57, [512]);  sum_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_251: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_249, add_52)
    mul_252: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_249, rsqrt_18);  mul_249 = rsqrt_18 = None
    sum_58: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_149: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_148, mul_252);  add_148 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_90: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    pow_59: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_90, 3);  alias_90 = None
    mul_253: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_58, -0.5);  sum_58 = None
    mul_254: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_253, pow_59);  mul_253 = pow_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_85: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_254, [1, 1024, 512]);  mul_254 = None
    div_37: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_85, 512);  expand_85 = None
    pow_60: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 1.0);  add_52 = None
    mul_255: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
    mul_256: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_37, mul_255);  div_37 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_150: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_149, mul_256);  add_149 = mul_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_35: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_257: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_258: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_150, mul_257);  mul_257 = None
    clone_77: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_258, memory_format = torch.contiguous_format);  mul_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_587: "f32[1024, 512]" = torch.ops.aten.view.default(clone_77, [1024, 512]);  clone_77 = None
    permute_433: "f32[512, 1024]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_183: "f32[512, 512]" = torch.ops.aten.mm.default(permute_433, view_228);  permute_433 = view_228 = None
    permute_434: "f32[512, 512]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    permute_435: "f32[512, 512]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_184: "f32[1024, 512]" = torch.ops.aten.mm.default(view_587, permute_435);  view_587 = permute_435 = None
    view_588: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_184, [1, 1024, 512]);  mm_184 = None
    permute_436: "f32[512, 512]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_589: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_588, [1, 1024, 8, 64]);  view_588 = None
    permute_437: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_590: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_437, [8, 1024, 64]);  permute_437 = None
    permute_438: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
    bmm_68: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_438, view_590);  permute_438 = None
    permute_439: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
    bmm_69: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_590, permute_439);  view_590 = permute_439 = None
    view_591: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_68, [1, 8, 1024, 64]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_151: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_10, view_591);  tangents_10 = view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_592: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_69, [1, 8, 1024, 1024]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_36: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_259: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_260: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_592, mul_259);  view_592 = mul_259 = None
    clone_78: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_260, memory_format = torch.contiguous_format);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_91: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    mul_261: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_78, alias_91);  clone_78 = None
    sum_59: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [-1], True)
    mul_262: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_91, sum_59);  alias_91 = sum_59 = None
    sub_34: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_9: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_34, 0);  sub_34 = None
    full_16: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_56: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_16, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_24: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_56, squeeze_9);  as_strided_56 = squeeze_9 = None
    as_strided_scatter_16: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_16, copy_24, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_16 = copy_24 = None
    as_strided_59: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_16, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_16 = None
    new_empty_strided_8: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_59, [8, 1024, 1024], [1048576, 1024, 1])
    copy_25: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_8, as_strided_59);  new_empty_strided_8 = as_strided_59 = None
    as_strided_61: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_25, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_79: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_61, memory_format = torch.contiguous_format)
    copy_26: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_61, clone_79);  as_strided_61 = clone_79 = None
    as_strided_scatter_17: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_25, copy_26, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_25 = copy_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_440: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
    bmm_70: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_440, as_strided_scatter_17);  permute_440 = None
    permute_441: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
    bmm_71: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_17, permute_441);  as_strided_scatter_17 = permute_441 = None
    view_593: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_70, [1, 8, 64, 1024]);  bmm_70 = None
    view_594: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_71, [1, 8, 1024, 64]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_442: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_593, [0, 1, 3, 2]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_152: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_9, permute_442);  tangents_9 = permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_443: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_151, [0, 2, 1, 3]);  add_151 = None
    clone_80: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_443, memory_format = torch.contiguous_format);  permute_443 = None
    view_595: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_80, [1, 1024, 512]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_596: "f32[1024, 512]" = torch.ops.aten.view.default(view_595, [1024, 512]);  view_595 = None
    permute_444: "f32[512, 1024]" = torch.ops.aten.permute.default(view_596, [1, 0])
    mm_185: "f32[512, 512]" = torch.ops.aten.mm.default(permute_444, view_216);  permute_444 = view_216 = None
    permute_445: "f32[512, 512]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    permute_446: "f32[512, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_186: "f32[1024, 512]" = torch.ops.aten.mm.default(view_596, permute_446);  view_596 = permute_446 = None
    view_597: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_186, [1, 1024, 512]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_153: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_139, view_597);  add_139 = view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_447: "f32[512, 512]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_448: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_152, [0, 2, 1, 3]);  add_152 = None
    clone_81: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
    view_598: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_81, [1, 1024, 512]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_599: "f32[1024, 512]" = torch.ops.aten.view.default(view_598, [1024, 512]);  view_598 = None
    permute_449: "f32[512, 1024]" = torch.ops.aten.permute.default(view_599, [1, 0])
    mm_187: "f32[512, 512]" = torch.ops.aten.mm.default(permute_449, view_213);  permute_449 = view_213 = None
    permute_450: "f32[512, 512]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    permute_451: "f32[512, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_188: "f32[1024, 512]" = torch.ops.aten.mm.default(view_599, permute_451);  view_599 = permute_451 = None
    view_600: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_188, [1, 1024, 512]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_154: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_153, view_600);  add_153 = view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_452: "f32[512, 512]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_453: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None
    clone_82: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_601: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_82, [1, 1024, 512]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_602: "f32[1024, 512]" = torch.ops.aten.view.default(view_601, [1024, 512]);  view_601 = None
    permute_454: "f32[512, 1024]" = torch.ops.aten.permute.default(view_602, [1, 0])
    mm_189: "f32[512, 512]" = torch.ops.aten.mm.default(permute_454, view_210);  permute_454 = view_210 = None
    permute_455: "f32[512, 512]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    permute_456: "f32[512, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_190: "f32[1024, 512]" = torch.ops.aten.mm.default(view_602, permute_456);  view_602 = permute_456 = None
    view_603: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_190, [1, 1024, 512]);  mm_190 = None
    permute_457: "f32[512, 512]" = torch.ops.aten.permute.default(permute_455, [1, 0]);  permute_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_263: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_603, primals_18);  primals_18 = None
    mul_264: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_603, mul_41);  view_603 = mul_41 = None
    sum_60: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 1], True);  mul_264 = None
    view_604: "f32[512]" = torch.ops.aten.view.default(sum_60, [512]);  sum_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_265: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_263, add_49)
    mul_266: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_263, rsqrt_17);  mul_263 = rsqrt_17 = None
    sum_61: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True);  mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_155: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_150, mul_266);  add_150 = mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_92: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    pow_61: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_92, 3);  alias_92 = None
    mul_267: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_61, -0.5);  sum_61 = None
    mul_268: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_267, pow_61);  mul_267 = pow_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_86: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_268, [1, 1024, 512]);  mul_268 = None
    div_38: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_86, 512);  expand_86 = None
    pow_62: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 1.0);  add_49 = None
    mul_269: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_62, 2.0);  pow_62 = None
    mul_270: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_38, mul_269);  div_38 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_156: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_155, mul_270);  add_155 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_37: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_69, torch.float32);  getitem_69 = None
    mul_271: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_272: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_156, mul_271);  mul_271 = None
    clone_83: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_272, memory_format = torch.contiguous_format);  mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_605: "f32[1024, 512]" = torch.ops.aten.view.default(clone_83, [1024, 512]);  clone_83 = None
    permute_458: "f32[512, 1024]" = torch.ops.aten.permute.default(view_605, [1, 0])
    mm_191: "f32[512, 512]" = torch.ops.aten.mm.default(permute_458, view_208);  permute_458 = view_208 = None
    permute_459: "f32[512, 512]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    permute_460: "f32[512, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_192: "f32[1024, 512]" = torch.ops.aten.mm.default(view_605, permute_460);  view_605 = permute_460 = None
    view_606: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_192, [1, 1024, 512]);  mm_192 = None
    permute_461: "f32[512, 512]" = torch.ops.aten.permute.default(permute_459, [1, 0]);  permute_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_607: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_606, [1, 1024, 8, 64]);  view_606 = None
    permute_462: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_608: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_462, [8, 1024, 64]);  permute_462 = None
    permute_463: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    bmm_72: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_463, view_608);  permute_463 = None
    permute_464: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    bmm_73: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_608, permute_464);  view_608 = permute_464 = None
    view_609: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_72, [1, 8, 1024, 64]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_157: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_8, view_609);  tangents_8 = view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_610: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_73, [1, 8, 1024, 1024]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_38: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_273: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_274: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_610, mul_273);  view_610 = mul_273 = None
    clone_84: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_274, memory_format = torch.contiguous_format);  mul_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_93: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    mul_275: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_84, alias_93);  clone_84 = None
    sum_62: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [-1], True)
    mul_276: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_93, sum_62);  alias_93 = sum_62 = None
    sub_35: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_10: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_35, 0);  sub_35 = None
    full_17: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_63: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_17, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_27: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_63, squeeze_10);  as_strided_63 = squeeze_10 = None
    as_strided_scatter_18: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_17, copy_27, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_17 = copy_27 = None
    as_strided_66: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_18, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_18 = None
    new_empty_strided_9: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_66, [8, 1024, 1024], [1048576, 1024, 1])
    copy_28: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_9, as_strided_66);  new_empty_strided_9 = as_strided_66 = None
    as_strided_68: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_28, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_85: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_68, memory_format = torch.contiguous_format)
    copy_29: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_68, clone_85);  as_strided_68 = None
    as_strided_scatter_19: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_28, copy_29, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_28 = copy_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_158: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_143, clone_85);  add_143 = clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_465: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
    bmm_74: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_465, as_strided_scatter_19);  permute_465 = None
    permute_466: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
    bmm_75: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_19, permute_466);  as_strided_scatter_19 = permute_466 = None
    view_611: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_74, [1, 8, 64, 1024]);  bmm_74 = None
    view_612: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_75, [1, 8, 1024, 64]);  bmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_467: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_611, [0, 1, 3, 2]);  view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_159: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_7, permute_467);  tangents_7 = permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_468: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_157, [0, 2, 1, 3]);  add_157 = None
    clone_86: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    view_613: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_86, [1, 1024, 512]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_614: "f32[1024, 512]" = torch.ops.aten.view.default(view_613, [1024, 512]);  view_613 = None
    permute_469: "f32[512, 1024]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_193: "f32[512, 512]" = torch.ops.aten.mm.default(permute_469, view_196);  permute_469 = view_196 = None
    permute_470: "f32[512, 512]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    permute_471: "f32[512, 512]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_194: "f32[1024, 512]" = torch.ops.aten.mm.default(view_614, permute_471);  view_614 = permute_471 = None
    view_615: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_194, [1, 1024, 512]);  mm_194 = None
    permute_472: "f32[512, 512]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_473: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_159, [0, 2, 1, 3]);  add_159 = None
    clone_87: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
    view_616: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_87, [1, 1024, 512]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_617: "f32[1024, 512]" = torch.ops.aten.view.default(view_616, [1024, 512]);  view_616 = None
    permute_474: "f32[512, 1024]" = torch.ops.aten.permute.default(view_617, [1, 0])
    mm_195: "f32[512, 512]" = torch.ops.aten.mm.default(permute_474, view_193);  permute_474 = view_193 = None
    permute_475: "f32[512, 512]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    permute_476: "f32[512, 512]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_196: "f32[1024, 512]" = torch.ops.aten.mm.default(view_617, permute_476);  view_617 = permute_476 = None
    view_618: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_196, [1, 1024, 512]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_160: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_615, view_618);  view_615 = view_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_477: "f32[512, 512]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_478: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
    clone_88: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
    view_619: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_88, [1, 1024, 512]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_620: "f32[1024, 512]" = torch.ops.aten.view.default(view_619, [1024, 512]);  view_619 = None
    permute_479: "f32[512, 1024]" = torch.ops.aten.permute.default(view_620, [1, 0])
    mm_197: "f32[512, 512]" = torch.ops.aten.mm.default(permute_479, view_190);  permute_479 = view_190 = None
    permute_480: "f32[512, 512]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_198: "f32[1024, 512]" = torch.ops.aten.mm.default(view_620, permute_481);  view_620 = permute_481 = None
    view_621: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_198, [1, 1024, 512]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_161: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_160, view_621);  add_160 = view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_482: "f32[512, 512]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_277: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_161, primals_17);  primals_17 = None
    mul_278: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_161, mul_39);  add_161 = mul_39 = None
    sum_63: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1], True);  mul_278 = None
    view_622: "f32[512]" = torch.ops.aten.view.default(sum_63, [512]);  sum_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_279: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_277, add_46)
    mul_280: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_277, rsqrt_16);  mul_277 = rsqrt_16 = None
    sum_64: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_162: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_156, mul_280);  add_156 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_94: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    pow_63: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_94, 3);  alias_94 = None
    mul_281: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_64, -0.5);  sum_64 = None
    mul_282: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_281, pow_63);  mul_281 = pow_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_87: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_282, [1, 1024, 512]);  mul_282 = None
    div_39: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_87, 512);  expand_87 = None
    pow_64: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 1.0);  add_46 = None
    mul_283: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
    mul_284: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_39, mul_283);  div_39 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_163: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_162, mul_284);  add_162 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_39: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_285: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
    mul_286: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_163, mul_285);  mul_285 = None
    clone_89: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_286, memory_format = torch.contiguous_format);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_623: "f32[1024, 512]" = torch.ops.aten.view.default(clone_89, [1024, 512]);  clone_89 = None
    permute_483: "f32[512, 1024]" = torch.ops.aten.permute.default(view_623, [1, 0])
    mm_199: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_483, view_188);  permute_483 = view_188 = None
    permute_484: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    permute_485: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_200: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_623, permute_485);  view_623 = permute_485 = None
    view_624: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_200, [1, 1024, 2048]);  mm_200 = None
    permute_486: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_484, [1, 0]);  permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_40: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_63, torch.float32);  getitem_63 = None
    mul_287: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_288: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_624, mul_287);  view_624 = mul_287 = None
    clone_90: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_288, memory_format = torch.contiguous_format);  mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_95: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_6: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_95, 0);  alias_95 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_6, scalar_tensor_9, clone_90);  le_6 = scalar_tensor_9 = clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_625: "f32[1024, 2048]" = torch.ops.aten.view.default(where_11, [1024, 2048]);  where_11 = None
    permute_487: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_625, [1, 0])
    mm_201: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_487, view_186);  permute_487 = view_186 = None
    permute_488: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    permute_489: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_202: "f32[1024, 512]" = torch.ops.aten.mm.default(view_625, permute_489);  view_625 = permute_489 = None
    view_626: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_202, [1, 1024, 512]);  mm_202 = None
    permute_490: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_488, [1, 0]);  permute_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_289: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_626, primals_16);  primals_16 = None
    mul_290: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_626, mul_37);  view_626 = mul_37 = None
    sum_65: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1], True);  mul_290 = None
    view_627: "f32[512]" = torch.ops.aten.view.default(sum_65, [512]);  sum_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_291: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_289, add_44)
    mul_292: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_289, rsqrt_15);  mul_289 = rsqrt_15 = None
    sum_66: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True);  mul_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_164: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_163, mul_292);  add_163 = mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_96: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    pow_65: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_96, 3);  alias_96 = None
    mul_293: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_66, -0.5);  sum_66 = None
    mul_294: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_293, pow_65);  mul_293 = pow_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_88: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_294, [1, 1024, 512]);  mul_294 = None
    div_40: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_88, 512);  expand_88 = None
    pow_66: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 1.0);  add_44 = None
    mul_295: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
    mul_296: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_40, mul_295);  div_40 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_165: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_164, mul_296);  add_164 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_41: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_297: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
    mul_298: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_165, mul_297);  mul_297 = None
    clone_91: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_298, memory_format = torch.contiguous_format);  mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_628: "f32[1024, 512]" = torch.ops.aten.view.default(clone_91, [1024, 512]);  clone_91 = None
    permute_491: "f32[512, 1024]" = torch.ops.aten.permute.default(view_628, [1, 0])
    mm_203: "f32[512, 512]" = torch.ops.aten.mm.default(permute_491, view_184);  permute_491 = view_184 = None
    permute_492: "f32[512, 512]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    permute_493: "f32[512, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_204: "f32[1024, 512]" = torch.ops.aten.mm.default(view_628, permute_493);  view_628 = permute_493 = None
    view_629: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_204, [1, 1024, 512]);  mm_204 = None
    permute_494: "f32[512, 512]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_630: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_629, [1, 1024, 8, 64]);  view_629 = None
    permute_495: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_631: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_495, [8, 1024, 64]);  permute_495 = None
    permute_496: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_180, [0, 2, 1]);  view_180 = None
    bmm_76: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_496, view_631);  permute_496 = None
    permute_497: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
    bmm_77: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_631, permute_497);  view_631 = permute_497 = None
    view_632: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_76, [1, 8, 1024, 64]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_166: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_6, view_632);  tangents_6 = view_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_633: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_77, [1, 8, 1024, 1024]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_42: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_59, torch.float32);  getitem_59 = None
    mul_299: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
    mul_300: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_633, mul_299);  view_633 = mul_299 = None
    clone_92: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_300, memory_format = torch.contiguous_format);  mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_97: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    mul_301: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_92, alias_97);  clone_92 = None
    sum_67: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
    mul_302: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_97, sum_67);  alias_97 = sum_67 = None
    sub_36: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_11: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_36, 0);  sub_36 = None
    full_18: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_70: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_18, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_30: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_70, squeeze_11);  as_strided_70 = squeeze_11 = None
    as_strided_scatter_20: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_18, copy_30, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_18 = copy_30 = None
    as_strided_73: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_20, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_20 = None
    new_empty_strided_10: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_73, [8, 1024, 1024], [1048576, 1024, 1])
    copy_31: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_10, as_strided_73);  new_empty_strided_10 = as_strided_73 = None
    as_strided_75: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_31, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_93: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_75, memory_format = torch.contiguous_format)
    copy_32: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_75, clone_93);  as_strided_75 = clone_93 = None
    as_strided_scatter_21: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_31, copy_32, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_31 = copy_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_498: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
    bmm_78: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_498, as_strided_scatter_21);  permute_498 = None
    permute_499: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
    bmm_79: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_21, permute_499);  as_strided_scatter_21 = permute_499 = None
    view_634: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_78, [1, 8, 64, 1024]);  bmm_78 = None
    view_635: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_79, [1, 8, 1024, 64]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_500: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_634, [0, 1, 3, 2]);  view_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_167: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_5, permute_500);  tangents_5 = permute_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_501: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_166, [0, 2, 1, 3]);  add_166 = None
    clone_94: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
    view_636: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_94, [1, 1024, 512]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_637: "f32[1024, 512]" = torch.ops.aten.view.default(view_636, [1024, 512]);  view_636 = None
    permute_502: "f32[512, 1024]" = torch.ops.aten.permute.default(view_637, [1, 0])
    mm_205: "f32[512, 512]" = torch.ops.aten.mm.default(permute_502, view_172);  permute_502 = view_172 = None
    permute_503: "f32[512, 512]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    permute_504: "f32[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_206: "f32[1024, 512]" = torch.ops.aten.mm.default(view_637, permute_504);  view_637 = permute_504 = None
    view_638: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_206, [1, 1024, 512]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_168: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_154, view_638);  add_154 = view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_505: "f32[512, 512]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_506: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_167, [0, 2, 1, 3]);  add_167 = None
    clone_95: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_506, memory_format = torch.contiguous_format);  permute_506 = None
    view_639: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_95, [1, 1024, 512]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_640: "f32[1024, 512]" = torch.ops.aten.view.default(view_639, [1024, 512]);  view_639 = None
    permute_507: "f32[512, 1024]" = torch.ops.aten.permute.default(view_640, [1, 0])
    mm_207: "f32[512, 512]" = torch.ops.aten.mm.default(permute_507, view_169);  permute_507 = view_169 = None
    permute_508: "f32[512, 512]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    permute_509: "f32[512, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_208: "f32[1024, 512]" = torch.ops.aten.mm.default(view_640, permute_509);  view_640 = permute_509 = None
    view_641: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_208, [1, 1024, 512]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_169: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_168, view_641);  add_168 = view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_510: "f32[512, 512]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_511: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_635, [0, 2, 1, 3]);  view_635 = None
    clone_96: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_511, memory_format = torch.contiguous_format);  permute_511 = None
    view_642: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_96, [1, 1024, 512]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_643: "f32[1024, 512]" = torch.ops.aten.view.default(view_642, [1024, 512]);  view_642 = None
    permute_512: "f32[512, 1024]" = torch.ops.aten.permute.default(view_643, [1, 0])
    mm_209: "f32[512, 512]" = torch.ops.aten.mm.default(permute_512, view_166);  permute_512 = view_166 = None
    permute_513: "f32[512, 512]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    permute_514: "f32[512, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_210: "f32[1024, 512]" = torch.ops.aten.mm.default(view_643, permute_514);  view_643 = permute_514 = None
    view_644: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_210, [1, 1024, 512]);  mm_210 = None
    permute_515: "f32[512, 512]" = torch.ops.aten.permute.default(permute_513, [1, 0]);  permute_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_303: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_644, primals_15);  primals_15 = None
    mul_304: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_644, mul_35);  view_644 = mul_35 = None
    sum_68: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1], True);  mul_304 = None
    view_645: "f32[512]" = torch.ops.aten.view.default(sum_68, [512]);  sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_305: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_303, add_40)
    mul_306: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_303, rsqrt_14);  mul_303 = rsqrt_14 = None
    sum_69: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_170: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_165, mul_306);  add_165 = mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_98: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    pow_67: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_98, 3);  alias_98 = None
    mul_307: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_69, -0.5);  sum_69 = None
    mul_308: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_307, pow_67);  mul_307 = pow_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_89: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_308, [1, 1024, 512]);  mul_308 = None
    div_41: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_89, 512);  expand_89 = None
    pow_68: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_40, 1.0);  add_40 = None
    mul_309: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
    mul_310: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_41, mul_309);  div_41 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_171: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_170, mul_310);  add_170 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_43: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_311: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
    mul_312: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_171, mul_311);  mul_311 = None
    clone_97: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_312, memory_format = torch.contiguous_format);  mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_646: "f32[1024, 512]" = torch.ops.aten.view.default(clone_97, [1024, 512]);  clone_97 = None
    permute_516: "f32[512, 1024]" = torch.ops.aten.permute.default(view_646, [1, 0])
    mm_211: "f32[512, 512]" = torch.ops.aten.mm.default(permute_516, view_164);  permute_516 = view_164 = None
    permute_517: "f32[512, 512]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    permute_518: "f32[512, 512]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_212: "f32[1024, 512]" = torch.ops.aten.mm.default(view_646, permute_518);  view_646 = permute_518 = None
    view_647: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_212, [1, 1024, 512]);  mm_212 = None
    permute_519: "f32[512, 512]" = torch.ops.aten.permute.default(permute_517, [1, 0]);  permute_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_648: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_647, [1, 1024, 8, 64]);  view_647 = None
    permute_520: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_649: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_520, [8, 1024, 64]);  permute_520 = None
    permute_521: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
    bmm_80: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_521, view_649);  permute_521 = None
    permute_522: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    bmm_81: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_649, permute_522);  view_649 = permute_522 = None
    view_650: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_80, [1, 8, 1024, 64]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_172: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_4, view_650);  tangents_4 = view_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_651: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_81, [1, 8, 1024, 1024]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_44: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_313: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
    mul_314: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_651, mul_313);  view_651 = mul_313 = None
    clone_98: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_314, memory_format = torch.contiguous_format);  mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_99: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    mul_315: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_98, alias_99);  clone_98 = None
    sum_70: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [-1], True)
    mul_316: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_99, sum_70);  alias_99 = sum_70 = None
    sub_37: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_12: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_37, 0);  sub_37 = None
    full_19: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_77: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_19, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_33: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_77, squeeze_12);  as_strided_77 = squeeze_12 = None
    as_strided_scatter_22: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_19, copy_33, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_19 = copy_33 = None
    as_strided_80: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_22, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_22 = None
    new_empty_strided_11: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_80, [8, 1024, 1024], [1048576, 1024, 1])
    copy_34: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_11, as_strided_80);  new_empty_strided_11 = as_strided_80 = None
    as_strided_82: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_34, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_99: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_82, memory_format = torch.contiguous_format)
    copy_35: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_82, clone_99);  as_strided_82 = None
    as_strided_scatter_23: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_34, copy_35, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_34 = copy_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_173: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_158, clone_99);  add_158 = clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    squeeze_13: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(add_173, 0);  add_173 = None
    permute_523: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_13, [1, 2, 0]);  squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    eq: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_37, -1)
    unsqueeze_19: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_19, scalar_tensor_10, permute_523);  unsqueeze_19 = scalar_tensor_10 = permute_523 = None
    clone_100: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_12, memory_format = torch.contiguous_format);  where_12 = None
    full_20: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[32, 8]" = torch.ops.aten._unsafe_index_put.default(full_20, [add_37], clone_100, True);  full_20 = add_37 = clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_524: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    bmm_82: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_524, as_strided_scatter_23);  permute_524 = None
    permute_525: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None
    bmm_83: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_23, permute_525);  as_strided_scatter_23 = permute_525 = None
    view_652: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_82, [1, 8, 64, 1024]);  bmm_82 = None
    view_653: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_83, [1, 8, 1024, 64]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_526: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_652, [0, 1, 3, 2]);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_174: "f32[1, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_3, permute_526);  tangents_3 = permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_527: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_172, [0, 2, 1, 3]);  add_172 = None
    clone_101: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_527, memory_format = torch.contiguous_format);  permute_527 = None
    view_654: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_101, [1, 1024, 512]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_655: "f32[1024, 512]" = torch.ops.aten.view.default(view_654, [1024, 512]);  view_654 = None
    permute_528: "f32[512, 1024]" = torch.ops.aten.permute.default(view_655, [1, 0])
    mm_213: "f32[512, 512]" = torch.ops.aten.mm.default(permute_528, view_152);  permute_528 = view_152 = None
    permute_529: "f32[512, 512]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    permute_530: "f32[512, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_214: "f32[1024, 512]" = torch.ops.aten.mm.default(view_655, permute_530);  view_655 = permute_530 = None
    view_656: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_214, [1, 1024, 512]);  mm_214 = None
    permute_531: "f32[512, 512]" = torch.ops.aten.permute.default(permute_529, [1, 0]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_532: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(add_174, [0, 2, 1, 3]);  add_174 = None
    clone_102: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_532, memory_format = torch.contiguous_format);  permute_532 = None
    view_657: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_102, [1, 1024, 512]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_658: "f32[1024, 512]" = torch.ops.aten.view.default(view_657, [1024, 512]);  view_657 = None
    permute_533: "f32[512, 1024]" = torch.ops.aten.permute.default(view_658, [1, 0])
    mm_215: "f32[512, 512]" = torch.ops.aten.mm.default(permute_533, view_149);  permute_533 = view_149 = None
    permute_534: "f32[512, 512]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    permute_535: "f32[512, 512]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_216: "f32[1024, 512]" = torch.ops.aten.mm.default(view_658, permute_535);  view_658 = permute_535 = None
    view_659: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_216, [1, 1024, 512]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_175: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_656, view_659);  view_656 = view_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_536: "f32[512, 512]" = torch.ops.aten.permute.default(permute_534, [1, 0]);  permute_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_537: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    clone_103: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_537, memory_format = torch.contiguous_format);  permute_537 = None
    view_660: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_103, [1, 1024, 512]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_661: "f32[1024, 512]" = torch.ops.aten.view.default(view_660, [1024, 512]);  view_660 = None
    permute_538: "f32[512, 1024]" = torch.ops.aten.permute.default(view_661, [1, 0])
    mm_217: "f32[512, 512]" = torch.ops.aten.mm.default(permute_538, view_146);  permute_538 = view_146 = None
    permute_539: "f32[512, 512]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    permute_540: "f32[512, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_218: "f32[1024, 512]" = torch.ops.aten.mm.default(view_661, permute_540);  view_661 = permute_540 = None
    view_662: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_218, [1, 1024, 512]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_176: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_175, view_662);  add_175 = view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_541: "f32[512, 512]" = torch.ops.aten.permute.default(permute_539, [1, 0]);  permute_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_317: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_176, primals_14);  primals_14 = None
    mul_318: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_176, mul_32);  add_176 = mul_32 = None
    sum_71: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1], True);  mul_318 = None
    view_663: "f32[512]" = torch.ops.aten.view.default(sum_71, [512]);  sum_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_319: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_317, getitem_52)
    mul_320: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_317, rsqrt_13);  mul_317 = rsqrt_13 = None
    sum_72: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True);  mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_177: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_171, mul_320);  add_171 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_100: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    pow_69: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_100, 3);  alias_100 = None
    mul_321: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_72, -0.5);  sum_72 = None
    mul_322: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_321, pow_69);  mul_321 = pow_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_90: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_322, [1, 1024, 512]);  mul_322 = None
    div_42: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_90, 512);  expand_90 = None
    pow_70: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem_52, 1.0);  getitem_52 = None
    mul_323: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
    mul_324: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_42, mul_323);  div_42 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_178: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_177, mul_324);  add_177 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    convert_element_type_45: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
    mul_325: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
    mul_326: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_178, mul_325);  add_178 = mul_325 = None
    clone_104: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_326, memory_format = torch.contiguous_format);  mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    eq_1: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(view_145, -1)
    unsqueeze_20: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[1, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_20, scalar_tensor_11, clone_104);  unsqueeze_20 = scalar_tensor_11 = clone_104 = None
    full_21: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[32128, 512]" = torch.ops.aten._unsafe_index_put.default(full_21, [view_145], where_13, True);  full_21 = view_145 = where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_46: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_327: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
    mul_328: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_169, mul_327);  add_169 = mul_327 = None
    clone_105: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_328, memory_format = torch.contiguous_format);  mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_329: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(clone_105, primals_13);  primals_13 = None
    mul_330: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(clone_105, mul_27);  clone_105 = mul_27 = None
    sum_73: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1], True);  mul_330 = None
    view_664: "f32[512]" = torch.ops.aten.view.default(sum_73, [512]);  sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_331: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_329, add_33)
    mul_332: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_329, rsqrt_12);  mul_329 = rsqrt_12 = None
    sum_74: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    alias_101: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    pow_71: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_101, 3);  alias_101 = None
    mul_333: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_74, -0.5);  sum_74 = None
    mul_334: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_333, pow_71);  mul_333 = pow_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_91: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_334, [1, 1024, 512]);  mul_334 = None
    div_43: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_91, 512);  expand_91 = None
    pow_72: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_33, 1.0);  add_33 = None
    mul_335: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
    mul_336: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_43, mul_335);  div_43 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_179: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(mul_332, mul_336);  mul_332 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_47: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_337: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
    mul_338: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_179, mul_337);  mul_337 = None
    clone_106: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_338, memory_format = torch.contiguous_format);  mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_665: "f32[1024, 512]" = torch.ops.aten.view.default(clone_106, [1024, 512]);  clone_106 = None
    permute_542: "f32[512, 1024]" = torch.ops.aten.permute.default(view_665, [1, 0])
    mm_219: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_542, view_143);  permute_542 = view_143 = None
    permute_543: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    permute_544: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_220: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_665, permute_544);  view_665 = permute_544 = None
    view_666: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_220, [1, 1024, 2048]);  mm_220 = None
    permute_545: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_48: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_339: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
    mul_340: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_666, mul_339);  view_666 = mul_339 = None
    clone_107: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_340, memory_format = torch.contiguous_format);  mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_102: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    le_7: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_102, 0);  alias_102 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_7, scalar_tensor_12, clone_107);  le_7 = scalar_tensor_12 = clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_667: "f32[1024, 2048]" = torch.ops.aten.view.default(where_14, [1024, 2048]);  where_14 = None
    permute_546: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_667, [1, 0])
    mm_221: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_546, view_141);  permute_546 = view_141 = None
    permute_547: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    permute_548: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_222: "f32[1024, 512]" = torch.ops.aten.mm.default(view_667, permute_548);  view_667 = permute_548 = None
    view_668: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_222, [1, 1024, 512]);  mm_222 = None
    permute_549: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_547, [1, 0]);  permute_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_341: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_668, primals_12);  primals_12 = None
    mul_342: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_668, mul_25);  view_668 = mul_25 = None
    sum_75: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1], True);  mul_342 = None
    view_669: "f32[512]" = torch.ops.aten.view.default(sum_75, [512]);  sum_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_343: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_341, add_31)
    mul_344: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_341, rsqrt_11);  mul_341 = rsqrt_11 = None
    sum_76: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_180: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_179, mul_344);  add_179 = mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_103: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    pow_73: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_103, 3);  alias_103 = None
    mul_345: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_76, -0.5);  sum_76 = None
    mul_346: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_345, pow_73);  mul_345 = pow_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_92: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_346, [1, 1024, 512]);  mul_346 = None
    div_44: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_92, 512);  expand_92 = None
    pow_74: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 1.0);  add_31 = None
    mul_347: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
    mul_348: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_44, mul_347);  div_44 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_181: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_180, mul_348);  add_180 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_49: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_349: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
    mul_350: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_181, mul_349);  mul_349 = None
    clone_108: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_350, memory_format = torch.contiguous_format);  mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_670: "f32[1024, 512]" = torch.ops.aten.view.default(clone_108, [1024, 512]);  clone_108 = None
    permute_550: "f32[512, 1024]" = torch.ops.aten.permute.default(view_670, [1, 0])
    mm_223: "f32[512, 512]" = torch.ops.aten.mm.default(permute_550, view_139);  permute_550 = view_139 = None
    permute_551: "f32[512, 512]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    permute_552: "f32[512, 512]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_224: "f32[1024, 512]" = torch.ops.aten.mm.default(view_670, permute_552);  view_670 = permute_552 = None
    view_671: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_224, [1, 1024, 512]);  mm_224 = None
    permute_553: "f32[512, 512]" = torch.ops.aten.permute.default(permute_551, [1, 0]);  permute_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_672: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_671, [1, 1024, 8, 64]);  view_671 = None
    permute_554: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_673: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_554, [8, 1024, 64]);  permute_554 = None
    permute_555: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    bmm_84: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_555, view_673);  permute_555 = None
    permute_556: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    bmm_85: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_673, permute_556);  view_673 = permute_556 = None
    view_674: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_84, [1, 8, 1024, 64]);  bmm_84 = None
    view_675: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_85, [1, 8, 1024, 1024]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_50: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
    mul_351: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
    mul_352: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_675, mul_351);  view_675 = mul_351 = None
    clone_109: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_352, memory_format = torch.contiguous_format);  mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_104: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_353: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_109, alias_104);  clone_109 = None
    sum_77: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [-1], True)
    mul_354: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_104, sum_77);  alias_104 = sum_77 = None
    sub_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_14: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_38, 0);  sub_38 = None
    full_22: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_84: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_22, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_36: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_84, squeeze_14);  as_strided_84 = squeeze_14 = None
    as_strided_scatter_24: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_22, copy_36, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_22 = copy_36 = None
    as_strided_87: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_24, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_24 = None
    new_empty_strided_12: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_87, [8, 1024, 1024], [1048576, 1024, 1])
    copy_37: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_12, as_strided_87);  new_empty_strided_12 = as_strided_87 = None
    as_strided_89: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_37, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_110: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_89, memory_format = torch.contiguous_format)
    copy_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_89, clone_110);  as_strided_89 = None
    as_strided_scatter_25: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_37, copy_38, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_37 = copy_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_557: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
    bmm_86: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_557, as_strided_scatter_25);  permute_557 = None
    permute_558: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
    bmm_87: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_25, permute_558);  as_strided_scatter_25 = permute_558 = None
    view_676: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_86, [1, 8, 64, 1024]);  bmm_86 = None
    view_677: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_87, [1, 8, 1024, 64]);  bmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_559: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_676, [0, 1, 3, 2]);  view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_560: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_674, [0, 2, 1, 3]);  view_674 = None
    clone_111: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format);  permute_560 = None
    view_678: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_111, [1, 1024, 512]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_679: "f32[1024, 512]" = torch.ops.aten.view.default(view_678, [1024, 512]);  view_678 = None
    permute_561: "f32[512, 1024]" = torch.ops.aten.permute.default(view_679, [1, 0])
    mm_225: "f32[512, 512]" = torch.ops.aten.mm.default(permute_561, view_127);  permute_561 = view_127 = None
    permute_562: "f32[512, 512]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    permute_563: "f32[512, 512]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_226: "f32[1024, 512]" = torch.ops.aten.mm.default(view_679, permute_563);  view_679 = permute_563 = None
    view_680: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_226, [1, 1024, 512]);  mm_226 = None
    permute_564: "f32[512, 512]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_565: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_559, [0, 2, 1, 3]);  permute_559 = None
    view_681: "f32[1, 1024, 512]" = torch.ops.aten.view.default(permute_565, [1, 1024, 512]);  permute_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_682: "f32[1024, 512]" = torch.ops.aten.view.default(view_681, [1024, 512]);  view_681 = None
    permute_566: "f32[512, 1024]" = torch.ops.aten.permute.default(view_682, [1, 0])
    mm_227: "f32[512, 512]" = torch.ops.aten.mm.default(permute_566, view_124);  permute_566 = view_124 = None
    permute_567: "f32[512, 512]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    permute_568: "f32[512, 512]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_228: "f32[1024, 512]" = torch.ops.aten.mm.default(view_682, permute_568);  view_682 = permute_568 = None
    view_683: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_228, [1, 1024, 512]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_182: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_680, view_683);  view_680 = view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_569: "f32[512, 512]" = torch.ops.aten.permute.default(permute_567, [1, 0]);  permute_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_570: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_677, [0, 2, 1, 3]);  view_677 = None
    clone_112: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_570, memory_format = torch.contiguous_format);  permute_570 = None
    view_684: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_112, [1, 1024, 512]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_685: "f32[1024, 512]" = torch.ops.aten.view.default(view_684, [1024, 512]);  view_684 = None
    permute_571: "f32[512, 1024]" = torch.ops.aten.permute.default(view_685, [1, 0])
    mm_229: "f32[512, 512]" = torch.ops.aten.mm.default(permute_571, view_121);  permute_571 = view_121 = None
    permute_572: "f32[512, 512]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    permute_573: "f32[512, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_230: "f32[1024, 512]" = torch.ops.aten.mm.default(view_685, permute_573);  view_685 = permute_573 = None
    view_686: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_230, [1, 1024, 512]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_183: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_182, view_686);  add_182 = view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_574: "f32[512, 512]" = torch.ops.aten.permute.default(permute_572, [1, 0]);  permute_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_355: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_183, primals_11);  primals_11 = None
    mul_356: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_183, mul_23);  add_183 = mul_23 = None
    sum_78: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 1], True);  mul_356 = None
    view_687: "f32[512]" = torch.ops.aten.view.default(sum_78, [512]);  sum_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_357: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_355, add_28)
    mul_358: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_355, rsqrt_10);  mul_355 = rsqrt_10 = None
    sum_79: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_184: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_181, mul_358);  add_181 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_105: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    pow_75: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_105, 3);  alias_105 = None
    mul_359: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_79, -0.5);  sum_79 = None
    mul_360: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_359, pow_75);  mul_359 = pow_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_93: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_360, [1, 1024, 512]);  mul_360 = None
    div_45: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_93, 512);  expand_93 = None
    pow_76: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_28, 1.0);  add_28 = None
    mul_361: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_76, 2.0);  pow_76 = None
    mul_362: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_45, mul_361);  div_45 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_185: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_184, mul_362);  add_184 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_51: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_363: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
    mul_364: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_185, mul_363);  mul_363 = None
    clone_113: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_364, memory_format = torch.contiguous_format);  mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_688: "f32[1024, 512]" = torch.ops.aten.view.default(clone_113, [1024, 512]);  clone_113 = None
    permute_575: "f32[512, 1024]" = torch.ops.aten.permute.default(view_688, [1, 0])
    mm_231: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_575, view_119);  permute_575 = view_119 = None
    permute_576: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    permute_577: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_232: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_688, permute_577);  view_688 = permute_577 = None
    view_689: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_232, [1, 1024, 2048]);  mm_232 = None
    permute_578: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_576, [1, 0]);  permute_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_52: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_39, torch.float32);  getitem_39 = None
    mul_365: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
    mul_366: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_689, mul_365);  view_689 = mul_365 = None
    clone_114: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_366, memory_format = torch.contiguous_format);  mul_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_106: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    le_8: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_8, scalar_tensor_13, clone_114);  le_8 = scalar_tensor_13 = clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_690: "f32[1024, 2048]" = torch.ops.aten.view.default(where_15, [1024, 2048]);  where_15 = None
    permute_579: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_690, [1, 0])
    mm_233: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_579, view_117);  permute_579 = view_117 = None
    permute_580: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    permute_581: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_234: "f32[1024, 512]" = torch.ops.aten.mm.default(view_690, permute_581);  view_690 = permute_581 = None
    view_691: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_234, [1, 1024, 512]);  mm_234 = None
    permute_582: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_580, [1, 0]);  permute_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_367: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_691, primals_10);  primals_10 = None
    mul_368: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_691, mul_21);  view_691 = mul_21 = None
    sum_80: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1], True);  mul_368 = None
    view_692: "f32[512]" = torch.ops.aten.view.default(sum_80, [512]);  sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_369: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_367, add_26)
    mul_370: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_367, rsqrt_9);  mul_367 = rsqrt_9 = None
    sum_81: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_186: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_185, mul_370);  add_185 = mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_107: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    pow_77: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_107, 3);  alias_107 = None
    mul_371: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_81, -0.5);  sum_81 = None
    mul_372: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_371, pow_77);  mul_371 = pow_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_94: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_372, [1, 1024, 512]);  mul_372 = None
    div_46: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_94, 512);  expand_94 = None
    pow_78: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 1.0);  add_26 = None
    mul_373: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_78, 2.0);  pow_78 = None
    mul_374: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_46, mul_373);  div_46 = mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_187: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_186, mul_374);  add_186 = mul_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_53: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_375: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
    mul_376: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_187, mul_375);  mul_375 = None
    clone_115: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_376, memory_format = torch.contiguous_format);  mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_693: "f32[1024, 512]" = torch.ops.aten.view.default(clone_115, [1024, 512]);  clone_115 = None
    permute_583: "f32[512, 1024]" = torch.ops.aten.permute.default(view_693, [1, 0])
    mm_235: "f32[512, 512]" = torch.ops.aten.mm.default(permute_583, view_115);  permute_583 = view_115 = None
    permute_584: "f32[512, 512]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    permute_585: "f32[512, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_236: "f32[1024, 512]" = torch.ops.aten.mm.default(view_693, permute_585);  view_693 = permute_585 = None
    view_694: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_236, [1, 1024, 512]);  mm_236 = None
    permute_586: "f32[512, 512]" = torch.ops.aten.permute.default(permute_584, [1, 0]);  permute_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_695: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_694, [1, 1024, 8, 64]);  view_694 = None
    permute_587: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_696: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_587, [8, 1024, 64]);  permute_587 = None
    permute_588: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    bmm_88: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_588, view_696);  permute_588 = None
    permute_589: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    bmm_89: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_696, permute_589);  view_696 = permute_589 = None
    view_697: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_88, [1, 8, 1024, 64]);  bmm_88 = None
    view_698: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_89, [1, 8, 1024, 1024]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_54: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_377: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
    mul_378: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_698, mul_377);  view_698 = mul_377 = None
    clone_116: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_378, memory_format = torch.contiguous_format);  mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_108: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_379: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_116, alias_108);  clone_116 = None
    sum_82: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [-1], True)
    mul_380: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_108, sum_82);  alias_108 = sum_82 = None
    sub_39: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_15: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_39, 0);  sub_39 = None
    full_23: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_91: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_23, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_39: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_91, squeeze_15);  as_strided_91 = squeeze_15 = None
    as_strided_scatter_26: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_23, copy_39, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_23 = copy_39 = None
    as_strided_94: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_26, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_26 = None
    new_empty_strided_13: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_94, [8, 1024, 1024], [1048576, 1024, 1])
    copy_40: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_13, as_strided_94);  new_empty_strided_13 = as_strided_94 = None
    as_strided_96: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_40, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_117: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_96, memory_format = torch.contiguous_format)
    copy_41: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_96, clone_117);  as_strided_96 = None
    as_strided_scatter_27: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_40, copy_41, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_40 = copy_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_188: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(clone_110, clone_117);  clone_110 = clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_590: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
    bmm_90: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_590, as_strided_scatter_27);  permute_590 = None
    permute_591: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    bmm_91: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_27, permute_591);  as_strided_scatter_27 = permute_591 = None
    view_699: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_90, [1, 8, 64, 1024]);  bmm_90 = None
    view_700: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_91, [1, 8, 1024, 64]);  bmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_592: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_699, [0, 1, 3, 2]);  view_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_593: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_697, [0, 2, 1, 3]);  view_697 = None
    clone_118: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_593, memory_format = torch.contiguous_format);  permute_593 = None
    view_701: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_118, [1, 1024, 512]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_702: "f32[1024, 512]" = torch.ops.aten.view.default(view_701, [1024, 512]);  view_701 = None
    permute_594: "f32[512, 1024]" = torch.ops.aten.permute.default(view_702, [1, 0])
    mm_237: "f32[512, 512]" = torch.ops.aten.mm.default(permute_594, view_103);  permute_594 = view_103 = None
    permute_595: "f32[512, 512]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    permute_596: "f32[512, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_238: "f32[1024, 512]" = torch.ops.aten.mm.default(view_702, permute_596);  view_702 = permute_596 = None
    view_703: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_238, [1, 1024, 512]);  mm_238 = None
    permute_597: "f32[512, 512]" = torch.ops.aten.permute.default(permute_595, [1, 0]);  permute_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_598: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_592, [0, 2, 1, 3]);  permute_592 = None
    view_704: "f32[1, 1024, 512]" = torch.ops.aten.view.default(permute_598, [1, 1024, 512]);  permute_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_705: "f32[1024, 512]" = torch.ops.aten.view.default(view_704, [1024, 512]);  view_704 = None
    permute_599: "f32[512, 1024]" = torch.ops.aten.permute.default(view_705, [1, 0])
    mm_239: "f32[512, 512]" = torch.ops.aten.mm.default(permute_599, view_100);  permute_599 = view_100 = None
    permute_600: "f32[512, 512]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    permute_601: "f32[512, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_240: "f32[1024, 512]" = torch.ops.aten.mm.default(view_705, permute_601);  view_705 = permute_601 = None
    view_706: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_240, [1, 1024, 512]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_189: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_703, view_706);  view_703 = view_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_602: "f32[512, 512]" = torch.ops.aten.permute.default(permute_600, [1, 0]);  permute_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_603: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
    clone_119: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
    view_707: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_119, [1, 1024, 512]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_708: "f32[1024, 512]" = torch.ops.aten.view.default(view_707, [1024, 512]);  view_707 = None
    permute_604: "f32[512, 1024]" = torch.ops.aten.permute.default(view_708, [1, 0])
    mm_241: "f32[512, 512]" = torch.ops.aten.mm.default(permute_604, view_97);  permute_604 = view_97 = None
    permute_605: "f32[512, 512]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    permute_606: "f32[512, 512]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_242: "f32[1024, 512]" = torch.ops.aten.mm.default(view_708, permute_606);  view_708 = permute_606 = None
    view_709: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_242, [1, 1024, 512]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_190: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_189, view_709);  add_189 = view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_607: "f32[512, 512]" = torch.ops.aten.permute.default(permute_605, [1, 0]);  permute_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_381: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_190, primals_9);  primals_9 = None
    mul_382: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_190, mul_19);  add_190 = mul_19 = None
    sum_83: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1], True);  mul_382 = None
    view_710: "f32[512]" = torch.ops.aten.view.default(sum_83, [512]);  sum_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_383: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_381, add_23)
    mul_384: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_381, rsqrt_8);  mul_381 = rsqrt_8 = None
    sum_84: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_191: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_187, mul_384);  add_187 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_109: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    pow_79: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_109, 3);  alias_109 = None
    mul_385: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_84, -0.5);  sum_84 = None
    mul_386: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_385, pow_79);  mul_385 = pow_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_95: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_386, [1, 1024, 512]);  mul_386 = None
    div_47: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_95, 512);  expand_95 = None
    pow_80: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 1.0);  add_23 = None
    mul_387: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_80, 2.0);  pow_80 = None
    mul_388: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_47, mul_387);  div_47 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_192: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_191, mul_388);  add_191 = mul_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_55: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_33, torch.float32);  getitem_33 = None
    mul_389: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
    mul_390: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_192, mul_389);  mul_389 = None
    clone_120: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_390, memory_format = torch.contiguous_format);  mul_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_711: "f32[1024, 512]" = torch.ops.aten.view.default(clone_120, [1024, 512]);  clone_120 = None
    permute_608: "f32[512, 1024]" = torch.ops.aten.permute.default(view_711, [1, 0])
    mm_243: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_608, view_95);  permute_608 = view_95 = None
    permute_609: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    permute_610: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_244: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_711, permute_610);  view_711 = permute_610 = None
    view_712: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_244, [1, 1024, 2048]);  mm_244 = None
    permute_611: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_609, [1, 0]);  permute_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_56: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_391: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
    mul_392: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_712, mul_391);  view_712 = mul_391 = None
    clone_121: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_392, memory_format = torch.contiguous_format);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_110: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    le_9: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_110, 0);  alias_110 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_9, scalar_tensor_14, clone_121);  le_9 = scalar_tensor_14 = clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_713: "f32[1024, 2048]" = torch.ops.aten.view.default(where_16, [1024, 2048]);  where_16 = None
    permute_612: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_713, [1, 0])
    mm_245: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_612, view_93);  permute_612 = view_93 = None
    permute_613: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    permute_614: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_246: "f32[1024, 512]" = torch.ops.aten.mm.default(view_713, permute_614);  view_713 = permute_614 = None
    view_714: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_246, [1, 1024, 512]);  mm_246 = None
    permute_615: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_613, [1, 0]);  permute_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_393: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_714, primals_8);  primals_8 = None
    mul_394: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_714, mul_17);  view_714 = mul_17 = None
    sum_85: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_394, [0, 1], True);  mul_394 = None
    view_715: "f32[512]" = torch.ops.aten.view.default(sum_85, [512]);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_395: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_393, add_21)
    mul_396: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_393, rsqrt_7);  mul_393 = rsqrt_7 = None
    sum_86: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True);  mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_193: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_192, mul_396);  add_192 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_111: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    pow_81: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_111, 3);  alias_111 = None
    mul_397: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_86, -0.5);  sum_86 = None
    mul_398: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_397, pow_81);  mul_397 = pow_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_96: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_398, [1, 1024, 512]);  mul_398 = None
    div_48: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_96, 512);  expand_96 = None
    pow_82: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 1.0);  add_21 = None
    mul_399: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_82, 2.0);  pow_82 = None
    mul_400: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_48, mul_399);  div_48 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_194: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_193, mul_400);  add_193 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_57: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_29, torch.float32);  getitem_29 = None
    mul_401: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
    mul_402: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_194, mul_401);  mul_401 = None
    clone_122: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_402, memory_format = torch.contiguous_format);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_716: "f32[1024, 512]" = torch.ops.aten.view.default(clone_122, [1024, 512]);  clone_122 = None
    permute_616: "f32[512, 1024]" = torch.ops.aten.permute.default(view_716, [1, 0])
    mm_247: "f32[512, 512]" = torch.ops.aten.mm.default(permute_616, view_91);  permute_616 = view_91 = None
    permute_617: "f32[512, 512]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    permute_618: "f32[512, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_248: "f32[1024, 512]" = torch.ops.aten.mm.default(view_716, permute_618);  view_716 = permute_618 = None
    view_717: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_248, [1, 1024, 512]);  mm_248 = None
    permute_619: "f32[512, 512]" = torch.ops.aten.permute.default(permute_617, [1, 0]);  permute_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_718: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_717, [1, 1024, 8, 64]);  view_717 = None
    permute_620: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_719: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_620, [8, 1024, 64]);  permute_620 = None
    permute_621: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
    bmm_92: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_621, view_719);  permute_621 = None
    permute_622: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
    bmm_93: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_719, permute_622);  view_719 = permute_622 = None
    view_720: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_92, [1, 8, 1024, 64]);  bmm_92 = None
    view_721: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_93, [1, 8, 1024, 1024]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_58: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_403: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
    mul_404: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_721, mul_403);  view_721 = mul_403 = None
    clone_123: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_404, memory_format = torch.contiguous_format);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_112: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_405: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_123, alias_112);  clone_123 = None
    sum_87: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [-1], True)
    mul_406: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_112, sum_87);  alias_112 = sum_87 = None
    sub_40: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_405, mul_406);  mul_405 = mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_16: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_40, 0);  sub_40 = None
    full_24: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_98: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_24, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_42: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_98, squeeze_16);  as_strided_98 = squeeze_16 = None
    as_strided_scatter_28: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_24, copy_42, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_24 = copy_42 = None
    as_strided_101: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_28, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_28 = None
    new_empty_strided_14: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_101, [8, 1024, 1024], [1048576, 1024, 1])
    copy_43: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_14, as_strided_101);  new_empty_strided_14 = as_strided_101 = None
    as_strided_103: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_43, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_124: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_103, memory_format = torch.contiguous_format)
    copy_44: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_103, clone_124);  as_strided_103 = None
    as_strided_scatter_29: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_43, copy_44, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_43 = copy_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_195: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_188, clone_124);  add_188 = clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_623: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    bmm_94: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_623, as_strided_scatter_29);  permute_623 = None
    permute_624: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    bmm_95: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_29, permute_624);  as_strided_scatter_29 = permute_624 = None
    view_722: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_94, [1, 8, 64, 1024]);  bmm_94 = None
    view_723: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_95, [1, 8, 1024, 64]);  bmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_625: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_722, [0, 1, 3, 2]);  view_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_626: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
    clone_125: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_626, memory_format = torch.contiguous_format);  permute_626 = None
    view_724: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_125, [1, 1024, 512]);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_725: "f32[1024, 512]" = torch.ops.aten.view.default(view_724, [1024, 512]);  view_724 = None
    permute_627: "f32[512, 1024]" = torch.ops.aten.permute.default(view_725, [1, 0])
    mm_249: "f32[512, 512]" = torch.ops.aten.mm.default(permute_627, view_79);  permute_627 = view_79 = None
    permute_628: "f32[512, 512]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    permute_629: "f32[512, 512]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    mm_250: "f32[1024, 512]" = torch.ops.aten.mm.default(view_725, permute_629);  view_725 = permute_629 = None
    view_726: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_250, [1, 1024, 512]);  mm_250 = None
    permute_630: "f32[512, 512]" = torch.ops.aten.permute.default(permute_628, [1, 0]);  permute_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_631: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_625, [0, 2, 1, 3]);  permute_625 = None
    view_727: "f32[1, 1024, 512]" = torch.ops.aten.view.default(permute_631, [1, 1024, 512]);  permute_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_728: "f32[1024, 512]" = torch.ops.aten.view.default(view_727, [1024, 512]);  view_727 = None
    permute_632: "f32[512, 1024]" = torch.ops.aten.permute.default(view_728, [1, 0])
    mm_251: "f32[512, 512]" = torch.ops.aten.mm.default(permute_632, view_76);  permute_632 = view_76 = None
    permute_633: "f32[512, 512]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    permute_634: "f32[512, 512]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_252: "f32[1024, 512]" = torch.ops.aten.mm.default(view_728, permute_634);  view_728 = permute_634 = None
    view_729: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_252, [1, 1024, 512]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_196: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_726, view_729);  view_726 = view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_635: "f32[512, 512]" = torch.ops.aten.permute.default(permute_633, [1, 0]);  permute_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_636: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None
    clone_126: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_636, memory_format = torch.contiguous_format);  permute_636 = None
    view_730: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_126, [1, 1024, 512]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_731: "f32[1024, 512]" = torch.ops.aten.view.default(view_730, [1024, 512]);  view_730 = None
    permute_637: "f32[512, 1024]" = torch.ops.aten.permute.default(view_731, [1, 0])
    mm_253: "f32[512, 512]" = torch.ops.aten.mm.default(permute_637, view_73);  permute_637 = view_73 = None
    permute_638: "f32[512, 512]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    permute_639: "f32[512, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_254: "f32[1024, 512]" = torch.ops.aten.mm.default(view_731, permute_639);  view_731 = permute_639 = None
    view_732: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_254, [1, 1024, 512]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_197: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_196, view_732);  add_196 = view_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_640: "f32[512, 512]" = torch.ops.aten.permute.default(permute_638, [1, 0]);  permute_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_407: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_197, primals_7);  primals_7 = None
    mul_408: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_197, mul_15);  add_197 = mul_15 = None
    sum_88: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 1], True);  mul_408 = None
    view_733: "f32[512]" = torch.ops.aten.view.default(sum_88, [512]);  sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_409: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_407, add_18)
    mul_410: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_407, rsqrt_6);  mul_407 = rsqrt_6 = None
    sum_89: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True);  mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_198: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_194, mul_410);  add_194 = mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_113: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    pow_83: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_113, 3);  alias_113 = None
    mul_411: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_89, -0.5);  sum_89 = None
    mul_412: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_411, pow_83);  mul_411 = pow_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_97: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_412, [1, 1024, 512]);  mul_412 = None
    div_49: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_97, 512);  expand_97 = None
    pow_84: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_18, 1.0);  add_18 = None
    mul_413: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
    mul_414: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_49, mul_413);  div_49 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_199: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_198, mul_414);  add_198 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_59: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_415: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
    mul_416: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_199, mul_415);  mul_415 = None
    clone_127: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_416, memory_format = torch.contiguous_format);  mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_734: "f32[1024, 512]" = torch.ops.aten.view.default(clone_127, [1024, 512]);  clone_127 = None
    permute_641: "f32[512, 1024]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_255: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_641, view_71);  permute_641 = view_71 = None
    permute_642: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    permute_643: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_256: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_734, permute_643);  view_734 = permute_643 = None
    view_735: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_256, [1, 1024, 2048]);  mm_256 = None
    permute_644: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_642, [1, 0]);  permute_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_60: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
    mul_417: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
    mul_418: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_735, mul_417);  view_735 = mul_417 = None
    clone_128: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_418, memory_format = torch.contiguous_format);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_114: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    le_10: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_114, 0);  alias_114 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_10, scalar_tensor_15, clone_128);  le_10 = scalar_tensor_15 = clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_736: "f32[1024, 2048]" = torch.ops.aten.view.default(where_17, [1024, 2048]);  where_17 = None
    permute_645: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_736, [1, 0])
    mm_257: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_645, view_69);  permute_645 = view_69 = None
    permute_646: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    permute_647: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_258: "f32[1024, 512]" = torch.ops.aten.mm.default(view_736, permute_647);  view_736 = permute_647 = None
    view_737: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_258, [1, 1024, 512]);  mm_258 = None
    permute_648: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_646, [1, 0]);  permute_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_419: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_737, primals_6);  primals_6 = None
    mul_420: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_737, mul_13);  view_737 = mul_13 = None
    sum_90: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 1], True);  mul_420 = None
    view_738: "f32[512]" = torch.ops.aten.view.default(sum_90, [512]);  sum_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_421: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_419, add_16)
    mul_422: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_419, rsqrt_5);  mul_419 = rsqrt_5 = None
    sum_91: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_200: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_199, mul_422);  add_199 = mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_115: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    pow_85: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_115, 3);  alias_115 = None
    mul_423: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_91, -0.5);  sum_91 = None
    mul_424: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_423, pow_85);  mul_423 = pow_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_98: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_424, [1, 1024, 512]);  mul_424 = None
    div_50: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_98, 512);  expand_98 = None
    pow_86: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 1.0);  add_16 = None
    mul_425: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
    mul_426: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_50, mul_425);  div_50 = mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_201: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_200, mul_426);  add_200 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_61: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_427: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
    mul_428: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_201, mul_427);  mul_427 = None
    clone_129: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_428, memory_format = torch.contiguous_format);  mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_739: "f32[1024, 512]" = torch.ops.aten.view.default(clone_129, [1024, 512]);  clone_129 = None
    permute_649: "f32[512, 1024]" = torch.ops.aten.permute.default(view_739, [1, 0])
    mm_259: "f32[512, 512]" = torch.ops.aten.mm.default(permute_649, view_67);  permute_649 = view_67 = None
    permute_650: "f32[512, 512]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    permute_651: "f32[512, 512]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_260: "f32[1024, 512]" = torch.ops.aten.mm.default(view_739, permute_651);  view_739 = permute_651 = None
    view_740: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_260, [1, 1024, 512]);  mm_260 = None
    permute_652: "f32[512, 512]" = torch.ops.aten.permute.default(permute_650, [1, 0]);  permute_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_741: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_740, [1, 1024, 8, 64]);  view_740 = None
    permute_653: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_741, [0, 2, 1, 3]);  view_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_742: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_653, [8, 1024, 64]);  permute_653 = None
    permute_654: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    bmm_96: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_654, view_742);  permute_654 = None
    permute_655: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    bmm_97: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_742, permute_655);  view_742 = permute_655 = None
    view_743: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_96, [1, 8, 1024, 64]);  bmm_96 = None
    view_744: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_97, [1, 8, 1024, 1024]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_62: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
    mul_429: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
    mul_430: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_744, mul_429);  view_744 = mul_429 = None
    clone_130: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_430, memory_format = torch.contiguous_format);  mul_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_116: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_431: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_130, alias_116);  clone_130 = None
    sum_92: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [-1], True)
    mul_432: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_116, sum_92);  alias_116 = sum_92 = None
    sub_41: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_17: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_41, 0);  sub_41 = None
    full_25: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_105: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_25, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_45: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_105, squeeze_17);  as_strided_105 = squeeze_17 = None
    as_strided_scatter_30: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_25, copy_45, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_25 = copy_45 = None
    as_strided_108: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_30, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_30 = None
    new_empty_strided_15: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_108, [8, 1024, 1024], [1048576, 1024, 1])
    copy_46: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_15, as_strided_108);  new_empty_strided_15 = as_strided_108 = None
    as_strided_110: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_46, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_131: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_110, memory_format = torch.contiguous_format)
    copy_47: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_110, clone_131);  as_strided_110 = None
    as_strided_scatter_31: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_46, copy_47, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_46 = copy_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_202: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_195, clone_131);  add_195 = clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_656: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_98: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_656, as_strided_scatter_31);  permute_656 = None
    permute_657: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_99: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_31, permute_657);  as_strided_scatter_31 = permute_657 = None
    view_745: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_98, [1, 8, 64, 1024]);  bmm_98 = None
    view_746: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_99, [1, 8, 1024, 64]);  bmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_658: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_745, [0, 1, 3, 2]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_659: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_743, [0, 2, 1, 3]);  view_743 = None
    clone_132: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_659, memory_format = torch.contiguous_format);  permute_659 = None
    view_747: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_132, [1, 1024, 512]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_748: "f32[1024, 512]" = torch.ops.aten.view.default(view_747, [1024, 512]);  view_747 = None
    permute_660: "f32[512, 1024]" = torch.ops.aten.permute.default(view_748, [1, 0])
    mm_261: "f32[512, 512]" = torch.ops.aten.mm.default(permute_660, view_55);  permute_660 = view_55 = None
    permute_661: "f32[512, 512]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    permute_662: "f32[512, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm_262: "f32[1024, 512]" = torch.ops.aten.mm.default(view_748, permute_662);  view_748 = permute_662 = None
    view_749: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_262, [1, 1024, 512]);  mm_262 = None
    permute_663: "f32[512, 512]" = torch.ops.aten.permute.default(permute_661, [1, 0]);  permute_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_664: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_658, [0, 2, 1, 3]);  permute_658 = None
    view_750: "f32[1, 1024, 512]" = torch.ops.aten.view.default(permute_664, [1, 1024, 512]);  permute_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_751: "f32[1024, 512]" = torch.ops.aten.view.default(view_750, [1024, 512]);  view_750 = None
    permute_665: "f32[512, 1024]" = torch.ops.aten.permute.default(view_751, [1, 0])
    mm_263: "f32[512, 512]" = torch.ops.aten.mm.default(permute_665, view_52);  permute_665 = view_52 = None
    permute_666: "f32[512, 512]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    permute_667: "f32[512, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_264: "f32[1024, 512]" = torch.ops.aten.mm.default(view_751, permute_667);  view_751 = permute_667 = None
    view_752: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_264, [1, 1024, 512]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_203: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_749, view_752);  view_749 = view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_668: "f32[512, 512]" = torch.ops.aten.permute.default(permute_666, [1, 0]);  permute_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_669: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None
    clone_133: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_669, memory_format = torch.contiguous_format);  permute_669 = None
    view_753: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_133, [1, 1024, 512]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_754: "f32[1024, 512]" = torch.ops.aten.view.default(view_753, [1024, 512]);  view_753 = None
    permute_670: "f32[512, 1024]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_265: "f32[512, 512]" = torch.ops.aten.mm.default(permute_670, view_49);  permute_670 = view_49 = None
    permute_671: "f32[512, 512]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    permute_672: "f32[512, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_266: "f32[1024, 512]" = torch.ops.aten.mm.default(view_754, permute_672);  view_754 = permute_672 = None
    view_755: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_266, [1, 1024, 512]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_204: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_203, view_755);  add_203 = view_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_673: "f32[512, 512]" = torch.ops.aten.permute.default(permute_671, [1, 0]);  permute_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_433: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_204, primals_5);  primals_5 = None
    mul_434: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_204, mul_11);  add_204 = mul_11 = None
    sum_93: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 1], True);  mul_434 = None
    view_756: "f32[512]" = torch.ops.aten.view.default(sum_93, [512]);  sum_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_435: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_433, add_13)
    mul_436: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_433, rsqrt_4);  mul_433 = rsqrt_4 = None
    sum_94: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [2], True);  mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_205: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_201, mul_436);  add_201 = mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_117: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    pow_87: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_117, 3);  alias_117 = None
    mul_437: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_94, -0.5);  sum_94 = None
    mul_438: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_437, pow_87);  mul_437 = pow_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_99: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_438, [1, 1024, 512]);  mul_438 = None
    div_51: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_99, 512);  expand_99 = None
    pow_88: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 1.0);  add_13 = None
    mul_439: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_88, 2.0);  pow_88 = None
    mul_440: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_51, mul_439);  div_51 = mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_206: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_205, mul_440);  add_205 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_63: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_441: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
    mul_442: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_206, mul_441);  mul_441 = None
    clone_134: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_442, memory_format = torch.contiguous_format);  mul_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_757: "f32[1024, 512]" = torch.ops.aten.view.default(clone_134, [1024, 512]);  clone_134 = None
    permute_674: "f32[512, 1024]" = torch.ops.aten.permute.default(view_757, [1, 0])
    mm_267: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_674, view_47);  permute_674 = view_47 = None
    permute_675: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    permute_676: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_268: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_757, permute_676);  view_757 = permute_676 = None
    view_758: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_268, [1, 1024, 2048]);  mm_268 = None
    permute_677: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_675, [1, 0]);  permute_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_64: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
    mul_443: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
    mul_444: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_758, mul_443);  view_758 = mul_443 = None
    clone_135: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_444, memory_format = torch.contiguous_format);  mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_118: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    le_11: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_11, scalar_tensor_16, clone_135);  le_11 = scalar_tensor_16 = clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_759: "f32[1024, 2048]" = torch.ops.aten.view.default(where_18, [1024, 2048]);  where_18 = None
    permute_678: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_759, [1, 0])
    mm_269: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_678, view_45);  permute_678 = view_45 = None
    permute_679: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    permute_680: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_270: "f32[1024, 512]" = torch.ops.aten.mm.default(view_759, permute_680);  view_759 = permute_680 = None
    view_760: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_270, [1, 1024, 512]);  mm_270 = None
    permute_681: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_679, [1, 0]);  permute_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_445: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_760, primals_4);  primals_4 = None
    mul_446: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_760, mul_9);  view_760 = mul_9 = None
    sum_95: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1], True);  mul_446 = None
    view_761: "f32[512]" = torch.ops.aten.view.default(sum_95, [512]);  sum_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_447: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_445, add_11)
    mul_448: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_445, rsqrt_3);  mul_445 = rsqrt_3 = None
    sum_96: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_207: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_206, mul_448);  add_206 = mul_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_119: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    pow_89: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_119, 3);  alias_119 = None
    mul_449: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_96, -0.5);  sum_96 = None
    mul_450: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_449, pow_89);  mul_449 = pow_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_100: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_450, [1, 1024, 512]);  mul_450 = None
    div_52: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_100, 512);  expand_100 = None
    pow_90: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 1.0);  add_11 = None
    mul_451: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_90, 2.0);  pow_90 = None
    mul_452: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_52, mul_451);  div_52 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_208: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_207, mul_452);  add_207 = mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_65: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
    mul_453: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
    mul_454: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_208, mul_453);  mul_453 = None
    clone_136: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_454, memory_format = torch.contiguous_format);  mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_762: "f32[1024, 512]" = torch.ops.aten.view.default(clone_136, [1024, 512]);  clone_136 = None
    permute_682: "f32[512, 1024]" = torch.ops.aten.permute.default(view_762, [1, 0])
    mm_271: "f32[512, 512]" = torch.ops.aten.mm.default(permute_682, view_43);  permute_682 = view_43 = None
    permute_683: "f32[512, 512]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    permute_684: "f32[512, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_272: "f32[1024, 512]" = torch.ops.aten.mm.default(view_762, permute_684);  view_762 = permute_684 = None
    view_763: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_272, [1, 1024, 512]);  mm_272 = None
    permute_685: "f32[512, 512]" = torch.ops.aten.permute.default(permute_683, [1, 0]);  permute_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_764: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_763, [1, 1024, 8, 64]);  view_763 = None
    permute_686: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_764, [0, 2, 1, 3]);  view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_765: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_686, [8, 1024, 64]);  permute_686 = None
    permute_687: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_39, [0, 2, 1]);  view_39 = None
    bmm_100: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_687, view_765);  permute_687 = None
    permute_688: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_40, [0, 2, 1]);  view_40 = None
    bmm_101: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_765, permute_688);  view_765 = permute_688 = None
    view_766: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_100, [1, 8, 1024, 64]);  bmm_100 = None
    view_767: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_101, [1, 8, 1024, 1024]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_66: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_455: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
    mul_456: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_767, mul_455);  view_767 = mul_455 = None
    clone_137: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_456, memory_format = torch.contiguous_format);  mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_120: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_457: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_137, alias_120);  clone_137 = None
    sum_97: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_457, [-1], True)
    mul_458: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_120, sum_97);  alias_120 = sum_97 = None
    sub_42: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_18: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_42, 0);  sub_42 = None
    full_26: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_112: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_26, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_48: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_112, squeeze_18);  as_strided_112 = squeeze_18 = None
    as_strided_scatter_32: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_26, copy_48, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_26 = copy_48 = None
    as_strided_115: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_32, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_32 = None
    new_empty_strided_16: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_115, [8, 1024, 1024], [1048576, 1024, 1])
    copy_49: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_16, as_strided_115);  new_empty_strided_16 = as_strided_115 = None
    as_strided_117: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_49, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_138: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_117, memory_format = torch.contiguous_format)
    copy_50: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_117, clone_138);  as_strided_117 = None
    as_strided_scatter_33: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_49, copy_50, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_49 = copy_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_209: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_202, clone_138);  add_202 = clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_689: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_102: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_689, as_strided_scatter_33);  permute_689 = None
    permute_690: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    bmm_103: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_33, permute_690);  as_strided_scatter_33 = permute_690 = None
    view_768: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_102, [1, 8, 64, 1024]);  bmm_102 = None
    view_769: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_103, [1, 8, 1024, 64]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_691: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_768, [0, 1, 3, 2]);  view_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_692: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
    clone_139: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_692, memory_format = torch.contiguous_format);  permute_692 = None
    view_770: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_139, [1, 1024, 512]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_771: "f32[1024, 512]" = torch.ops.aten.view.default(view_770, [1024, 512]);  view_770 = None
    permute_693: "f32[512, 1024]" = torch.ops.aten.permute.default(view_771, [1, 0])
    mm_273: "f32[512, 512]" = torch.ops.aten.mm.default(permute_693, view_31);  permute_693 = view_31 = None
    permute_694: "f32[512, 512]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    permute_695: "f32[512, 512]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm_274: "f32[1024, 512]" = torch.ops.aten.mm.default(view_771, permute_695);  view_771 = permute_695 = None
    view_772: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_274, [1, 1024, 512]);  mm_274 = None
    permute_696: "f32[512, 512]" = torch.ops.aten.permute.default(permute_694, [1, 0]);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_697: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_691, [0, 2, 1, 3]);  permute_691 = None
    view_773: "f32[1, 1024, 512]" = torch.ops.aten.view.default(permute_697, [1, 1024, 512]);  permute_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_774: "f32[1024, 512]" = torch.ops.aten.view.default(view_773, [1024, 512]);  view_773 = None
    permute_698: "f32[512, 1024]" = torch.ops.aten.permute.default(view_774, [1, 0])
    mm_275: "f32[512, 512]" = torch.ops.aten.mm.default(permute_698, view_28);  permute_698 = view_28 = None
    permute_699: "f32[512, 512]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    permute_700: "f32[512, 512]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_276: "f32[1024, 512]" = torch.ops.aten.mm.default(view_774, permute_700);  view_774 = permute_700 = None
    view_775: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_276, [1, 1024, 512]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_210: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_772, view_775);  view_772 = view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_701: "f32[512, 512]" = torch.ops.aten.permute.default(permute_699, [1, 0]);  permute_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_702: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_769, [0, 2, 1, 3]);  view_769 = None
    clone_140: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
    view_776: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_140, [1, 1024, 512]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_777: "f32[1024, 512]" = torch.ops.aten.view.default(view_776, [1024, 512]);  view_776 = None
    permute_703: "f32[512, 1024]" = torch.ops.aten.permute.default(view_777, [1, 0])
    mm_277: "f32[512, 512]" = torch.ops.aten.mm.default(permute_703, view_25);  permute_703 = view_25 = None
    permute_704: "f32[512, 512]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    permute_705: "f32[512, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_278: "f32[1024, 512]" = torch.ops.aten.mm.default(view_777, permute_705);  view_777 = permute_705 = None
    view_778: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_278, [1, 1024, 512]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_211: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_210, view_778);  add_210 = view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_706: "f32[512, 512]" = torch.ops.aten.permute.default(permute_704, [1, 0]);  permute_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_459: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_211, primals_3);  primals_3 = None
    mul_460: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_211, mul_7);  add_211 = mul_7 = None
    sum_98: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_460, [0, 1], True);  mul_460 = None
    view_779: "f32[512]" = torch.ops.aten.view.default(sum_98, [512]);  sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_461: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_459, add_8)
    mul_462: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_459, rsqrt_2);  mul_459 = rsqrt_2 = None
    sum_99: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_212: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_208, mul_462);  add_208 = mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_121: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    pow_91: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_121, 3);  alias_121 = None
    mul_463: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_99, -0.5);  sum_99 = None
    mul_464: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_463, pow_91);  mul_463 = pow_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_101: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_464, [1, 1024, 512]);  mul_464 = None
    div_53: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_101, 512);  expand_101 = None
    pow_92: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_8, 1.0);  add_8 = None
    mul_465: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_92, 2.0);  pow_92 = None
    mul_466: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_53, mul_465);  div_53 = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_213: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_212, mul_466);  add_212 = mul_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_67: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_9, torch.float32);  getitem_9 = None
    mul_467: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
    mul_468: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_213, mul_467);  mul_467 = None
    clone_141: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_468, memory_format = torch.contiguous_format);  mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_780: "f32[1024, 512]" = torch.ops.aten.view.default(clone_141, [1024, 512]);  clone_141 = None
    permute_707: "f32[512, 1024]" = torch.ops.aten.permute.default(view_780, [1, 0])
    mm_279: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_707, view_23);  permute_707 = view_23 = None
    permute_708: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    permute_709: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_280: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_780, permute_709);  view_780 = permute_709 = None
    view_781: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_280, [1, 1024, 2048]);  mm_280 = None
    permute_710: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_708, [1, 0]);  permute_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_68: "f32[1, 1024, 2048]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_469: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
    mul_470: "f32[1, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_781, mul_469);  view_781 = mul_469 = None
    clone_142: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(mul_470, memory_format = torch.contiguous_format);  mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_122: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    le_12: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_122, 0);  alias_122 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[1, 1024, 2048]" = torch.ops.aten.where.self(le_12, scalar_tensor_17, clone_142);  le_12 = scalar_tensor_17 = clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_782: "f32[1024, 2048]" = torch.ops.aten.view.default(where_19, [1024, 2048]);  where_19 = None
    permute_711: "f32[2048, 1024]" = torch.ops.aten.permute.default(view_782, [1, 0])
    mm_281: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_711, view_21);  permute_711 = view_21 = None
    permute_712: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    permute_713: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_282: "f32[1024, 512]" = torch.ops.aten.mm.default(view_782, permute_713);  view_782 = permute_713 = None
    view_783: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_282, [1, 1024, 512]);  mm_282 = None
    permute_714: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_712, [1, 0]);  permute_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_471: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_783, primals_2);  primals_2 = None
    mul_472: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(view_783, mul_5);  view_783 = mul_5 = None
    sum_100: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1], True);  mul_472 = None
    view_784: "f32[512]" = torch.ops.aten.view.default(sum_100, [512]);  sum_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_473: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_471, add_6)
    mul_474: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_471, rsqrt_1);  mul_471 = rsqrt_1 = None
    sum_101: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True);  mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_214: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_213, mul_474);  add_213 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_123: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    pow_93: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_123, 3);  alias_123 = None
    mul_475: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_101, -0.5);  sum_101 = None
    mul_476: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_475, pow_93);  mul_475 = pow_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_102: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_476, [1, 1024, 512]);  mul_476 = None
    div_54: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_102, 512);  expand_102 = None
    pow_94: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 1.0);  add_6 = None
    mul_477: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_94, 2.0);  pow_94 = None
    mul_478: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_54, mul_477);  div_54 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_215: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_214, mul_478);  add_214 = mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_69: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_479: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
    mul_480: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_215, mul_479);  mul_479 = None
    clone_143: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_480, memory_format = torch.contiguous_format);  mul_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_785: "f32[1024, 512]" = torch.ops.aten.view.default(clone_143, [1024, 512]);  clone_143 = None
    permute_715: "f32[512, 1024]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_283: "f32[512, 512]" = torch.ops.aten.mm.default(permute_715, view_19);  permute_715 = view_19 = None
    permute_716: "f32[512, 512]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    permute_717: "f32[512, 512]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_284: "f32[1024, 512]" = torch.ops.aten.mm.default(view_785, permute_717);  view_785 = permute_717 = None
    view_786: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_284, [1, 1024, 512]);  mm_284 = None
    permute_718: "f32[512, 512]" = torch.ops.aten.permute.default(permute_716, [1, 0]);  permute_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_787: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_786, [1, 1024, 8, 64]);  view_786 = None
    permute_719: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_787, [0, 2, 1, 3]);  view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_788: "f32[8, 1024, 64]" = torch.ops.aten.view.default(permute_719, [8, 1024, 64]);  permute_719 = None
    permute_720: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    bmm_104: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(permute_720, view_788);  permute_720 = None
    permute_721: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
    bmm_105: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_788, permute_721);  view_788 = permute_721 = None
    view_789: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_104, [1, 8, 1024, 64]);  bmm_104 = None
    view_790: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_105, [1, 8, 1024, 1024]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    convert_element_type_70: "f32[1, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_481: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
    mul_482: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_790, mul_481);  view_790 = mul_481 = None
    clone_144: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(mul_482, memory_format = torch.contiguous_format);  mul_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_124: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_483: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_144, alias_124);  clone_144 = None
    sum_102: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [-1], True)
    mul_484: "f32[1, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_124, sum_102);  alias_124 = sum_102 = None
    sub_43: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    squeeze_19: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sub_43, 0);  sub_43 = None
    full_27: "f32[8388608]" = torch.ops.aten.full.default([8388608], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_119: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_27, [8, 1024, 1024], [1048576, 1024, 1], 0)
    copy_51: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_119, squeeze_19);  as_strided_119 = squeeze_19 = None
    as_strided_scatter_34: "f32[8388608]" = torch.ops.aten.as_strided_scatter.default(full_27, copy_51, [8, 1024, 1024], [1048576, 1024, 1], 0);  full_27 = copy_51 = None
    as_strided_122: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_34, [8, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_34 = None
    new_empty_strided_17: "f32[8, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_122, [8, 1024, 1024], [1048576, 1024, 1])
    copy_52: "f32[8, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_17, as_strided_122);  new_empty_strided_17 = as_strided_122 = None
    as_strided_124: "f32[1, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_52, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_145: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_124, memory_format = torch.contiguous_format)
    copy_53: "f32[1, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_124, clone_145);  as_strided_124 = None
    as_strided_scatter_35: "f32[8, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_52, copy_53, [1, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_52 = copy_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_216: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_209, clone_145);  add_209 = clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    squeeze_20: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(add_216, 0);  add_216 = None
    permute_722: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_20, [1, 2, 0]);  squeeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    eq_2: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_3, -1)
    unsqueeze_21: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_21, scalar_tensor_18, permute_722);  unsqueeze_21 = scalar_tensor_18 = permute_722 = None
    clone_146: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_20, memory_format = torch.contiguous_format);  where_20 = None
    full_28: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[32, 8]" = torch.ops.aten._unsafe_index_put.default(full_28, [add_3], clone_146, True);  full_28 = add_3 = clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_723: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_106: "f32[8, 64, 1024]" = torch.ops.aten.bmm.default(permute_723, as_strided_scatter_35);  permute_723 = None
    permute_724: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_107: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_35, permute_724);  as_strided_scatter_35 = permute_724 = None
    view_791: "f32[1, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_106, [1, 8, 64, 1024]);  bmm_106 = None
    view_792: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_107, [1, 8, 1024, 64]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_725: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_791, [0, 1, 3, 2]);  view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_726: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_789, [0, 2, 1, 3]);  view_789 = None
    clone_147: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_726, memory_format = torch.contiguous_format);  permute_726 = None
    view_793: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_147, [1, 1024, 512]);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_794: "f32[1024, 512]" = torch.ops.aten.view.default(view_793, [1024, 512]);  view_793 = None
    permute_727: "f32[512, 1024]" = torch.ops.aten.permute.default(view_794, [1, 0])
    mm_285: "f32[512, 512]" = torch.ops.aten.mm.default(permute_727, view_7);  permute_727 = view_7 = None
    permute_728: "f32[512, 512]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    permute_729: "f32[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_286: "f32[1024, 512]" = torch.ops.aten.mm.default(view_794, permute_729);  view_794 = permute_729 = None
    view_795: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_286, [1, 1024, 512]);  mm_286 = None
    permute_730: "f32[512, 512]" = torch.ops.aten.permute.default(permute_728, [1, 0]);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_731: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_725, [0, 2, 1, 3]);  permute_725 = None
    view_796: "f32[1, 1024, 512]" = torch.ops.aten.view.default(permute_731, [1, 1024, 512]);  permute_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_797: "f32[1024, 512]" = torch.ops.aten.view.default(view_796, [1024, 512]);  view_796 = None
    permute_732: "f32[512, 1024]" = torch.ops.aten.permute.default(view_797, [1, 0])
    mm_287: "f32[512, 512]" = torch.ops.aten.mm.default(permute_732, view_4);  permute_732 = view_4 = None
    permute_733: "f32[512, 512]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    permute_734: "f32[512, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_288: "f32[1024, 512]" = torch.ops.aten.mm.default(view_797, permute_734);  view_797 = permute_734 = None
    view_798: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_288, [1, 1024, 512]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_217: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(view_795, view_798);  view_795 = view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_735: "f32[512, 512]" = torch.ops.aten.permute.default(permute_733, [1, 0]);  permute_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_736: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_792, [0, 2, 1, 3]);  view_792 = None
    clone_148: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_736, memory_format = torch.contiguous_format);  permute_736 = None
    view_799: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_148, [1, 1024, 512]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_800: "f32[1024, 512]" = torch.ops.aten.view.default(view_799, [1024, 512]);  view_799 = None
    permute_737: "f32[512, 1024]" = torch.ops.aten.permute.default(view_800, [1, 0])
    mm_289: "f32[512, 512]" = torch.ops.aten.mm.default(permute_737, view_1);  permute_737 = view_1 = None
    permute_738: "f32[512, 512]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    permute_739: "f32[512, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_290: "f32[1024, 512]" = torch.ops.aten.mm.default(view_800, permute_739);  view_800 = permute_739 = None
    view_801: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_290, [1, 1024, 512]);  mm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_218: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_217, view_801);  add_217 = view_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_740: "f32[512, 512]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_485: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_218, primals_1);  primals_1 = None
    mul_486: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_218, mul_1);  add_218 = mul_1 = None
    sum_103: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1], True);  mul_486 = None
    view_802: "f32[512]" = torch.ops.aten.view.default(sum_103, [512]);  sum_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_487: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_485, getitem)
    mul_488: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_485, rsqrt);  mul_485 = rsqrt = None
    sum_104: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True);  mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_219: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_215, mul_488);  add_215 = mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_125: "f32[1, 1024, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    pow_95: "f32[1, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_125, 3);  alias_125 = None
    mul_489: "f32[1, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_104, -0.5);  sum_104 = None
    mul_490: "f32[1, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_489, pow_95);  mul_489 = pow_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_103: "f32[1, 1024, 512]" = torch.ops.aten.expand.default(mul_490, [1, 1024, 512]);  mul_490 = None
    div_55: "f32[1, 1024, 512]" = torch.ops.aten.div.Scalar(expand_103, 512);  expand_103 = None
    pow_96: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem, 1.0);  getitem = None
    mul_491: "f32[1, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_96, 2.0);  pow_96 = None
    mul_492: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(div_55, mul_491);  div_55 = mul_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_220: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_219, mul_492);  add_219 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    convert_element_type_71: "f32[1, 1024, 512]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_493: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
    mul_494: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_220, mul_493);  add_220 = mul_493 = None
    clone_149: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_494, memory_format = torch.contiguous_format);  mul_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    eq_3: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_22: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[1, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_22, scalar_tensor_19, clone_149);  unsqueeze_22 = scalar_tensor_19 = clone_149 = None
    full_29: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_3: "f32[32128, 512]" = torch.ops.aten._unsafe_index_put.default(full_29, [view], where_21, True);  full_29 = view = where_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    add_221: "f32[32128, 512]" = torch.ops.aten.add.Tensor(_unsafe_index_put_1, _unsafe_index_put_3);  _unsafe_index_put_1 = _unsafe_index_put_3 = None
    return pytree.tree_unflatten([div_22, view_411, permute_70, permute_72, permute_80, permute_82, permute_91, permute_93, permute_100, permute_102, permute_111, permute_113, permute_120, permute_122, permute_131, permute_133, permute_140, permute_142, permute_151, permute_153, permute_160, permute_162, permute_171, permute_173, permute_180, permute_182, getitem_50, view_802, view_784, view_779, view_761, view_756, view_738, view_733, view_715, view_710, view_692, view_687, view_669, view_664, view_663, view_645, view_627, view_622, view_604, view_586, view_581, view_563, view_545, view_540, view_522, view_504, view_499, view_481, view_463, view_458, view_440, view_422, view_417, add_221, permute_740, permute_735, permute_730, _unsafe_index_put_2, permute_718, permute_714, permute_710, permute_706, permute_701, permute_696, permute_685, permute_681, permute_677, permute_673, permute_668, permute_663, permute_652, permute_648, permute_644, permute_640, permute_635, permute_630, permute_619, permute_615, permute_611, permute_607, permute_602, permute_597, permute_586, permute_582, permute_578, permute_574, permute_569, permute_564, permute_553, permute_549, permute_545, permute_541, permute_536, permute_531, _unsafe_index_put, permute_519, permute_515, permute_510, permute_505, permute_494, permute_490, permute_486, permute_482, permute_477, permute_472, permute_461, permute_457, permute_452, permute_447, permute_436, permute_432, permute_428, permute_424, permute_419, permute_414, permute_403, permute_399, permute_394, permute_389, permute_378, permute_374, permute_370, permute_366, permute_361, permute_356, permute_345, permute_341, permute_336, permute_331, permute_320, permute_316, permute_312, permute_308, permute_303, permute_298, permute_287, permute_283, permute_278, permute_273, permute_262, permute_258, permute_254, permute_250, permute_245, permute_240, permute_229, permute_225, permute_220, permute_215, permute_204, permute_200, permute_196, permute_192, None, None, None], self._out_spec)
    