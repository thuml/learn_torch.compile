from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[512]"; primals_2: "f32[512]"; primals_3: "f32[512]"; primals_4: "f32[512]"; primals_5: "f32[512]"; primals_6: "f32[512]"; primals_7: "f32[512]"; primals_8: "f32[512]"; primals_9: "f32[512]"; primals_10: "f32[512]"; primals_11: "f32[512]"; primals_12: "f32[512]"; primals_13: "f32[512]"; primals_14: "f32[512]"; primals_15: "f32[512]"; primals_16: "f32[512]"; primals_17: "f32[512]"; primals_18: "f32[512]"; primals_19: "f32[512]"; primals_20: "f32[512]"; primals_21: "f32[512]"; primals_22: "f32[512]"; primals_23: "f32[512]"; primals_24: "f32[512]"; primals_25: "f32[512]"; primals_26: "f32[512]"; primals_27: "f32[512]"; primals_28: "f32[512]"; primals_29: "f32[512]"; primals_30: "f32[512]"; primals_31: "f32[512]"; primals_32: "f32[512]"; primals_33: "f32[32128, 512]"; primals_34: "f32[512, 512]"; primals_35: "f32[512, 512]"; primals_36: "f32[512, 512]"; primals_37: "f32[32, 8]"; primals_38: "f32[512, 512]"; primals_39: "f32[2048, 512]"; primals_40: "f32[512, 2048]"; primals_41: "f32[512, 512]"; primals_42: "f32[512, 512]"; primals_43: "f32[512, 512]"; primals_44: "f32[512, 512]"; primals_45: "f32[2048, 512]"; primals_46: "f32[512, 2048]"; primals_47: "f32[512, 512]"; primals_48: "f32[512, 512]"; primals_49: "f32[512, 512]"; primals_50: "f32[512, 512]"; primals_51: "f32[2048, 512]"; primals_52: "f32[512, 2048]"; primals_53: "f32[512, 512]"; primals_54: "f32[512, 512]"; primals_55: "f32[512, 512]"; primals_56: "f32[512, 512]"; primals_57: "f32[2048, 512]"; primals_58: "f32[512, 2048]"; primals_59: "f32[512, 512]"; primals_60: "f32[512, 512]"; primals_61: "f32[512, 512]"; primals_62: "f32[512, 512]"; primals_63: "f32[2048, 512]"; primals_64: "f32[512, 2048]"; primals_65: "f32[512, 512]"; primals_66: "f32[512, 512]"; primals_67: "f32[512, 512]"; primals_68: "f32[512, 512]"; primals_69: "f32[2048, 512]"; primals_70: "f32[512, 2048]"; primals_71: "f32[512, 512]"; primals_72: "f32[512, 512]"; primals_73: "f32[512, 512]"; primals_74: "f32[32, 8]"; primals_75: "f32[512, 512]"; primals_76: "f32[512, 512]"; primals_77: "f32[512, 512]"; primals_78: "f32[512, 512]"; primals_79: "f32[512, 512]"; primals_80: "f32[2048, 512]"; primals_81: "f32[512, 2048]"; primals_82: "f32[512, 512]"; primals_83: "f32[512, 512]"; primals_84: "f32[512, 512]"; primals_85: "f32[512, 512]"; primals_86: "f32[512, 512]"; primals_87: "f32[512, 512]"; primals_88: "f32[512, 512]"; primals_89: "f32[512, 512]"; primals_90: "f32[2048, 512]"; primals_91: "f32[512, 2048]"; primals_92: "f32[512, 512]"; primals_93: "f32[512, 512]"; primals_94: "f32[512, 512]"; primals_95: "f32[512, 512]"; primals_96: "f32[512, 512]"; primals_97: "f32[512, 512]"; primals_98: "f32[512, 512]"; primals_99: "f32[512, 512]"; primals_100: "f32[2048, 512]"; primals_101: "f32[512, 2048]"; primals_102: "f32[512, 512]"; primals_103: "f32[512, 512]"; primals_104: "f32[512, 512]"; primals_105: "f32[512, 512]"; primals_106: "f32[512, 512]"; primals_107: "f32[512, 512]"; primals_108: "f32[512, 512]"; primals_109: "f32[512, 512]"; primals_110: "f32[2048, 512]"; primals_111: "f32[512, 2048]"; primals_112: "f32[512, 512]"; primals_113: "f32[512, 512]"; primals_114: "f32[512, 512]"; primals_115: "f32[512, 512]"; primals_116: "f32[512, 512]"; primals_117: "f32[512, 512]"; primals_118: "f32[512, 512]"; primals_119: "f32[512, 512]"; primals_120: "f32[2048, 512]"; primals_121: "f32[512, 2048]"; primals_122: "f32[512, 512]"; primals_123: "f32[512, 512]"; primals_124: "f32[512, 512]"; primals_125: "f32[512, 512]"; primals_126: "f32[512, 512]"; primals_127: "f32[512, 512]"; primals_128: "f32[512, 512]"; primals_129: "f32[512, 512]"; primals_130: "f32[2048, 512]"; primals_131: "f32[512, 2048]"; primals_132: "f32[32128, 512]"; primals_133: "i64[4, 1024]"; primals_134: "i64[4, 1024]"; tangents_1: "f32[4, 1024, 32128]"; tangents_2: "f32[4, 8, 1024, 64]"; tangents_3: "f32[4, 8, 1024, 64]"; tangents_4: "f32[4, 8, 1024, 64]"; tangents_5: "f32[4, 8, 1024, 64]"; tangents_6: "f32[4, 8, 1024, 64]"; tangents_7: "f32[4, 8, 1024, 64]"; tangents_8: "f32[4, 8, 1024, 64]"; tangents_9: "f32[4, 8, 1024, 64]"; tangents_10: "f32[4, 8, 1024, 64]"; tangents_11: "f32[4, 8, 1024, 64]"; tangents_12: "f32[4, 8, 1024, 64]"; tangents_13: "f32[4, 8, 1024, 64]"; tangents_14: "f32[4, 8, 1024, 64]"; tangents_15: "f32[4, 8, 1024, 64]"; tangents_16: "f32[4, 8, 1024, 64]"; tangents_17: "f32[4, 8, 1024, 64]"; tangents_18: "f32[4, 8, 1024, 64]"; tangents_19: "f32[4, 8, 1024, 64]"; tangents_20: "f32[4, 8, 1024, 64]"; tangents_21: "f32[4, 8, 1024, 64]"; tangents_22: "f32[4, 8, 1024, 64]"; tangents_23: "f32[4, 8, 1024, 64]"; tangents_24: "f32[4, 8, 1024, 64]"; tangents_25: "f32[4, 8, 1024, 64]"; tangents_26: "f32[4, 1024, 512]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1011, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[4, 1024]" = torch.ops.aten.view.default(primals_133, [-1, 1024]);  primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding: "f32[4, 1024, 512]" = torch.ops.aten.embedding.default(primals_33, view)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1033, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    full: "f32[4, 1024]" = torch.ops.aten.full.default([4, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_1: "f32[4, 1024]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[4, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
    unsqueeze_1: "f32[4, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_2: "f32[4, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[4, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(1.0, slice_2);  slice_2 = None
    mul: "f32[4, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    clone: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(embedding);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_1: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(clone, 2)
    mean: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
    rsqrt: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    alias: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt)
    mul_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(clone, rsqrt)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_1, mul_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute: "f32[512, 512]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    view_1: "f32[4096, 512]" = torch.ops.aten.view.default(mul_2, [4096, 512])
    mm: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1, permute)
    view_2: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm, [4, 1024, 512]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_3: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_2, [4, -1, 8, 64]);  view_2 = None
    permute_1: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    view_4: "f32[4096, 512]" = torch.ops.aten.view.default(mul_2, [4096, 512])
    mm_1: "f32[4096, 512]" = torch.ops.aten.mm.default(view_4, permute_2)
    view_5: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_1, [4, 1024, 512]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_6: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_5, [4, -1, 8, 64]);  view_5 = None
    permute_3: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    view_7: "f32[4096, 512]" = torch.ops.aten.view.default(mul_2, [4096, 512]);  mul_2 = None
    mm_2: "f32[4096, 512]" = torch.ops.aten.mm.default(view_7, permute_4)
    view_8: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_2, [4, 1024, 512]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_9: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_8, [4, -1, 8, 64]);  view_8 = None
    permute_5: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_6: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_1, [4, 8, 1024, 64]);  permute_1 = None
    clone_1: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_10: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_1, [32, 1024, 64]);  clone_1 = None
    expand_1: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_6, [4, 8, 64, 1024]);  permute_6 = None
    clone_2: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_11: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_2, [32, 64, 1024]);  clone_2 = None
    bmm: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_10, view_11)
    view_12: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm, [4, 8, 1024, 1024]);  bmm = None
    
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
    add_4: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_4, mul);  unsqueeze_4 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_5: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_12, add_4);  view_12 = None
    view_13: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_5, [32, 1024, 1024]);  add_5 = None
    view_14: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_13, [4, 8, 1024, 1024]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_14, [-1], True)
    sub_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_14, amax);  view_14 = amax = None
    exp: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_1: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_3: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_3, [4, 8, 1024, 1024]);  clone_3 = None
    view_15: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_2, [32, 1024, 1024]);  expand_2 = None
    expand_3: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [4, 8, 1024, 64]);  permute_5 = None
    clone_4: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_16: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_4, [32, 1024, 64]);  clone_4 = None
    bmm_1: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_15, view_16)
    view_17: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_1, [4, 8, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_8: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
    clone_5: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_18: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_5, [4, -1, 512]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_9: "f32[512, 512]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    view_19: "f32[4096, 512]" = torch.ops.aten.view.default(view_18, [4096, 512]);  view_18 = None
    mm_3: "f32[4096, 512]" = torch.ops.aten.mm.default(view_19, permute_9)
    view_20: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_3, [4, 1024, 512]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_6: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    add_6: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(clone, clone_6);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_2: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
    mean_1: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_7: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
    rsqrt_1: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    alias_2: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_1)
    mul_5: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_6: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_2, mul_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_10: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    view_21: "f32[4096, 512]" = torch.ops.aten.view.default(mul_6, [4096, 512]);  mul_6 = None
    mm_4: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_21, permute_10)
    view_22: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_4, [4, 1024, 2048]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_22);  view_22 = None
    alias_3: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_7: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu);  relu = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_11: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    view_23: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_7, [4096, 2048]);  clone_7 = None
    mm_5: "f32[4096, 512]" = torch.ops.aten.mm.default(view_23, permute_11)
    view_24: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_5, [4, 1024, 512]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_8: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    add_8: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_6, clone_8);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_3: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_8, 2)
    mean_2: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_9: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
    rsqrt_2: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    alias_4: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_2)
    mul_7: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_8, rsqrt_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_8: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_3, mul_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    view_25: "f32[4096, 512]" = torch.ops.aten.view.default(mul_8, [4096, 512])
    mm_6: "f32[4096, 512]" = torch.ops.aten.mm.default(view_25, permute_12)
    view_26: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_6, [4, 1024, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_27: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_26, [4, -1, 8, 64]);  view_26 = None
    permute_13: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_14: "f32[512, 512]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    view_28: "f32[4096, 512]" = torch.ops.aten.view.default(mul_8, [4096, 512])
    mm_7: "f32[4096, 512]" = torch.ops.aten.mm.default(view_28, permute_14)
    view_29: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_7, [4, 1024, 512]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_30: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_29, [4, -1, 8, 64]);  view_29 = None
    permute_15: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_16: "f32[512, 512]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    view_31: "f32[4096, 512]" = torch.ops.aten.view.default(mul_8, [4096, 512]);  mul_8 = None
    mm_8: "f32[4096, 512]" = torch.ops.aten.mm.default(view_31, permute_16)
    view_32: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_8, [4, 1024, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_33: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_32, [4, -1, 8, 64]);  view_32 = None
    permute_17: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_18: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_4: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_13, [4, 8, 1024, 64]);  permute_13 = None
    clone_9: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_34: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_9, [32, 1024, 64]);  clone_9 = None
    expand_5: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [4, 8, 64, 1024]);  permute_18 = None
    clone_10: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_35: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_10, [32, 64, 1024]);  clone_10 = None
    bmm_2: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_34, view_35)
    view_36: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [4, 8, 1024, 1024]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_10: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_36, add_4);  view_36 = None
    view_37: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_10, [32, 1024, 1024]);  add_10 = None
    view_38: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_37, [4, 8, 1024, 1024]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_1: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_38, [-1], True)
    sub_3: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_38, amax_1);  view_38 = amax_1 = None
    exp_1: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_5: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_11: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_6: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_11, [4, 8, 1024, 1024]);  clone_11 = None
    view_39: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_6, [32, 1024, 1024]);  expand_6 = None
    expand_7: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [4, 8, 1024, 64]);  permute_17 = None
    clone_12: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_40: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_12, [32, 1024, 64]);  clone_12 = None
    bmm_3: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_39, view_40)
    view_41: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_3, [4, 8, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_19: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    clone_13: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_42: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_13, [4, -1, 512]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_20: "f32[512, 512]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    view_43: "f32[4096, 512]" = torch.ops.aten.view.default(view_42, [4096, 512]);  view_42 = None
    mm_9: "f32[4096, 512]" = torch.ops.aten.mm.default(view_43, permute_20)
    view_44: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_9, [4, 1024, 512]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_14: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    add_11: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_8, clone_14);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_4: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
    mean_3: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_12: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
    rsqrt_3: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    alias_6: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_3)
    mul_9: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_10: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_4, mul_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_21: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    view_45: "f32[4096, 512]" = torch.ops.aten.view.default(mul_10, [4096, 512]);  mul_10 = None
    mm_10: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_45, permute_21)
    view_46: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_10, [4, 1024, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_1: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_46);  view_46 = None
    alias_7: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_15: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_1);  relu_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_22: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    view_47: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_15, [4096, 2048]);  clone_15 = None
    mm_11: "f32[4096, 512]" = torch.ops.aten.mm.default(view_47, permute_22)
    view_48: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_11, [4, 1024, 512]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_16: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    add_13: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_11, clone_16);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_5: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
    mean_4: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_14: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
    rsqrt_4: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    alias_8: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_4)
    mul_11: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_12: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_5, mul_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_23: "f32[512, 512]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    view_49: "f32[4096, 512]" = torch.ops.aten.view.default(mul_12, [4096, 512])
    mm_12: "f32[4096, 512]" = torch.ops.aten.mm.default(view_49, permute_23)
    view_50: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_12, [4, 1024, 512]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_51: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_50, [4, -1, 8, 64]);  view_50 = None
    permute_24: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    view_52: "f32[4096, 512]" = torch.ops.aten.view.default(mul_12, [4096, 512])
    mm_13: "f32[4096, 512]" = torch.ops.aten.mm.default(view_52, permute_25)
    view_53: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_13, [4, 1024, 512]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_54: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_53, [4, -1, 8, 64]);  view_53 = None
    permute_26: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_27: "f32[512, 512]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    view_55: "f32[4096, 512]" = torch.ops.aten.view.default(mul_12, [4096, 512]);  mul_12 = None
    mm_14: "f32[4096, 512]" = torch.ops.aten.mm.default(view_55, permute_27)
    view_56: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_14, [4, 1024, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_57: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_56, [4, -1, 8, 64]);  view_56 = None
    permute_28: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_29: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_8: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_24, [4, 8, 1024, 64]);  permute_24 = None
    clone_17: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_58: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_17, [32, 1024, 64]);  clone_17 = None
    expand_9: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_29, [4, 8, 64, 1024]);  permute_29 = None
    clone_18: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_59: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_18, [32, 64, 1024]);  clone_18 = None
    bmm_4: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_58, view_59)
    view_60: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_4, [4, 8, 1024, 1024]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_15: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_60, add_4);  view_60 = None
    view_61: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_15, [32, 1024, 1024]);  add_15 = None
    view_62: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_61, [4, 8, 1024, 1024]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_2: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_62, [-1], True)
    sub_4: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_62, amax_2);  view_62 = amax_2 = None
    exp_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_3: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_4: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_9: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_19: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_10: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_19, [4, 8, 1024, 1024]);  clone_19 = None
    view_63: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_10, [32, 1024, 1024]);  expand_10 = None
    expand_11: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_28, [4, 8, 1024, 64]);  permute_28 = None
    clone_20: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_64: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_20, [32, 1024, 64]);  clone_20 = None
    bmm_5: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_63, view_64)
    view_65: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_5, [4, 8, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_30: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_21: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_66: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_21, [4, -1, 512]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_31: "f32[512, 512]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    view_67: "f32[4096, 512]" = torch.ops.aten.view.default(view_66, [4096, 512]);  view_66 = None
    mm_15: "f32[4096, 512]" = torch.ops.aten.mm.default(view_67, permute_31)
    view_68: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_15, [4, 1024, 512]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_22: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    add_16: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_13, clone_22);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_6: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
    mean_5: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_17: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
    rsqrt_5: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    alias_10: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_5)
    mul_13: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_14: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_6, mul_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_32: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    view_69: "f32[4096, 512]" = torch.ops.aten.view.default(mul_14, [4096, 512]);  mul_14 = None
    mm_16: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_69, permute_32)
    view_70: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_16, [4, 1024, 2048]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_2: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_70);  view_70 = None
    alias_11: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_23: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_2);  relu_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_33: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    view_71: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_23, [4096, 2048]);  clone_23 = None
    mm_17: "f32[4096, 512]" = torch.ops.aten.mm.default(view_71, permute_33)
    view_72: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_17, [4, 1024, 512]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_24: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    add_18: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_16, clone_24);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_7: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_18, 2)
    mean_6: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_19: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
    rsqrt_6: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    alias_12: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_6)
    mul_15: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_18, rsqrt_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_16: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_7, mul_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    view_73: "f32[4096, 512]" = torch.ops.aten.view.default(mul_16, [4096, 512])
    mm_18: "f32[4096, 512]" = torch.ops.aten.mm.default(view_73, permute_34)
    view_74: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_18, [4, 1024, 512]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_75: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_74, [4, -1, 8, 64]);  view_74 = None
    permute_35: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_36: "f32[512, 512]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    view_76: "f32[4096, 512]" = torch.ops.aten.view.default(mul_16, [4096, 512])
    mm_19: "f32[4096, 512]" = torch.ops.aten.mm.default(view_76, permute_36)
    view_77: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_19, [4, 1024, 512]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_78: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_77, [4, -1, 8, 64]);  view_77 = None
    permute_37: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_38: "f32[512, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    view_79: "f32[4096, 512]" = torch.ops.aten.view.default(mul_16, [4096, 512]);  mul_16 = None
    mm_20: "f32[4096, 512]" = torch.ops.aten.mm.default(view_79, permute_38)
    view_80: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_20, [4, 1024, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_81: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_80, [4, -1, 8, 64]);  view_80 = None
    permute_39: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_40: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_12: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [4, 8, 1024, 64]);  permute_35 = None
    clone_25: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_82: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_25, [32, 1024, 64]);  clone_25 = None
    expand_13: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_40, [4, 8, 64, 1024]);  permute_40 = None
    clone_26: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_83: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_26, [32, 64, 1024]);  clone_26 = None
    bmm_6: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_6, [4, 8, 1024, 1024]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_20: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_84, add_4);  view_84 = None
    view_85: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_20, [32, 1024, 1024]);  add_20 = None
    view_86: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_85, [4, 8, 1024, 1024]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_3: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_86, [-1], True)
    sub_5: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_86, amax_3);  view_86 = amax_3 = None
    exp_3: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_4: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_5: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_13: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_27: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_14: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_27, [4, 8, 1024, 1024]);  clone_27 = None
    view_87: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_14, [32, 1024, 1024]);  expand_14 = None
    expand_15: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_39, [4, 8, 1024, 64]);  permute_39 = None
    clone_28: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_88: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_28, [32, 1024, 64]);  clone_28 = None
    bmm_7: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_87, view_88)
    view_89: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_7, [4, 8, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_41: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_29: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_90: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_29, [4, -1, 512]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_42: "f32[512, 512]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    view_91: "f32[4096, 512]" = torch.ops.aten.view.default(view_90, [4096, 512]);  view_90 = None
    mm_21: "f32[4096, 512]" = torch.ops.aten.mm.default(view_91, permute_42)
    view_92: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_21, [4, 1024, 512]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_30: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_92);  view_92 = None
    add_21: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_18, clone_30);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_8: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
    mean_7: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_22: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
    rsqrt_7: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    alias_14: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_7)
    mul_17: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_18: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_8, mul_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_43: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    view_93: "f32[4096, 512]" = torch.ops.aten.view.default(mul_18, [4096, 512]);  mul_18 = None
    mm_22: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_93, permute_43)
    view_94: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_22, [4, 1024, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_3: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_94);  view_94 = None
    alias_15: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_31: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_3);  relu_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_44: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    view_95: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_31, [4096, 2048]);  clone_31 = None
    mm_23: "f32[4096, 512]" = torch.ops.aten.mm.default(view_95, permute_44)
    view_96: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_23, [4, 1024, 512]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_32: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    add_23: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_21, clone_32);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_9: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 2)
    mean_8: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_24: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
    rsqrt_8: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    alias_16: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_8)
    mul_19: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_23, rsqrt_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_20: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_9, mul_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_45: "f32[512, 512]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    view_97: "f32[4096, 512]" = torch.ops.aten.view.default(mul_20, [4096, 512])
    mm_24: "f32[4096, 512]" = torch.ops.aten.mm.default(view_97, permute_45)
    view_98: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_24, [4, 1024, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_99: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_98, [4, -1, 8, 64]);  view_98 = None
    permute_46: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_47: "f32[512, 512]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    view_100: "f32[4096, 512]" = torch.ops.aten.view.default(mul_20, [4096, 512])
    mm_25: "f32[4096, 512]" = torch.ops.aten.mm.default(view_100, permute_47)
    view_101: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_25, [4, 1024, 512]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_102: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_101, [4, -1, 8, 64]);  view_101 = None
    permute_48: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_49: "f32[512, 512]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    view_103: "f32[4096, 512]" = torch.ops.aten.view.default(mul_20, [4096, 512]);  mul_20 = None
    mm_26: "f32[4096, 512]" = torch.ops.aten.mm.default(view_103, permute_49)
    view_104: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_26, [4, 1024, 512]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_105: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_104, [4, -1, 8, 64]);  view_104 = None
    permute_50: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_51: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_16: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_46, [4, 8, 1024, 64]);  permute_46 = None
    clone_33: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_106: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_33, [32, 1024, 64]);  clone_33 = None
    expand_17: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_51, [4, 8, 64, 1024]);  permute_51 = None
    clone_34: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_107: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_34, [32, 64, 1024]);  clone_34 = None
    bmm_8: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_106, view_107)
    view_108: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_8, [4, 8, 1024, 1024]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_25: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_108, add_4);  view_108 = None
    view_109: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_25, [32, 1024, 1024]);  add_25 = None
    view_110: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_109, [4, 8, 1024, 1024]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_4: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_110, [-1], True)
    sub_6: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_110, amax_4);  view_110 = amax_4 = None
    exp_4: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_5: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_6: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_17: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_35: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_18: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_35, [4, 8, 1024, 1024]);  clone_35 = None
    view_111: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_18, [32, 1024, 1024]);  expand_18 = None
    expand_19: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [4, 8, 1024, 64]);  permute_50 = None
    clone_36: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_112: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_36, [32, 1024, 64]);  clone_36 = None
    bmm_9: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_111, view_112)
    view_113: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_9, [4, 8, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_52: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    clone_37: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_114: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_37, [4, -1, 512]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    view_115: "f32[4096, 512]" = torch.ops.aten.view.default(view_114, [4096, 512]);  view_114 = None
    mm_27: "f32[4096, 512]" = torch.ops.aten.mm.default(view_115, permute_53)
    view_116: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_27, [4, 1024, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_38: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    add_26: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_23, clone_38);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_10: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 2)
    mean_9: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_27: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
    rsqrt_9: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    alias_18: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_9)
    mul_21: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_22: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_10, mul_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_54: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    view_117: "f32[4096, 512]" = torch.ops.aten.view.default(mul_22, [4096, 512]);  mul_22 = None
    mm_28: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_117, permute_54)
    view_118: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_28, [4, 1024, 2048]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_4: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_118);  view_118 = None
    alias_19: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_39: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_4);  relu_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_55: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    view_119: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_39, [4096, 2048]);  clone_39 = None
    mm_29: "f32[4096, 512]" = torch.ops.aten.mm.default(view_119, permute_55)
    view_120: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_29, [4, 1024, 512]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_40: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    add_28: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_26, clone_40);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_11: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_28, 2)
    mean_10: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_29: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
    rsqrt_10: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    alias_20: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_10)
    mul_23: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_28, rsqrt_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_24: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_11, mul_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_56: "f32[512, 512]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    view_121: "f32[4096, 512]" = torch.ops.aten.view.default(mul_24, [4096, 512])
    mm_30: "f32[4096, 512]" = torch.ops.aten.mm.default(view_121, permute_56)
    view_122: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_30, [4, 1024, 512]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_123: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_122, [4, -1, 8, 64]);  view_122 = None
    permute_57: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_58: "f32[512, 512]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    view_124: "f32[4096, 512]" = torch.ops.aten.view.default(mul_24, [4096, 512])
    mm_31: "f32[4096, 512]" = torch.ops.aten.mm.default(view_124, permute_58)
    view_125: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_31, [4, 1024, 512]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_126: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_125, [4, -1, 8, 64]);  view_125 = None
    permute_59: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_60: "f32[512, 512]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    view_127: "f32[4096, 512]" = torch.ops.aten.view.default(mul_24, [4096, 512]);  mul_24 = None
    mm_32: "f32[4096, 512]" = torch.ops.aten.mm.default(view_127, permute_60)
    view_128: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_32, [4, 1024, 512]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_129: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_128, [4, -1, 8, 64]);  view_128 = None
    permute_61: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_62: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_20: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [4, 8, 1024, 64]);  permute_57 = None
    clone_41: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_130: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_41, [32, 1024, 64]);  clone_41 = None
    expand_21: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_62, [4, 8, 64, 1024]);  permute_62 = None
    clone_42: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_131: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_42, [32, 64, 1024]);  clone_42 = None
    bmm_10: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_130, view_131)
    view_132: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_10, [4, 8, 1024, 1024]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_30: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_132, add_4);  view_132 = add_4 = None
    view_133: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_30, [32, 1024, 1024]);  add_30 = None
    view_134: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_133, [4, 8, 1024, 1024]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_5: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_134, [-1], True)
    sub_7: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_134, amax_5);  view_134 = amax_5 = None
    exp_5: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_6: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_7: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_21: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_43: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_22: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_43, [4, 8, 1024, 1024]);  clone_43 = None
    view_135: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_22, [32, 1024, 1024]);  expand_22 = None
    expand_23: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_61, [4, 8, 1024, 64]);  permute_61 = None
    clone_44: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_136: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_44, [32, 1024, 64]);  clone_44 = None
    bmm_11: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_135, view_136)
    view_137: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_11, [4, 8, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_63: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    clone_45: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_138: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_45, [4, -1, 512]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_64: "f32[512, 512]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    view_139: "f32[4096, 512]" = torch.ops.aten.view.default(view_138, [4096, 512]);  view_138 = None
    mm_33: "f32[4096, 512]" = torch.ops.aten.mm.default(view_139, permute_64)
    view_140: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_33, [4, 1024, 512]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_46: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    add_31: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_28, clone_46);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_12: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
    mean_11: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_32: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
    rsqrt_11: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    alias_22: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_11)
    mul_25: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_26: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_12, mul_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_65: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    view_141: "f32[4096, 512]" = torch.ops.aten.view.default(mul_26, [4096, 512]);  mul_26 = None
    mm_34: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_141, permute_65)
    view_142: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_34, [4, 1024, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_5: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_142);  view_142 = None
    alias_23: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_47: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_5);  relu_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_66: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_143: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_47, [4096, 2048]);  clone_47 = None
    mm_35: "f32[4096, 512]" = torch.ops.aten.mm.default(view_143, permute_66)
    view_144: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_35, [4, 1024, 512]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_48: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    add_33: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_31, clone_48);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_13: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_33, 2)
    mean_12: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_34: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
    rsqrt_12: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    alias_24: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_12)
    mul_27: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_33, rsqrt_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_28: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_13, mul_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    clone_49: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1011, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_145: "i64[4, 1024]" = torch.ops.aten.view.default(primals_134, [-1, 1024]);  primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding_2: "f32[4, 1024, 512]" = torch.ops.aten.embedding.default(primals_33, view_145);  primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1033, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    full_2: "f32[4, 1024]" = torch.ops.aten.full.default([4, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1036, code: encoder_attention_mask = torch.ones(
    full_3: "i64[4, 1024]" = torch.ops.aten.full.default([4, 1024], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:860, code: seq_ids = torch.arange(seq_length, device=device)
    iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:861, code: causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    unsqueeze_5: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_2, 0)
    unsqueeze_6: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
    slice_5: "i64[1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_6, 2, 0, 9223372036854775807);  unsqueeze_6 = None
    repeat: "i64[4, 1024, 1024]" = torch.ops.aten.repeat.default(slice_5, [4, 1024, 1]);  slice_5 = None
    unsqueeze_7: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
    slice_6: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_7, 1, 0, 9223372036854775807);  unsqueeze_7 = None
    unsqueeze_8: "i64[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_6, 2);  slice_6 = None
    le: "b8[4, 1024, 1024]" = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:864, code: causal_mask = causal_mask.to(attention_mask.dtype)
    convert_element_type_3: "f32[4, 1024, 1024]" = torch.ops.prims.convert_element_type.default(le, torch.float32);  le = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:876, code: extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    slice_7: "f32[4, 1024, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_3, 0, 0, 9223372036854775807);  convert_element_type_3 = None
    unsqueeze_9: "f32[4, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(slice_7, 1);  slice_7 = None
    slice_8: "f32[4, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_9, 2, 0, 9223372036854775807);  unsqueeze_9 = None
    slice_9: "f32[4, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 9223372036854775807);  slice_8 = None
    slice_10: "f32[4, 1024]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807);  full_2 = None
    unsqueeze_10: "f32[4, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_10, 1);  slice_10 = None
    unsqueeze_11: "f32[4, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
    slice_11: "f32[4, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_11, 3, 0, 9223372036854775807);  unsqueeze_11 = None
    mul_29: "f32[4, 1, 1024, 1024]" = torch.ops.aten.mul.Tensor(slice_9, slice_11);  slice_9 = slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub_8: "f32[4, 1, 1024, 1024]" = torch.ops.aten.sub.Tensor(1.0, mul_29);  mul_29 = None
    mul_30: "f32[4, 1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_8, -3.4028234663852886e+38);  sub_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:840, code: encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    slice_12: "i64[4, 1024]" = torch.ops.aten.slice.Tensor(full_3, 0, 0, 9223372036854775807);  full_3 = None
    unsqueeze_12: "i64[4, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_12, 1);  slice_12 = None
    unsqueeze_13: "i64[4, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
    slice_13: "i64[4, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 0, 9223372036854775807);  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:846, code: encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    convert_element_type_4: "f32[4, 1, 1, 1024]" = torch.ops.prims.convert_element_type.default(slice_13, torch.float32);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:847, code: encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
    sub_9: "f32[4, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(1.0, convert_element_type_4);  convert_element_type_4 = None
    mul_31: "f32[4, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_9, -3.4028234663852886e+38);  sub_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    clone_50: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(embedding_2);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_14: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(clone_50, 2)
    mean_13: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_35: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
    rsqrt_13: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    alias_25: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_13)
    mul_32: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(clone_50, rsqrt_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_33: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_14, mul_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    view_146: "f32[4096, 512]" = torch.ops.aten.view.default(mul_33, [4096, 512])
    mm_36: "f32[4096, 512]" = torch.ops.aten.mm.default(view_146, permute_67)
    view_147: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_36, [4, 1024, 512]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_148: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_147, [4, -1, 8, 64]);  view_147 = None
    permute_68: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_69: "f32[512, 512]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    view_149: "f32[4096, 512]" = torch.ops.aten.view.default(mul_33, [4096, 512])
    mm_37: "f32[4096, 512]" = torch.ops.aten.mm.default(view_149, permute_69)
    view_150: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_37, [4, 1024, 512]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_151: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_150, [4, -1, 8, 64]);  view_150 = None
    permute_70: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_71: "f32[512, 512]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    view_152: "f32[4096, 512]" = torch.ops.aten.view.default(mul_33, [4096, 512]);  mul_33 = None
    mm_38: "f32[4096, 512]" = torch.ops.aten.mm.default(view_152, permute_71)
    view_153: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_38, [4, 1024, 512]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_154: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_153, [4, -1, 8, 64]);  view_153 = None
    permute_72: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_73: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_24: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_68, [4, 8, 1024, 64]);  permute_68 = None
    clone_51: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_155: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_51, [32, 1024, 64]);  clone_51 = None
    expand_25: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_73, [4, 8, 64, 1024]);  permute_73 = None
    clone_52: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_156: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_52, [32, 64, 1024]);  clone_52 = None
    bmm_12: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_155, view_156)
    view_157: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_12, [4, 8, 1024, 1024]);  bmm_12 = None
    
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
    add_38: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_16, mul_30);  unsqueeze_16 = mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_39: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_157, add_38);  view_157 = None
    view_158: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_39, [32, 1024, 1024]);  add_39 = None
    view_159: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_158, [4, 8, 1024, 1024]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_6: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_159, [-1], True)
    sub_11: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_159, amax_6);  view_159 = amax_6 = None
    exp_6: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_7: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_10: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_26: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_53: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_26: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_53, [4, 8, 1024, 1024]);  clone_53 = None
    view_160: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_26, [32, 1024, 1024]);  expand_26 = None
    expand_27: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_72, [4, 8, 1024, 64])
    clone_54: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_161: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_54, [32, 1024, 64]);  clone_54 = None
    bmm_13: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_160, view_161)
    view_162: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_13, [4, 8, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_75: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_55: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    view_163: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_55, [4, -1, 512]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_76: "f32[512, 512]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    view_164: "f32[4096, 512]" = torch.ops.aten.view.default(view_163, [4096, 512]);  view_163 = None
    mm_39: "f32[4096, 512]" = torch.ops.aten.mm.default(view_164, permute_76)
    view_165: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_39, [4, 1024, 512]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_56: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_165);  view_165 = None
    add_40: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(clone_50, clone_56);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_15: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_40, 2)
    mean_14: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_41: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
    rsqrt_14: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    alias_27: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_14)
    mul_35: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_40, rsqrt_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_36: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_15, mul_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_77: "f32[512, 512]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    view_166: "f32[4096, 512]" = torch.ops.aten.view.default(mul_36, [4096, 512]);  mul_36 = None
    mm_40: "f32[4096, 512]" = torch.ops.aten.mm.default(view_166, permute_77)
    view_167: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_40, [4, 1024, 512]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_168: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_167, [4, -1, 8, 64]);  view_167 = None
    permute_78: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_79: "f32[512, 512]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    view_169: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_41: "f32[4096, 512]" = torch.ops.aten.mm.default(view_169, permute_79)
    view_170: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_41, [4, 1024, 512]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_171: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_170, [4, -1, 8, 64]);  view_170 = None
    permute_80: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    view_172: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_42: "f32[4096, 512]" = torch.ops.aten.mm.default(view_172, permute_81)
    view_173: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_42, [4, 1024, 512]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_174: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_173, [4, -1, 8, 64]);  view_173 = None
    permute_82: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_83: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_28: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_78, [4, 8, 1024, 64]);  permute_78 = None
    clone_57: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_175: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_57, [32, 1024, 64]);  clone_57 = None
    expand_29: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_83, [4, 8, 64, 1024]);  permute_83 = None
    clone_58: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_176: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_58, [32, 64, 1024]);  clone_58 = None
    bmm_14: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_175, view_176)
    view_177: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, [4, 8, 1024, 1024]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: position_bias = torch.zeros(
    full_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.full.default([1, 8, 1024, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    add_42: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(full_6, mul_31);  full_6 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_43: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_177, add_42);  view_177 = None
    view_178: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_43, [32, 1024, 1024]);  add_43 = None
    view_179: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_178, [4, 8, 1024, 1024]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_7: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_179, [-1], True)
    sub_12: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_179, amax_7);  view_179 = amax_7 = None
    exp_7: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_8: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_11: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_28: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_59: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_30: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_59, [4, 8, 1024, 1024]);  clone_59 = None
    view_180: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_30, [32, 1024, 1024]);  expand_30 = None
    expand_31: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_82, [4, 8, 1024, 64])
    clone_60: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_181: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_60, [32, 1024, 64]);  clone_60 = None
    bmm_15: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_180, view_181)
    view_182: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_15, [4, 8, 1024, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_84: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    clone_61: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_183: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_61, [4, -1, 512]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_85: "f32[512, 512]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    view_184: "f32[4096, 512]" = torch.ops.aten.view.default(view_183, [4096, 512]);  view_183 = None
    mm_43: "f32[4096, 512]" = torch.ops.aten.mm.default(view_184, permute_85)
    view_185: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_43, [4, 1024, 512]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_62: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_185);  view_185 = None
    add_44: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_40, clone_62);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_16: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 2)
    mean_15: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_45: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
    rsqrt_15: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    alias_29: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_15)
    mul_37: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_38: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_16, mul_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_86: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    view_186: "f32[4096, 512]" = torch.ops.aten.view.default(mul_38, [4096, 512]);  mul_38 = None
    mm_44: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_186, permute_86)
    view_187: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_44, [4, 1024, 2048]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_6: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_187);  view_187 = None
    alias_30: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_63: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_6);  relu_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_87: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    view_188: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_63, [4096, 2048]);  clone_63 = None
    mm_45: "f32[4096, 512]" = torch.ops.aten.mm.default(view_188, permute_87)
    view_189: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_45, [4, 1024, 512]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_64: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_189);  view_189 = None
    add_46: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_44, clone_64);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_17: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 2)
    mean_16: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_47: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
    rsqrt_16: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    alias_31: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_16)
    mul_39: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_40: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_17, mul_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    view_190: "f32[4096, 512]" = torch.ops.aten.view.default(mul_40, [4096, 512])
    mm_46: "f32[4096, 512]" = torch.ops.aten.mm.default(view_190, permute_88)
    view_191: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_46, [4, 1024, 512]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_192: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_191, [4, -1, 8, 64]);  view_191 = None
    permute_89: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_90: "f32[512, 512]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    view_193: "f32[4096, 512]" = torch.ops.aten.view.default(mul_40, [4096, 512])
    mm_47: "f32[4096, 512]" = torch.ops.aten.mm.default(view_193, permute_90)
    view_194: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_47, [4, 1024, 512]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_195: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_194, [4, -1, 8, 64]);  view_194 = None
    permute_91: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_92: "f32[512, 512]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    view_196: "f32[4096, 512]" = torch.ops.aten.view.default(mul_40, [4096, 512]);  mul_40 = None
    mm_48: "f32[4096, 512]" = torch.ops.aten.mm.default(view_196, permute_92)
    view_197: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_48, [4, 1024, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_198: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_197, [4, -1, 8, 64]);  view_197 = None
    permute_93: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_94: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_32: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_89, [4, 8, 1024, 64]);  permute_89 = None
    clone_65: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_199: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_65, [32, 1024, 64]);  clone_65 = None
    expand_33: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_94, [4, 8, 64, 1024]);  permute_94 = None
    clone_66: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_200: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_66, [32, 64, 1024]);  clone_66 = None
    bmm_16: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_199, view_200)
    view_201: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_16, [4, 8, 1024, 1024]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_48: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_201, add_38);  view_201 = None
    view_202: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_48, [32, 1024, 1024]);  add_48 = None
    view_203: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_202, [4, 8, 1024, 1024]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_8: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_203, [-1], True)
    sub_13: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_203, amax_8);  view_203 = amax_8 = None
    exp_8: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_9: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_12: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_32: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_67: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_34: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_67, [4, 8, 1024, 1024]);  clone_67 = None
    view_204: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_34, [32, 1024, 1024]);  expand_34 = None
    expand_35: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_93, [4, 8, 1024, 64])
    clone_68: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_205: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_68, [32, 1024, 64]);  clone_68 = None
    bmm_17: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_204, view_205)
    view_206: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_17, [4, 8, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_95: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    clone_69: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_207: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_69, [4, -1, 512]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_96: "f32[512, 512]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    view_208: "f32[4096, 512]" = torch.ops.aten.view.default(view_207, [4096, 512]);  view_207 = None
    mm_49: "f32[4096, 512]" = torch.ops.aten.mm.default(view_208, permute_96)
    view_209: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_49, [4, 1024, 512]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_70: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_209);  view_209 = None
    add_49: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_46, clone_70);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_18: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 2)
    mean_17: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_50: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
    rsqrt_17: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    alias_33: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_17)
    mul_41: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_42: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_18, mul_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_97: "f32[512, 512]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    view_210: "f32[4096, 512]" = torch.ops.aten.view.default(mul_42, [4096, 512]);  mul_42 = None
    mm_50: "f32[4096, 512]" = torch.ops.aten.mm.default(view_210, permute_97)
    view_211: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_50, [4, 1024, 512]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_212: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_211, [4, -1, 8, 64]);  view_211 = None
    permute_98: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_99: "f32[512, 512]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    view_213: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_51: "f32[4096, 512]" = torch.ops.aten.mm.default(view_213, permute_99)
    view_214: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_51, [4, 1024, 512]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_215: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_214, [4, -1, 8, 64]);  view_214 = None
    permute_100: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_101: "f32[512, 512]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    view_216: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_52: "f32[4096, 512]" = torch.ops.aten.mm.default(view_216, permute_101)
    view_217: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_52, [4, 1024, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_218: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_217, [4, -1, 8, 64]);  view_217 = None
    permute_102: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_103: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_36: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_98, [4, 8, 1024, 64]);  permute_98 = None
    clone_71: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_219: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_71, [32, 1024, 64]);  clone_71 = None
    expand_37: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_103, [4, 8, 64, 1024]);  permute_103 = None
    clone_72: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_220: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_72, [32, 64, 1024]);  clone_72 = None
    bmm_18: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_219, view_220)
    view_221: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, [4, 8, 1024, 1024]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_51: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_221, add_42);  view_221 = None
    view_222: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_51, [32, 1024, 1024]);  add_51 = None
    view_223: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_222, [4, 8, 1024, 1024]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_9: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_223, [-1], True)
    sub_14: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_223, amax_9);  view_223 = amax_9 = None
    exp_9: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_10: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_13: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_34: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_73: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_38: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_73, [4, 8, 1024, 1024]);  clone_73 = None
    view_224: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_38, [32, 1024, 1024]);  expand_38 = None
    expand_39: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_102, [4, 8, 1024, 64])
    clone_74: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_225: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_74, [32, 1024, 64]);  clone_74 = None
    bmm_19: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_224, view_225)
    view_226: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_19, [4, 8, 1024, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_104: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_75: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_227: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_75, [4, -1, 512]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_105: "f32[512, 512]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    view_228: "f32[4096, 512]" = torch.ops.aten.view.default(view_227, [4096, 512]);  view_227 = None
    mm_53: "f32[4096, 512]" = torch.ops.aten.mm.default(view_228, permute_105)
    view_229: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_53, [4, 1024, 512]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_76: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_229);  view_229 = None
    add_52: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_49, clone_76);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_19: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 2)
    mean_18: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_53: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
    rsqrt_18: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    alias_35: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_18)
    mul_43: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_44: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_19, mul_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_106: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    view_230: "f32[4096, 512]" = torch.ops.aten.view.default(mul_44, [4096, 512]);  mul_44 = None
    mm_54: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_230, permute_106)
    view_231: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_54, [4, 1024, 2048]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_7: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_231);  view_231 = None
    alias_36: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_77: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_7);  relu_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_107: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    view_232: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_77, [4096, 2048]);  clone_77 = None
    mm_55: "f32[4096, 512]" = torch.ops.aten.mm.default(view_232, permute_107)
    view_233: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_55, [4, 1024, 512]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_78: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_233);  view_233 = None
    add_54: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_52, clone_78);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_20: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 2)
    mean_19: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_55: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
    rsqrt_19: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    alias_37: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_19)
    mul_45: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_46: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_20, mul_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_108: "f32[512, 512]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    view_234: "f32[4096, 512]" = torch.ops.aten.view.default(mul_46, [4096, 512])
    mm_56: "f32[4096, 512]" = torch.ops.aten.mm.default(view_234, permute_108)
    view_235: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_56, [4, 1024, 512]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_236: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_235, [4, -1, 8, 64]);  view_235 = None
    permute_109: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_110: "f32[512, 512]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    view_237: "f32[4096, 512]" = torch.ops.aten.view.default(mul_46, [4096, 512])
    mm_57: "f32[4096, 512]" = torch.ops.aten.mm.default(view_237, permute_110)
    view_238: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_57, [4, 1024, 512]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_239: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_238, [4, -1, 8, 64]);  view_238 = None
    permute_111: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_112: "f32[512, 512]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    view_240: "f32[4096, 512]" = torch.ops.aten.view.default(mul_46, [4096, 512]);  mul_46 = None
    mm_58: "f32[4096, 512]" = torch.ops.aten.mm.default(view_240, permute_112)
    view_241: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_58, [4, 1024, 512]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_242: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_241, [4, -1, 8, 64]);  view_241 = None
    permute_113: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_114: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_40: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_109, [4, 8, 1024, 64]);  permute_109 = None
    clone_79: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_243: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_79, [32, 1024, 64]);  clone_79 = None
    expand_41: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_114, [4, 8, 64, 1024]);  permute_114 = None
    clone_80: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_244: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_80, [32, 64, 1024]);  clone_80 = None
    bmm_20: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_243, view_244)
    view_245: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_20, [4, 8, 1024, 1024]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_56: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_245, add_38);  view_245 = None
    view_246: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_56, [32, 1024, 1024]);  add_56 = None
    view_247: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_246, [4, 8, 1024, 1024]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_10: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_247, [-1], True)
    sub_15: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_247, amax_10);  view_247 = amax_10 = None
    exp_10: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_11: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_14: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_38: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_81: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_42: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_81, [4, 8, 1024, 1024]);  clone_81 = None
    view_248: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_42, [32, 1024, 1024]);  expand_42 = None
    expand_43: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_113, [4, 8, 1024, 64])
    clone_82: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_249: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_82, [32, 1024, 64]);  clone_82 = None
    bmm_21: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_248, view_249)
    view_250: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_21, [4, 8, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_115: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    clone_83: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_251: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_83, [4, -1, 512]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    view_252: "f32[4096, 512]" = torch.ops.aten.view.default(view_251, [4096, 512]);  view_251 = None
    mm_59: "f32[4096, 512]" = torch.ops.aten.mm.default(view_252, permute_116)
    view_253: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_59, [4, 1024, 512]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_84: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_253);  view_253 = None
    add_57: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_54, clone_84);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_21: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_57, 2)
    mean_20: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_58: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
    rsqrt_20: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    alias_39: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_20)
    mul_47: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_57, rsqrt_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_48: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_21, mul_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_117: "f32[512, 512]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    view_254: "f32[4096, 512]" = torch.ops.aten.view.default(mul_48, [4096, 512]);  mul_48 = None
    mm_60: "f32[4096, 512]" = torch.ops.aten.mm.default(view_254, permute_117)
    view_255: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_60, [4, 1024, 512]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_256: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_255, [4, -1, 8, 64]);  view_255 = None
    permute_118: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_119: "f32[512, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    view_257: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_61: "f32[4096, 512]" = torch.ops.aten.mm.default(view_257, permute_119)
    view_258: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_61, [4, 1024, 512]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_259: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_258, [4, -1, 8, 64]);  view_258 = None
    permute_120: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_121: "f32[512, 512]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    view_260: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_62: "f32[4096, 512]" = torch.ops.aten.mm.default(view_260, permute_121)
    view_261: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_62, [4, 1024, 512]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_262: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_261, [4, -1, 8, 64]);  view_261 = None
    permute_122: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_123: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_44: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_118, [4, 8, 1024, 64]);  permute_118 = None
    clone_85: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_263: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_85, [32, 1024, 64]);  clone_85 = None
    expand_45: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_123, [4, 8, 64, 1024]);  permute_123 = None
    clone_86: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_264: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_86, [32, 64, 1024]);  clone_86 = None
    bmm_22: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_263, view_264)
    view_265: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, [4, 8, 1024, 1024]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_59: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_265, add_42);  view_265 = None
    view_266: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_59, [32, 1024, 1024]);  add_59 = None
    view_267: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_266, [4, 8, 1024, 1024]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_11: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_267, [-1], True)
    sub_16: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_267, amax_11);  view_267 = amax_11 = None
    exp_11: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_12: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_15: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_40: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_87: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_46: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_87, [4, 8, 1024, 1024]);  clone_87 = None
    view_268: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_46, [32, 1024, 1024]);  expand_46 = None
    expand_47: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_122, [4, 8, 1024, 64])
    clone_88: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_269: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_88, [32, 1024, 64]);  clone_88 = None
    bmm_23: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_268, view_269)
    view_270: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_23, [4, 8, 1024, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_124: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
    clone_89: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    view_271: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_89, [4, -1, 512]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_125: "f32[512, 512]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    view_272: "f32[4096, 512]" = torch.ops.aten.view.default(view_271, [4096, 512]);  view_271 = None
    mm_63: "f32[4096, 512]" = torch.ops.aten.mm.default(view_272, permute_125)
    view_273: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_63, [4, 1024, 512]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_90: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_273);  view_273 = None
    add_60: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_57, clone_90);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_22: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 2)
    mean_21: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_61: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
    rsqrt_21: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    alias_41: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_21)
    mul_49: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_50: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_22, mul_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_126: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    view_274: "f32[4096, 512]" = torch.ops.aten.view.default(mul_50, [4096, 512]);  mul_50 = None
    mm_64: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_274, permute_126)
    view_275: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_64, [4, 1024, 2048]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_8: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_275);  view_275 = None
    alias_42: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_91: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_8);  relu_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_127: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    view_276: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_91, [4096, 2048]);  clone_91 = None
    mm_65: "f32[4096, 512]" = torch.ops.aten.mm.default(view_276, permute_127)
    view_277: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_65, [4, 1024, 512]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_92: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_277);  view_277 = None
    add_62: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_60, clone_92);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_23: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
    mean_22: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_63: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
    rsqrt_22: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    alias_43: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_22)
    mul_51: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_52: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_23, mul_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_128: "f32[512, 512]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    view_278: "f32[4096, 512]" = torch.ops.aten.view.default(mul_52, [4096, 512])
    mm_66: "f32[4096, 512]" = torch.ops.aten.mm.default(view_278, permute_128)
    view_279: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_66, [4, 1024, 512]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_280: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_279, [4, -1, 8, 64]);  view_279 = None
    permute_129: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    view_281: "f32[4096, 512]" = torch.ops.aten.view.default(mul_52, [4096, 512])
    mm_67: "f32[4096, 512]" = torch.ops.aten.mm.default(view_281, permute_130)
    view_282: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_67, [4, 1024, 512]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_283: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_282, [4, -1, 8, 64]);  view_282 = None
    permute_131: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_132: "f32[512, 512]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    view_284: "f32[4096, 512]" = torch.ops.aten.view.default(mul_52, [4096, 512]);  mul_52 = None
    mm_68: "f32[4096, 512]" = torch.ops.aten.mm.default(view_284, permute_132)
    view_285: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_68, [4, 1024, 512]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_286: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_285, [4, -1, 8, 64]);  view_285 = None
    permute_133: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_134: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_48: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_129, [4, 8, 1024, 64]);  permute_129 = None
    clone_93: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_287: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_93, [32, 1024, 64]);  clone_93 = None
    expand_49: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_134, [4, 8, 64, 1024]);  permute_134 = None
    clone_94: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_288: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_94, [32, 64, 1024]);  clone_94 = None
    bmm_24: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_287, view_288)
    view_289: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_24, [4, 8, 1024, 1024]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_64: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_289, add_38);  view_289 = None
    view_290: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_64, [32, 1024, 1024]);  add_64 = None
    view_291: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_290, [4, 8, 1024, 1024]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_12: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_291, [-1], True)
    sub_17: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_291, amax_12);  view_291 = amax_12 = None
    exp_12: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_13: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_16: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_44: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_95: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_50: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_95, [4, 8, 1024, 1024]);  clone_95 = None
    view_292: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_50, [32, 1024, 1024]);  expand_50 = None
    expand_51: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_133, [4, 8, 1024, 64])
    clone_96: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_293: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_96, [32, 1024, 64]);  clone_96 = None
    bmm_25: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_292, view_293)
    view_294: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_25, [4, 8, 1024, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_135: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    clone_97: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    view_295: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_97, [4, -1, 512]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_136: "f32[512, 512]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    view_296: "f32[4096, 512]" = torch.ops.aten.view.default(view_295, [4096, 512]);  view_295 = None
    mm_69: "f32[4096, 512]" = torch.ops.aten.mm.default(view_296, permute_136)
    view_297: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_69, [4, 1024, 512]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_98: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_297);  view_297 = None
    add_65: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_62, clone_98);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_24: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_65, 2)
    mean_23: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_66: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
    rsqrt_23: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    alias_45: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_23)
    mul_53: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_54: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_24, mul_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_137: "f32[512, 512]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    view_298: "f32[4096, 512]" = torch.ops.aten.view.default(mul_54, [4096, 512]);  mul_54 = None
    mm_70: "f32[4096, 512]" = torch.ops.aten.mm.default(view_298, permute_137)
    view_299: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_70, [4, 1024, 512]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_300: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_299, [4, -1, 8, 64]);  view_299 = None
    permute_138: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_139: "f32[512, 512]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    view_301: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_71: "f32[4096, 512]" = torch.ops.aten.mm.default(view_301, permute_139)
    view_302: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_71, [4, 1024, 512]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_303: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_302, [4, -1, 8, 64]);  view_302 = None
    permute_140: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_141: "f32[512, 512]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    view_304: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_72: "f32[4096, 512]" = torch.ops.aten.mm.default(view_304, permute_141)
    view_305: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_72, [4, 1024, 512]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_306: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_305, [4, -1, 8, 64]);  view_305 = None
    permute_142: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_143: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_52: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_138, [4, 8, 1024, 64]);  permute_138 = None
    clone_99: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_307: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_99, [32, 1024, 64]);  clone_99 = None
    expand_53: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_143, [4, 8, 64, 1024]);  permute_143 = None
    clone_100: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_308: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_100, [32, 64, 1024]);  clone_100 = None
    bmm_26: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_307, view_308)
    view_309: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_26, [4, 8, 1024, 1024]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_67: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_309, add_42);  view_309 = None
    view_310: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_67, [32, 1024, 1024]);  add_67 = None
    view_311: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_310, [4, 8, 1024, 1024]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_13: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_311, [-1], True)
    sub_18: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_311, amax_13);  view_311 = amax_13 = None
    exp_13: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_14: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_17: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_46: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_101: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_54: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_101, [4, 8, 1024, 1024]);  clone_101 = None
    view_312: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_54, [32, 1024, 1024]);  expand_54 = None
    expand_55: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_142, [4, 8, 1024, 64])
    clone_102: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_313: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_102, [32, 1024, 64]);  clone_102 = None
    bmm_27: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_312, view_313)
    view_314: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_27, [4, 8, 1024, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_144: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    clone_103: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    view_315: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_103, [4, -1, 512]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_145: "f32[512, 512]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    view_316: "f32[4096, 512]" = torch.ops.aten.view.default(view_315, [4096, 512]);  view_315 = None
    mm_73: "f32[4096, 512]" = torch.ops.aten.mm.default(view_316, permute_145)
    view_317: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_73, [4, 1024, 512]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_104: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_317);  view_317 = None
    add_68: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_65, clone_104);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_25: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_68, 2)
    mean_24: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_69: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
    rsqrt_24: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    alias_47: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_24)
    mul_55: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_68, rsqrt_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_56: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_25, mul_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_146: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    view_318: "f32[4096, 512]" = torch.ops.aten.view.default(mul_56, [4096, 512]);  mul_56 = None
    mm_74: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_318, permute_146)
    view_319: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_74, [4, 1024, 2048]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_9: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_319);  view_319 = None
    alias_48: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_105: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_9);  relu_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_147: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    view_320: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_105, [4096, 2048]);  clone_105 = None
    mm_75: "f32[4096, 512]" = torch.ops.aten.mm.default(view_320, permute_147)
    view_321: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_75, [4, 1024, 512]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_106: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_321);  view_321 = None
    add_70: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_68, clone_106);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_26: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
    mean_25: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_71: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
    rsqrt_25: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    alias_49: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_25)
    mul_57: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_58: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_26, mul_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_148: "f32[512, 512]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    view_322: "f32[4096, 512]" = torch.ops.aten.view.default(mul_58, [4096, 512])
    mm_76: "f32[4096, 512]" = torch.ops.aten.mm.default(view_322, permute_148)
    view_323: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_76, [4, 1024, 512]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_324: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_323, [4, -1, 8, 64]);  view_323 = None
    permute_149: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_150: "f32[512, 512]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    view_325: "f32[4096, 512]" = torch.ops.aten.view.default(mul_58, [4096, 512])
    mm_77: "f32[4096, 512]" = torch.ops.aten.mm.default(view_325, permute_150)
    view_326: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_77, [4, 1024, 512]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_327: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_326, [4, -1, 8, 64]);  view_326 = None
    permute_151: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_152: "f32[512, 512]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_328: "f32[4096, 512]" = torch.ops.aten.view.default(mul_58, [4096, 512]);  mul_58 = None
    mm_78: "f32[4096, 512]" = torch.ops.aten.mm.default(view_328, permute_152)
    view_329: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_78, [4, 1024, 512]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_330: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_329, [4, -1, 8, 64]);  view_329 = None
    permute_153: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_154: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_56: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_149, [4, 8, 1024, 64]);  permute_149 = None
    clone_107: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_331: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_107, [32, 1024, 64]);  clone_107 = None
    expand_57: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_154, [4, 8, 64, 1024]);  permute_154 = None
    clone_108: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_332: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_108, [32, 64, 1024]);  clone_108 = None
    bmm_28: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_331, view_332)
    view_333: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_28, [4, 8, 1024, 1024]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_72: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_333, add_38);  view_333 = None
    view_334: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_72, [32, 1024, 1024]);  add_72 = None
    view_335: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_334, [4, 8, 1024, 1024]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_14: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_335, [-1], True)
    sub_19: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_335, amax_14);  view_335 = amax_14 = None
    exp_14: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_15: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_18: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_50: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_109: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_58: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_109, [4, 8, 1024, 1024]);  clone_109 = None
    view_336: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_58, [32, 1024, 1024]);  expand_58 = None
    expand_59: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_153, [4, 8, 1024, 64])
    clone_110: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_337: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_110, [32, 1024, 64]);  clone_110 = None
    bmm_29: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_336, view_337)
    view_338: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_29, [4, 8, 1024, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_155: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    clone_111: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    view_339: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_111, [4, -1, 512]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_156: "f32[512, 512]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    view_340: "f32[4096, 512]" = torch.ops.aten.view.default(view_339, [4096, 512]);  view_339 = None
    mm_79: "f32[4096, 512]" = torch.ops.aten.mm.default(view_340, permute_156)
    view_341: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_79, [4, 1024, 512]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_112: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_341);  view_341 = None
    add_73: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_70, clone_112);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_27: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_73, 2)
    mean_26: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_74: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
    rsqrt_26: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    alias_51: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_26)
    mul_59: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_73, rsqrt_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_60: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_27, mul_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_157: "f32[512, 512]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    view_342: "f32[4096, 512]" = torch.ops.aten.view.default(mul_60, [4096, 512]);  mul_60 = None
    mm_80: "f32[4096, 512]" = torch.ops.aten.mm.default(view_342, permute_157)
    view_343: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_80, [4, 1024, 512]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_344: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_343, [4, -1, 8, 64]);  view_343 = None
    permute_158: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_159: "f32[512, 512]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    view_345: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_81: "f32[4096, 512]" = torch.ops.aten.mm.default(view_345, permute_159)
    view_346: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_81, [4, 1024, 512]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_347: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_346, [4, -1, 8, 64]);  view_346 = None
    permute_160: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_161: "f32[512, 512]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    view_348: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_82: "f32[4096, 512]" = torch.ops.aten.mm.default(view_348, permute_161)
    view_349: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_82, [4, 1024, 512]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_350: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_349, [4, -1, 8, 64]);  view_349 = None
    permute_162: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_163: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_60: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_158, [4, 8, 1024, 64]);  permute_158 = None
    clone_113: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_351: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_113, [32, 1024, 64]);  clone_113 = None
    expand_61: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_163, [4, 8, 64, 1024]);  permute_163 = None
    clone_114: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_352: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_114, [32, 64, 1024]);  clone_114 = None
    bmm_30: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_351, view_352)
    view_353: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_30, [4, 8, 1024, 1024]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_75: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_353, add_42);  view_353 = None
    view_354: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_75, [32, 1024, 1024]);  add_75 = None
    view_355: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_354, [4, 8, 1024, 1024]);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_15: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_355, [-1], True)
    sub_20: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_355, amax_15);  view_355 = amax_15 = None
    exp_15: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_16: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_19: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_52: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_115: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_62: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_115, [4, 8, 1024, 1024]);  clone_115 = None
    view_356: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_62, [32, 1024, 1024]);  expand_62 = None
    expand_63: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_162, [4, 8, 1024, 64])
    clone_116: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_357: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_116, [32, 1024, 64]);  clone_116 = None
    bmm_31: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_356, view_357)
    view_358: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_31, [4, 8, 1024, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_164: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
    clone_117: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_359: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_117, [4, -1, 512]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_165: "f32[512, 512]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    view_360: "f32[4096, 512]" = torch.ops.aten.view.default(view_359, [4096, 512]);  view_359 = None
    mm_83: "f32[4096, 512]" = torch.ops.aten.mm.default(view_360, permute_165)
    view_361: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_83, [4, 1024, 512]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_118: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_361);  view_361 = None
    add_76: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_73, clone_118);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_28: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_76, 2)
    mean_27: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_77: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
    rsqrt_27: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    alias_53: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_27)
    mul_61: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_62: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_28, mul_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_166: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    view_362: "f32[4096, 512]" = torch.ops.aten.view.default(mul_62, [4096, 512]);  mul_62 = None
    mm_84: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_362, permute_166)
    view_363: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_84, [4, 1024, 2048]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_10: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_363);  view_363 = None
    alias_54: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_119: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_10);  relu_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_167: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    view_364: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_119, [4096, 2048]);  clone_119 = None
    mm_85: "f32[4096, 512]" = torch.ops.aten.mm.default(view_364, permute_167)
    view_365: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_85, [4, 1024, 512]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_120: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_365);  view_365 = None
    add_78: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_76, clone_120);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_29: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
    mean_28: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_79: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
    rsqrt_28: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    alias_55: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_28)
    mul_63: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_64: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_29, mul_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_168: "f32[512, 512]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    view_366: "f32[4096, 512]" = torch.ops.aten.view.default(mul_64, [4096, 512])
    mm_86: "f32[4096, 512]" = torch.ops.aten.mm.default(view_366, permute_168)
    view_367: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_86, [4, 1024, 512]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_368: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_367, [4, -1, 8, 64]);  view_367 = None
    permute_169: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_170: "f32[512, 512]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    view_369: "f32[4096, 512]" = torch.ops.aten.view.default(mul_64, [4096, 512])
    mm_87: "f32[4096, 512]" = torch.ops.aten.mm.default(view_369, permute_170)
    view_370: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_87, [4, 1024, 512]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_371: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_370, [4, -1, 8, 64]);  view_370 = None
    permute_171: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_172: "f32[512, 512]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    view_372: "f32[4096, 512]" = torch.ops.aten.view.default(mul_64, [4096, 512]);  mul_64 = None
    mm_88: "f32[4096, 512]" = torch.ops.aten.mm.default(view_372, permute_172)
    view_373: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_88, [4, 1024, 512]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_374: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_373, [4, -1, 8, 64]);  view_373 = None
    permute_173: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_174: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_64: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_169, [4, 8, 1024, 64]);  permute_169 = None
    clone_121: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_375: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_121, [32, 1024, 64]);  clone_121 = None
    expand_65: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_174, [4, 8, 64, 1024]);  permute_174 = None
    clone_122: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_376: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_122, [32, 64, 1024]);  clone_122 = None
    bmm_32: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_375, view_376)
    view_377: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_32, [4, 8, 1024, 1024]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_80: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_377, add_38);  view_377 = add_38 = None
    view_378: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_80, [32, 1024, 1024]);  add_80 = None
    view_379: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_378, [4, 8, 1024, 1024]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_16: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_379, [-1], True)
    sub_21: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_379, amax_16);  view_379 = amax_16 = None
    exp_16: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_17: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_20: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_56: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_123: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_66: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_123, [4, 8, 1024, 1024]);  clone_123 = None
    view_380: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_66, [32, 1024, 1024]);  expand_66 = None
    expand_67: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_173, [4, 8, 1024, 64])
    clone_124: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_381: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_124, [32, 1024, 64]);  clone_124 = None
    bmm_33: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_380, view_381)
    view_382: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_33, [4, 8, 1024, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_175: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
    clone_125: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    view_383: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_125, [4, -1, 512]);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_176: "f32[512, 512]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    view_384: "f32[4096, 512]" = torch.ops.aten.view.default(view_383, [4096, 512]);  view_383 = None
    mm_89: "f32[4096, 512]" = torch.ops.aten.mm.default(view_384, permute_176)
    view_385: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_89, [4, 1024, 512]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_126: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_385);  view_385 = None
    add_81: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_78, clone_126);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_30: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_81, 2)
    mean_29: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_82: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
    rsqrt_29: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    alias_57: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_29)
    mul_65: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_66: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_30, mul_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_177: "f32[512, 512]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    view_386: "f32[4096, 512]" = torch.ops.aten.view.default(mul_66, [4096, 512]);  mul_66 = None
    mm_90: "f32[4096, 512]" = torch.ops.aten.mm.default(view_386, permute_177)
    view_387: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_90, [4, 1024, 512]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_388: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_387, [4, -1, 8, 64]);  view_387 = None
    permute_178: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_179: "f32[512, 512]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    view_389: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_91: "f32[4096, 512]" = torch.ops.aten.mm.default(view_389, permute_179)
    view_390: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_91, [4, 1024, 512]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_391: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_390, [4, -1, 8, 64]);  view_390 = None
    permute_180: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_181: "f32[512, 512]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    view_392: "f32[4096, 512]" = torch.ops.aten.view.default(clone_49, [4096, 512])
    mm_92: "f32[4096, 512]" = torch.ops.aten.mm.default(view_392, permute_181)
    view_393: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_92, [4, 1024, 512]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_394: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_393, [4, -1, 8, 64]);  view_393 = None
    permute_182: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_183: "f32[4, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_68: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_178, [4, 8, 1024, 64]);  permute_178 = None
    clone_127: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_395: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_127, [32, 1024, 64]);  clone_127 = None
    expand_69: "f32[4, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_183, [4, 8, 64, 1024]);  permute_183 = None
    clone_128: "f32[4, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_396: "f32[32, 64, 1024]" = torch.ops.aten.view.default(clone_128, [32, 64, 1024]);  clone_128 = None
    bmm_34: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_395, view_396)
    view_397: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_34, [4, 8, 1024, 1024]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_83: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_397, add_42);  view_397 = add_42 = None
    view_398: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(add_83, [32, 1024, 1024]);  add_83 = None
    view_399: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(view_398, [4, 8, 1024, 1024]);  view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_17: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(view_399, [-1], True)
    sub_22: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_399, amax_17);  view_399 = amax_17 = None
    exp_17: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_18: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_21: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_58: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_129: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_70: "f32[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_129, [4, 8, 1024, 1024]);  clone_129 = None
    view_400: "f32[32, 1024, 1024]" = torch.ops.aten.view.default(expand_70, [32, 1024, 1024]);  expand_70 = None
    expand_71: "f32[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_182, [4, 8, 1024, 64])
    clone_130: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_401: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_130, [32, 1024, 64]);  clone_130 = None
    bmm_35: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(view_400, view_401)
    view_402: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_35, [4, 8, 1024, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_184: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    clone_131: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    view_403: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_131, [4, -1, 512]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_185: "f32[512, 512]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    view_404: "f32[4096, 512]" = torch.ops.aten.view.default(view_403, [4096, 512]);  view_403 = None
    mm_93: "f32[4096, 512]" = torch.ops.aten.mm.default(view_404, permute_185)
    view_405: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_93, [4, 1024, 512]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_132: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_405);  view_405 = None
    add_84: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_81, clone_132);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_31: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
    mean_30: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_85: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
    rsqrt_30: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    alias_59: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_30)
    mul_67: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_68: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_31, mul_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_186: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    view_406: "f32[4096, 512]" = torch.ops.aten.view.default(mul_68, [4096, 512]);  mul_68 = None
    mm_94: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_406, permute_186)
    view_407: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_94, [4, 1024, 2048]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_11: "f32[4, 1024, 2048]" = torch.ops.aten.relu.default(view_407);  view_407 = None
    alias_60: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(relu_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_133: "f32[4, 1024, 2048]" = torch.ops.aten.clone.default(relu_11);  relu_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_187: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    view_408: "f32[4096, 2048]" = torch.ops.aten.view.default(clone_133, [4096, 2048]);  clone_133 = None
    mm_95: "f32[4096, 512]" = torch.ops.aten.mm.default(view_408, permute_187)
    view_409: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_95, [4, 1024, 512]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_134: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_409);  view_409 = None
    add_86: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_84, clone_134);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_32: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_86, 2)
    mean_31: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_87: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
    rsqrt_31: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    alias_61: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(rsqrt_31)
    mul_69: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_70: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_32, mul_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    clone_135: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1772, code: sequence_output = sequence_output * (self.model_dim**-0.5)
    mul_71: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(clone_135, 0.04419417382415922);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1774, code: lm_logits = self.lm_head(sequence_output)
    permute_188: "f32[512, 32128]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    view_410: "f32[4096, 512]" = torch.ops.aten.view.default(mul_71, [4096, 512]);  mul_71 = None
    mm_96: "f32[4096, 32128]" = torch.ops.aten.mm.default(view_410, permute_188)
    view_411: "f32[4, 1024, 32128]" = torch.ops.aten.view.default(mm_96, [4, 1024, 32128]);  mm_96 = None
    view_412: "f32[4096, 32128]" = torch.ops.aten.view.default(tangents_1, [4096, 32128]);  tangents_1 = None
    permute_189: "f32[32128, 4096]" = torch.ops.aten.permute.default(view_412, [1, 0])
    mm_97: "f32[32128, 512]" = torch.ops.aten.mm.default(permute_189, view_410);  permute_189 = view_410 = None
    permute_190: "f32[512, 32128]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    permute_191: "f32[32128, 512]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    mm_98: "f32[4096, 512]" = torch.ops.aten.mm.default(view_412, permute_191);  view_412 = permute_191 = None
    view_413: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_98, [4, 1024, 512]);  mm_98 = None
    permute_192: "f32[32128, 512]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1772, code: sequence_output = sequence_output * (self.model_dim**-0.5)
    mul_72: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_413, 0.04419417382415922);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_73: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_72, primals_32);  primals_32 = None
    mul_74: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_72, mul_69);  mul_72 = mul_69 = None
    sum_19: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_74, [0, 1], True);  mul_74 = None
    view_414: "f32[512]" = torch.ops.aten.view.default(sum_19, [512]);  sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_75: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_73, add_86)
    mul_76: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_73, rsqrt_31);  mul_73 = rsqrt_31 = None
    sum_20: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_75, [2], True);  mul_75 = None
    alias_62: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    pow_33: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_62, 3);  alias_62 = None
    mul_77: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_20, -0.5);  sum_20 = None
    mul_78: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_77, pow_33);  mul_77 = pow_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_72: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_78, [4, 1024, 512]);  mul_78 = None
    div_22: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_72, 512);  expand_72 = None
    pow_34: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_86, 1.0);  add_86 = None
    mul_79: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
    mul_80: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_22, mul_79);  div_22 = mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_88: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(mul_76, mul_80);  mul_76 = mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_415: "f32[4096, 512]" = torch.ops.aten.view.default(add_88, [4096, 512])
    permute_193: "f32[512, 4096]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_99: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_193, view_408);  permute_193 = view_408 = None
    permute_194: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    permute_195: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    mm_100: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_415, permute_195);  view_415 = permute_195 = None
    view_416: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_100, [4, 1024, 2048]);  mm_100 = None
    permute_196: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_63: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_1: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_63, 0);  alias_63 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_1, scalar_tensor, view_416);  le_1 = scalar_tensor = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_417: "f32[4096, 2048]" = torch.ops.aten.view.default(where_2, [4096, 2048]);  where_2 = None
    permute_197: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_417, [1, 0])
    mm_101: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_197, view_406);  permute_197 = view_406 = None
    permute_198: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    permute_199: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    mm_102: "f32[4096, 512]" = torch.ops.aten.mm.default(view_417, permute_199);  view_417 = permute_199 = None
    view_418: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_102, [4, 1024, 512]);  mm_102 = None
    permute_200: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_81: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_418, primals_31);  primals_31 = None
    mul_82: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_418, mul_67);  view_418 = mul_67 = None
    sum_21: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_82, [0, 1], True);  mul_82 = None
    view_419: "f32[512]" = torch.ops.aten.view.default(sum_21, [512]);  sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_83: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_81, add_84)
    mul_84: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_81, rsqrt_30);  mul_81 = rsqrt_30 = None
    sum_22: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_83, [2], True);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_89: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_88, mul_84);  add_88 = mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_64: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    pow_35: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_64, 3);  alias_64 = None
    mul_85: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_22, -0.5);  sum_22 = None
    mul_86: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_85, pow_35);  mul_85 = pow_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_73: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_86, [4, 1024, 512]);  mul_86 = None
    div_23: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_73, 512);  expand_73 = None
    pow_36: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 1.0);  add_84 = None
    mul_87: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_36, 2.0);  pow_36 = None
    mul_88: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_23, mul_87);  div_23 = mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_90: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_89, mul_88);  add_89 = mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_420: "f32[4096, 512]" = torch.ops.aten.view.default(add_90, [4096, 512])
    permute_201: "f32[512, 4096]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_103: "f32[512, 512]" = torch.ops.aten.mm.default(permute_201, view_404);  permute_201 = view_404 = None
    permute_202: "f32[512, 512]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    permute_203: "f32[512, 512]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    mm_104: "f32[4096, 512]" = torch.ops.aten.mm.default(view_420, permute_203);  view_420 = permute_203 = None
    view_421: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_104, [4, 1024, 512]);  mm_104 = None
    permute_204: "f32[512, 512]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_422: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_421, [4, 1024, 8, 64]);  view_421 = None
    permute_205: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_136: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    view_423: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_136, [32, 1024, 64]);  clone_136 = None
    permute_206: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_400, [0, 2, 1]);  view_400 = None
    bmm_36: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_206, view_423);  permute_206 = None
    permute_207: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_401, [0, 2, 1]);  view_401 = None
    bmm_37: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_423, permute_207);  view_423 = permute_207 = None
    view_424: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_36, [4, 8, 1024, 64]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_91: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_25, view_424);  tangents_25 = view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_425: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_37, [4, 8, 1024, 1024]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_65: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    mul_89: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_425, alias_65);  view_425 = None
    sum_23: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [-1], True)
    mul_90: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_65, sum_23);  alias_65 = sum_23 = None
    sub_23: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_7: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_7, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided, sub_23);  as_strided = sub_23 = None
    as_strided_scatter: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_7, copy, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_7 = copy = None
    as_strided_3: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter = None
    new_empty_strided: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_3, [32, 1024, 1024], [1048576, 1024, 1])
    copy_1: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided, as_strided_3);  new_empty_strided = as_strided_3 = None
    as_strided_5: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_1, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_137: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_5, memory_format = torch.contiguous_format)
    copy_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_5, clone_137);  as_strided_5 = clone_137 = None
    as_strided_scatter_1: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_1, copy_2, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_1 = copy_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_208: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_395, [0, 2, 1]);  view_395 = None
    bmm_38: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_208, as_strided_scatter_1);  permute_208 = None
    permute_209: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_396, [0, 2, 1]);  view_396 = None
    bmm_39: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_1, permute_209);  as_strided_scatter_1 = permute_209 = None
    view_426: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_38, [4, 8, 64, 1024]);  bmm_38 = None
    view_427: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_39, [4, 8, 1024, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_210: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_426, [0, 1, 3, 2]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_92: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_24, permute_210);  tangents_24 = permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_211: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_91, [0, 2, 1, 3]);  add_91 = None
    clone_138: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
    view_428: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_138, [4, 1024, 512]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_429: "f32[4096, 512]" = torch.ops.aten.view.default(view_428, [4096, 512]);  view_428 = None
    permute_212: "f32[512, 4096]" = torch.ops.aten.permute.default(view_429, [1, 0])
    mm_105: "f32[512, 512]" = torch.ops.aten.mm.default(permute_212, view_392);  permute_212 = view_392 = None
    permute_213: "f32[512, 512]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    permute_214: "f32[512, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    mm_106: "f32[4096, 512]" = torch.ops.aten.mm.default(view_429, permute_214);  view_429 = permute_214 = None
    view_430: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_106, [4, 1024, 512]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_93: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(tangents_26, view_430);  tangents_26 = view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_215: "f32[512, 512]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_216: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_92, [0, 2, 1, 3]);  add_92 = None
    clone_139: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    view_431: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_139, [4, 1024, 512]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_432: "f32[4096, 512]" = torch.ops.aten.view.default(view_431, [4096, 512]);  view_431 = None
    permute_217: "f32[512, 4096]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_107: "f32[512, 512]" = torch.ops.aten.mm.default(permute_217, view_389);  permute_217 = view_389 = None
    permute_218: "f32[512, 512]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    permute_219: "f32[512, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    mm_108: "f32[4096, 512]" = torch.ops.aten.mm.default(view_432, permute_219);  view_432 = permute_219 = None
    view_433: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_108, [4, 1024, 512]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_94: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_93, view_433);  add_93 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_220: "f32[512, 512]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_221: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
    clone_140: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    view_434: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_140, [4, 1024, 512]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_435: "f32[4096, 512]" = torch.ops.aten.view.default(view_434, [4096, 512]);  view_434 = None
    permute_222: "f32[512, 4096]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_109: "f32[512, 512]" = torch.ops.aten.mm.default(permute_222, view_386);  permute_222 = view_386 = None
    permute_223: "f32[512, 512]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    permute_224: "f32[512, 512]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    mm_110: "f32[4096, 512]" = torch.ops.aten.mm.default(view_435, permute_224);  view_435 = permute_224 = None
    view_436: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_110, [4, 1024, 512]);  mm_110 = None
    permute_225: "f32[512, 512]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_91: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_436, primals_30);  primals_30 = None
    mul_92: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_436, mul_65);  view_436 = mul_65 = None
    sum_24: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_92, [0, 1], True);  mul_92 = None
    view_437: "f32[512]" = torch.ops.aten.view.default(sum_24, [512]);  sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_93: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_91, add_81)
    mul_94: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_91, rsqrt_29);  mul_91 = rsqrt_29 = None
    sum_25: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_93, [2], True);  mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_95: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_90, mul_94);  add_90 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_66: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    pow_37: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_66, 3);  alias_66 = None
    mul_95: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_25, -0.5);  sum_25 = None
    mul_96: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_95, pow_37);  mul_95 = pow_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_74: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_96, [4, 1024, 512]);  mul_96 = None
    div_24: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_74, 512);  expand_74 = None
    pow_38: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_81, 1.0);  add_81 = None
    mul_97: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
    mul_98: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_24, mul_97);  div_24 = mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_96: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_95, mul_98);  add_95 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_438: "f32[4096, 512]" = torch.ops.aten.view.default(add_96, [4096, 512])
    permute_226: "f32[512, 4096]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_111: "f32[512, 512]" = torch.ops.aten.mm.default(permute_226, view_384);  permute_226 = view_384 = None
    permute_227: "f32[512, 512]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    permute_228: "f32[512, 512]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    mm_112: "f32[4096, 512]" = torch.ops.aten.mm.default(view_438, permute_228);  view_438 = permute_228 = None
    view_439: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_112, [4, 1024, 512]);  mm_112 = None
    permute_229: "f32[512, 512]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_440: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_439, [4, 1024, 8, 64]);  view_439 = None
    permute_230: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_141: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    view_441: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_141, [32, 1024, 64]);  clone_141 = None
    permute_231: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None
    bmm_40: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_231, view_441);  permute_231 = None
    permute_232: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
    bmm_41: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_441, permute_232);  view_441 = permute_232 = None
    view_442: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_40, [4, 8, 1024, 64]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_97: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_23, view_442);  tangents_23 = view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_443: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_41, [4, 8, 1024, 1024]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_67: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    mul_99: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_443, alias_67);  view_443 = None
    sum_26: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_99, [-1], True)
    mul_100: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_67, sum_26);  alias_67 = sum_26 = None
    sub_24: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_8: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_7: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_8, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_3: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_7, sub_24);  as_strided_7 = sub_24 = None
    as_strided_scatter_2: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_8, copy_3, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_8 = copy_3 = None
    as_strided_10: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_2, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_2 = None
    new_empty_strided_1: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_10, [32, 1024, 1024], [1048576, 1024, 1])
    copy_4: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_1, as_strided_10);  new_empty_strided_1 = as_strided_10 = None
    as_strided_12: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_4, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_142: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_12, memory_format = torch.contiguous_format)
    copy_5: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_12, clone_142);  as_strided_12 = None
    as_strided_scatter_3: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_4, copy_5, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_4 = copy_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_233: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
    bmm_42: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_233, as_strided_scatter_3);  permute_233 = None
    permute_234: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1]);  view_376 = None
    bmm_43: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_3, permute_234);  as_strided_scatter_3 = permute_234 = None
    view_444: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_42, [4, 8, 64, 1024]);  bmm_42 = None
    view_445: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_43, [4, 8, 1024, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_235: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_444, [0, 1, 3, 2]);  view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_98: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_22, permute_235);  tangents_22 = permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_236: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_97, [0, 2, 1, 3]);  add_97 = None
    clone_143: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    view_446: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_143, [4, 1024, 512]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_447: "f32[4096, 512]" = torch.ops.aten.view.default(view_446, [4096, 512]);  view_446 = None
    permute_237: "f32[512, 4096]" = torch.ops.aten.permute.default(view_447, [1, 0])
    mm_113: "f32[512, 512]" = torch.ops.aten.mm.default(permute_237, view_372);  permute_237 = view_372 = None
    permute_238: "f32[512, 512]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    permute_239: "f32[512, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    mm_114: "f32[4096, 512]" = torch.ops.aten.mm.default(view_447, permute_239);  view_447 = permute_239 = None
    view_448: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_114, [4, 1024, 512]);  mm_114 = None
    permute_240: "f32[512, 512]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_241: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_98, [0, 2, 1, 3]);  add_98 = None
    clone_144: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    view_449: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_144, [4, 1024, 512]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_450: "f32[4096, 512]" = torch.ops.aten.view.default(view_449, [4096, 512]);  view_449 = None
    permute_242: "f32[512, 4096]" = torch.ops.aten.permute.default(view_450, [1, 0])
    mm_115: "f32[512, 512]" = torch.ops.aten.mm.default(permute_242, view_369);  permute_242 = view_369 = None
    permute_243: "f32[512, 512]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    permute_244: "f32[512, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    mm_116: "f32[4096, 512]" = torch.ops.aten.mm.default(view_450, permute_244);  view_450 = permute_244 = None
    view_451: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_116, [4, 1024, 512]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_99: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_448, view_451);  view_448 = view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_245: "f32[512, 512]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_246: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    clone_145: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    view_452: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_145, [4, 1024, 512]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_453: "f32[4096, 512]" = torch.ops.aten.view.default(view_452, [4096, 512]);  view_452 = None
    permute_247: "f32[512, 4096]" = torch.ops.aten.permute.default(view_453, [1, 0])
    mm_117: "f32[512, 512]" = torch.ops.aten.mm.default(permute_247, view_366);  permute_247 = view_366 = None
    permute_248: "f32[512, 512]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    permute_249: "f32[512, 512]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    mm_118: "f32[4096, 512]" = torch.ops.aten.mm.default(view_453, permute_249);  view_453 = permute_249 = None
    view_454: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_118, [4, 1024, 512]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_100: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_99, view_454);  add_99 = view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_250: "f32[512, 512]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_101: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_100, primals_29);  primals_29 = None
    mul_102: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_100, mul_63);  add_100 = mul_63 = None
    sum_27: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 1], True);  mul_102 = None
    view_455: "f32[512]" = torch.ops.aten.view.default(sum_27, [512]);  sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_103: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_101, add_78)
    mul_104: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_101, rsqrt_28);  mul_101 = rsqrt_28 = None
    sum_28: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True);  mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_101: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_96, mul_104);  add_96 = mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_68: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    pow_39: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_68, 3);  alias_68 = None
    mul_105: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_28, -0.5);  sum_28 = None
    mul_106: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_105, pow_39);  mul_105 = pow_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_75: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_106, [4, 1024, 512]);  mul_106 = None
    div_25: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_75, 512);  expand_75 = None
    pow_40: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 1.0);  add_78 = None
    mul_107: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
    mul_108: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_25, mul_107);  div_25 = mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_102: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_101, mul_108);  add_101 = mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_456: "f32[4096, 512]" = torch.ops.aten.view.default(add_102, [4096, 512])
    permute_251: "f32[512, 4096]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_119: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_251, view_364);  permute_251 = view_364 = None
    permute_252: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    permute_253: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    mm_120: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_456, permute_253);  view_456 = permute_253 = None
    view_457: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_120, [4, 1024, 2048]);  mm_120 = None
    permute_254: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_69: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_2: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_69, 0);  alias_69 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_2, scalar_tensor_1, view_457);  le_2 = scalar_tensor_1 = view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_458: "f32[4096, 2048]" = torch.ops.aten.view.default(where_3, [4096, 2048]);  where_3 = None
    permute_255: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_458, [1, 0])
    mm_121: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_255, view_362);  permute_255 = view_362 = None
    permute_256: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    permute_257: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    mm_122: "f32[4096, 512]" = torch.ops.aten.mm.default(view_458, permute_257);  view_458 = permute_257 = None
    view_459: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_122, [4, 1024, 512]);  mm_122 = None
    permute_258: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_109: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_459, primals_28);  primals_28 = None
    mul_110: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_459, mul_61);  view_459 = mul_61 = None
    sum_29: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_110, [0, 1], True);  mul_110 = None
    view_460: "f32[512]" = torch.ops.aten.view.default(sum_29, [512]);  sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_111: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_109, add_76)
    mul_112: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_109, rsqrt_27);  mul_109 = rsqrt_27 = None
    sum_30: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_103: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_102, mul_112);  add_102 = mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_70: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    pow_41: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_70, 3);  alias_70 = None
    mul_113: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_30, -0.5);  sum_30 = None
    mul_114: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_113, pow_41);  mul_113 = pow_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_76: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_114, [4, 1024, 512]);  mul_114 = None
    div_26: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_76, 512);  expand_76 = None
    pow_42: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_76, 1.0);  add_76 = None
    mul_115: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
    mul_116: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_26, mul_115);  div_26 = mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_104: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_103, mul_116);  add_103 = mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_461: "f32[4096, 512]" = torch.ops.aten.view.default(add_104, [4096, 512])
    permute_259: "f32[512, 4096]" = torch.ops.aten.permute.default(view_461, [1, 0])
    mm_123: "f32[512, 512]" = torch.ops.aten.mm.default(permute_259, view_360);  permute_259 = view_360 = None
    permute_260: "f32[512, 512]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    permute_261: "f32[512, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    mm_124: "f32[4096, 512]" = torch.ops.aten.mm.default(view_461, permute_261);  view_461 = permute_261 = None
    view_462: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_124, [4, 1024, 512]);  mm_124 = None
    permute_262: "f32[512, 512]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_463: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_462, [4, 1024, 8, 64]);  view_462 = None
    permute_263: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_463, [0, 2, 1, 3]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_146: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
    view_464: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_146, [32, 1024, 64]);  clone_146 = None
    permute_264: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_356, [0, 2, 1]);  view_356 = None
    bmm_44: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_264, view_464);  permute_264 = None
    permute_265: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_357, [0, 2, 1]);  view_357 = None
    bmm_45: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_464, permute_265);  view_464 = permute_265 = None
    view_465: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_44, [4, 8, 1024, 64]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_105: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_21, view_465);  tangents_21 = view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_466: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_45, [4, 8, 1024, 1024]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_71: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    mul_117: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_466, alias_71);  view_466 = None
    sum_31: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [-1], True)
    mul_118: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_71, sum_31);  alias_71 = sum_31 = None
    sub_25: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_9: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_14: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_9, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_6: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_14, sub_25);  as_strided_14 = sub_25 = None
    as_strided_scatter_4: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_9, copy_6, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_9 = copy_6 = None
    as_strided_17: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_4, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_4 = None
    new_empty_strided_2: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_17, [32, 1024, 1024], [1048576, 1024, 1])
    copy_7: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_2, as_strided_17);  new_empty_strided_2 = as_strided_17 = None
    as_strided_19: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_7, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_147: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_19, memory_format = torch.contiguous_format)
    copy_8: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_19, clone_147);  as_strided_19 = clone_147 = None
    as_strided_scatter_5: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_7, copy_8, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_7 = copy_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_266: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
    bmm_46: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_266, as_strided_scatter_5);  permute_266 = None
    permute_267: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
    bmm_47: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_5, permute_267);  as_strided_scatter_5 = permute_267 = None
    view_467: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_46, [4, 8, 64, 1024]);  bmm_46 = None
    view_468: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_47, [4, 8, 1024, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_268: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_467, [0, 1, 3, 2]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_106: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_20, permute_268);  tangents_20 = permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_269: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_105, [0, 2, 1, 3]);  add_105 = None
    clone_148: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
    view_469: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_148, [4, 1024, 512]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_470: "f32[4096, 512]" = torch.ops.aten.view.default(view_469, [4096, 512]);  view_469 = None
    permute_270: "f32[512, 4096]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_125: "f32[512, 512]" = torch.ops.aten.mm.default(permute_270, view_348);  permute_270 = view_348 = None
    permute_271: "f32[512, 512]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    permute_272: "f32[512, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    mm_126: "f32[4096, 512]" = torch.ops.aten.mm.default(view_470, permute_272);  view_470 = permute_272 = None
    view_471: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_126, [4, 1024, 512]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_107: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_94, view_471);  add_94 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_273: "f32[512, 512]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_274: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_106, [0, 2, 1, 3]);  add_106 = None
    clone_149: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_274, memory_format = torch.contiguous_format);  permute_274 = None
    view_472: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_149, [4, 1024, 512]);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_473: "f32[4096, 512]" = torch.ops.aten.view.default(view_472, [4096, 512]);  view_472 = None
    permute_275: "f32[512, 4096]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_127: "f32[512, 512]" = torch.ops.aten.mm.default(permute_275, view_345);  permute_275 = view_345 = None
    permute_276: "f32[512, 512]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    permute_277: "f32[512, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    mm_128: "f32[4096, 512]" = torch.ops.aten.mm.default(view_473, permute_277);  view_473 = permute_277 = None
    view_474: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_128, [4, 1024, 512]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_108: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_107, view_474);  add_107 = view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_278: "f32[512, 512]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_279: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
    clone_150: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    view_475: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_150, [4, 1024, 512]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_476: "f32[4096, 512]" = torch.ops.aten.view.default(view_475, [4096, 512]);  view_475 = None
    permute_280: "f32[512, 4096]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_129: "f32[512, 512]" = torch.ops.aten.mm.default(permute_280, view_342);  permute_280 = view_342 = None
    permute_281: "f32[512, 512]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    permute_282: "f32[512, 512]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    mm_130: "f32[4096, 512]" = torch.ops.aten.mm.default(view_476, permute_282);  view_476 = permute_282 = None
    view_477: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_130, [4, 1024, 512]);  mm_130 = None
    permute_283: "f32[512, 512]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_119: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_477, primals_27);  primals_27 = None
    mul_120: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_477, mul_59);  view_477 = mul_59 = None
    sum_32: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_120, [0, 1], True);  mul_120 = None
    view_478: "f32[512]" = torch.ops.aten.view.default(sum_32, [512]);  sum_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_121: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_119, add_73)
    mul_122: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_119, rsqrt_26);  mul_119 = rsqrt_26 = None
    sum_33: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True);  mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_109: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_104, mul_122);  add_104 = mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_72: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    pow_43: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_72, 3);  alias_72 = None
    mul_123: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_33, -0.5);  sum_33 = None
    mul_124: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_123, pow_43);  mul_123 = pow_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_77: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_124, [4, 1024, 512]);  mul_124 = None
    div_27: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_77, 512);  expand_77 = None
    pow_44: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_73, 1.0);  add_73 = None
    mul_125: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
    mul_126: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_27, mul_125);  div_27 = mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_110: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_109, mul_126);  add_109 = mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_479: "f32[4096, 512]" = torch.ops.aten.view.default(add_110, [4096, 512])
    permute_284: "f32[512, 4096]" = torch.ops.aten.permute.default(view_479, [1, 0])
    mm_131: "f32[512, 512]" = torch.ops.aten.mm.default(permute_284, view_340);  permute_284 = view_340 = None
    permute_285: "f32[512, 512]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    permute_286: "f32[512, 512]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    mm_132: "f32[4096, 512]" = torch.ops.aten.mm.default(view_479, permute_286);  view_479 = permute_286 = None
    view_480: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_132, [4, 1024, 512]);  mm_132 = None
    permute_287: "f32[512, 512]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_481: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_480, [4, 1024, 8, 64]);  view_480 = None
    permute_288: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_151: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_482: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_151, [32, 1024, 64]);  clone_151 = None
    permute_289: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None
    bmm_48: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_289, view_482);  permute_289 = None
    permute_290: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
    bmm_49: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_482, permute_290);  view_482 = permute_290 = None
    view_483: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_48, [4, 8, 1024, 64]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_111: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_19, view_483);  tangents_19 = view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_484: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_49, [4, 8, 1024, 1024]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_73: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    mul_127: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_484, alias_73);  view_484 = None
    sum_34: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [-1], True)
    mul_128: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_73, sum_34);  alias_73 = sum_34 = None
    sub_26: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_10: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_21: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_10, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_9: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_21, sub_26);  as_strided_21 = sub_26 = None
    as_strided_scatter_6: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_10, copy_9, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_10 = copy_9 = None
    as_strided_24: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_6, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_6 = None
    new_empty_strided_3: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_24, [32, 1024, 1024], [1048576, 1024, 1])
    copy_10: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_3, as_strided_24);  new_empty_strided_3 = as_strided_24 = None
    as_strided_26: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_10, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_152: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_26, memory_format = torch.contiguous_format)
    copy_11: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_26, clone_152);  as_strided_26 = None
    as_strided_scatter_7: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_10, copy_11, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_10 = copy_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_112: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(clone_142, clone_152);  clone_142 = clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_291: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
    bmm_50: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_291, as_strided_scatter_7);  permute_291 = None
    permute_292: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None
    bmm_51: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_7, permute_292);  as_strided_scatter_7 = permute_292 = None
    view_485: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_50, [4, 8, 64, 1024]);  bmm_50 = None
    view_486: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_51, [4, 8, 1024, 64]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_293: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_485, [0, 1, 3, 2]);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_113: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_18, permute_293);  tangents_18 = permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_294: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_111, [0, 2, 1, 3]);  add_111 = None
    clone_153: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    view_487: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_153, [4, 1024, 512]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_488: "f32[4096, 512]" = torch.ops.aten.view.default(view_487, [4096, 512]);  view_487 = None
    permute_295: "f32[512, 4096]" = torch.ops.aten.permute.default(view_488, [1, 0])
    mm_133: "f32[512, 512]" = torch.ops.aten.mm.default(permute_295, view_328);  permute_295 = view_328 = None
    permute_296: "f32[512, 512]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    permute_297: "f32[512, 512]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    mm_134: "f32[4096, 512]" = torch.ops.aten.mm.default(view_488, permute_297);  view_488 = permute_297 = None
    view_489: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_134, [4, 1024, 512]);  mm_134 = None
    permute_298: "f32[512, 512]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_299: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_113, [0, 2, 1, 3]);  add_113 = None
    clone_154: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_490: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_154, [4, 1024, 512]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_491: "f32[4096, 512]" = torch.ops.aten.view.default(view_490, [4096, 512]);  view_490 = None
    permute_300: "f32[512, 4096]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_135: "f32[512, 512]" = torch.ops.aten.mm.default(permute_300, view_325);  permute_300 = view_325 = None
    permute_301: "f32[512, 512]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    permute_302: "f32[512, 512]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    mm_136: "f32[4096, 512]" = torch.ops.aten.mm.default(view_491, permute_302);  view_491 = permute_302 = None
    view_492: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_136, [4, 1024, 512]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_114: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_489, view_492);  view_489 = view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_303: "f32[512, 512]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_304: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
    clone_155: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
    view_493: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_155, [4, 1024, 512]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_494: "f32[4096, 512]" = torch.ops.aten.view.default(view_493, [4096, 512]);  view_493 = None
    permute_305: "f32[512, 4096]" = torch.ops.aten.permute.default(view_494, [1, 0])
    mm_137: "f32[512, 512]" = torch.ops.aten.mm.default(permute_305, view_322);  permute_305 = view_322 = None
    permute_306: "f32[512, 512]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    permute_307: "f32[512, 512]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    mm_138: "f32[4096, 512]" = torch.ops.aten.mm.default(view_494, permute_307);  view_494 = permute_307 = None
    view_495: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_138, [4, 1024, 512]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_115: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_114, view_495);  add_114 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_308: "f32[512, 512]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_129: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_115, primals_26);  primals_26 = None
    mul_130: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_115, mul_57);  add_115 = mul_57 = None
    sum_35: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_130, [0, 1], True);  mul_130 = None
    view_496: "f32[512]" = torch.ops.aten.view.default(sum_35, [512]);  sum_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_131: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_129, add_70)
    mul_132: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_129, rsqrt_25);  mul_129 = rsqrt_25 = None
    sum_36: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_116: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_110, mul_132);  add_110 = mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_74: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    pow_45: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_74, 3);  alias_74 = None
    mul_133: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_36, -0.5);  sum_36 = None
    mul_134: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_133, pow_45);  mul_133 = pow_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_78: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_134, [4, 1024, 512]);  mul_134 = None
    div_28: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_78, 512);  expand_78 = None
    pow_46: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 1.0);  add_70 = None
    mul_135: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
    mul_136: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_28, mul_135);  div_28 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_117: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_116, mul_136);  add_116 = mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_497: "f32[4096, 512]" = torch.ops.aten.view.default(add_117, [4096, 512])
    permute_309: "f32[512, 4096]" = torch.ops.aten.permute.default(view_497, [1, 0])
    mm_139: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_309, view_320);  permute_309 = view_320 = None
    permute_310: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    permute_311: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    mm_140: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_497, permute_311);  view_497 = permute_311 = None
    view_498: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_140, [4, 1024, 2048]);  mm_140 = None
    permute_312: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_75: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_3: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_75, 0);  alias_75 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_3, scalar_tensor_2, view_498);  le_3 = scalar_tensor_2 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_499: "f32[4096, 2048]" = torch.ops.aten.view.default(where_4, [4096, 2048]);  where_4 = None
    permute_313: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_499, [1, 0])
    mm_141: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_313, view_318);  permute_313 = view_318 = None
    permute_314: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    permute_315: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    mm_142: "f32[4096, 512]" = torch.ops.aten.mm.default(view_499, permute_315);  view_499 = permute_315 = None
    view_500: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_142, [4, 1024, 512]);  mm_142 = None
    permute_316: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_137: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_500, primals_25);  primals_25 = None
    mul_138: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_500, mul_55);  view_500 = mul_55 = None
    sum_37: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_138, [0, 1], True);  mul_138 = None
    view_501: "f32[512]" = torch.ops.aten.view.default(sum_37, [512]);  sum_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_139: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_137, add_68)
    mul_140: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_137, rsqrt_24);  mul_137 = rsqrt_24 = None
    sum_38: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_118: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_117, mul_140);  add_117 = mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_76: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    pow_47: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_76, 3);  alias_76 = None
    mul_141: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_38, -0.5);  sum_38 = None
    mul_142: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_141, pow_47);  mul_141 = pow_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_79: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_142, [4, 1024, 512]);  mul_142 = None
    div_29: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_79, 512);  expand_79 = None
    pow_48: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_68, 1.0);  add_68 = None
    mul_143: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
    mul_144: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_29, mul_143);  div_29 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_119: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_118, mul_144);  add_118 = mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_502: "f32[4096, 512]" = torch.ops.aten.view.default(add_119, [4096, 512])
    permute_317: "f32[512, 4096]" = torch.ops.aten.permute.default(view_502, [1, 0])
    mm_143: "f32[512, 512]" = torch.ops.aten.mm.default(permute_317, view_316);  permute_317 = view_316 = None
    permute_318: "f32[512, 512]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    permute_319: "f32[512, 512]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_144: "f32[4096, 512]" = torch.ops.aten.mm.default(view_502, permute_319);  view_502 = permute_319 = None
    view_503: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_144, [4, 1024, 512]);  mm_144 = None
    permute_320: "f32[512, 512]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_504: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_503, [4, 1024, 8, 64]);  view_503 = None
    permute_321: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_156: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_505: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_156, [32, 1024, 64]);  clone_156 = None
    permute_322: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
    bmm_52: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_322, view_505);  permute_322 = None
    permute_323: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    bmm_53: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_505, permute_323);  view_505 = permute_323 = None
    view_506: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_52, [4, 8, 1024, 64]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_120: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_17, view_506);  tangents_17 = view_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_507: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_53, [4, 8, 1024, 1024]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_77: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    mul_145: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_507, alias_77);  view_507 = None
    sum_39: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [-1], True)
    mul_146: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_77, sum_39);  alias_77 = sum_39 = None
    sub_27: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_11: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_28: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_11, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_12: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_28, sub_27);  as_strided_28 = sub_27 = None
    as_strided_scatter_8: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_11, copy_12, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_11 = copy_12 = None
    as_strided_31: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_8, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_8 = None
    new_empty_strided_4: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_31, [32, 1024, 1024], [1048576, 1024, 1])
    copy_13: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_4, as_strided_31);  new_empty_strided_4 = as_strided_31 = None
    as_strided_33: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_13, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_157: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_33, memory_format = torch.contiguous_format)
    copy_14: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_33, clone_157);  as_strided_33 = clone_157 = None
    as_strided_scatter_9: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_13, copy_14, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_13 = copy_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_324: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
    bmm_54: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_324, as_strided_scatter_9);  permute_324 = None
    permute_325: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
    bmm_55: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_9, permute_325);  as_strided_scatter_9 = permute_325 = None
    view_508: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_54, [4, 8, 64, 1024]);  bmm_54 = None
    view_509: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_55, [4, 8, 1024, 64]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_326: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_508, [0, 1, 3, 2]);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_121: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_16, permute_326);  tangents_16 = permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_327: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_120, [0, 2, 1, 3]);  add_120 = None
    clone_158: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    view_510: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_158, [4, 1024, 512]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_511: "f32[4096, 512]" = torch.ops.aten.view.default(view_510, [4096, 512]);  view_510 = None
    permute_328: "f32[512, 4096]" = torch.ops.aten.permute.default(view_511, [1, 0])
    mm_145: "f32[512, 512]" = torch.ops.aten.mm.default(permute_328, view_304);  permute_328 = view_304 = None
    permute_329: "f32[512, 512]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    permute_330: "f32[512, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    mm_146: "f32[4096, 512]" = torch.ops.aten.mm.default(view_511, permute_330);  view_511 = permute_330 = None
    view_512: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_146, [4, 1024, 512]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_122: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_108, view_512);  add_108 = view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_331: "f32[512, 512]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_332: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_121, [0, 2, 1, 3]);  add_121 = None
    clone_159: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_332, memory_format = torch.contiguous_format);  permute_332 = None
    view_513: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_159, [4, 1024, 512]);  clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_514: "f32[4096, 512]" = torch.ops.aten.view.default(view_513, [4096, 512]);  view_513 = None
    permute_333: "f32[512, 4096]" = torch.ops.aten.permute.default(view_514, [1, 0])
    mm_147: "f32[512, 512]" = torch.ops.aten.mm.default(permute_333, view_301);  permute_333 = view_301 = None
    permute_334: "f32[512, 512]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    permute_335: "f32[512, 512]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    mm_148: "f32[4096, 512]" = torch.ops.aten.mm.default(view_514, permute_335);  view_514 = permute_335 = None
    view_515: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_148, [4, 1024, 512]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_123: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_122, view_515);  add_122 = view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_336: "f32[512, 512]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_337: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None
    clone_160: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
    view_516: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_160, [4, 1024, 512]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_517: "f32[4096, 512]" = torch.ops.aten.view.default(view_516, [4096, 512]);  view_516 = None
    permute_338: "f32[512, 4096]" = torch.ops.aten.permute.default(view_517, [1, 0])
    mm_149: "f32[512, 512]" = torch.ops.aten.mm.default(permute_338, view_298);  permute_338 = view_298 = None
    permute_339: "f32[512, 512]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    permute_340: "f32[512, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    mm_150: "f32[4096, 512]" = torch.ops.aten.mm.default(view_517, permute_340);  view_517 = permute_340 = None
    view_518: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_150, [4, 1024, 512]);  mm_150 = None
    permute_341: "f32[512, 512]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_147: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_518, primals_24);  primals_24 = None
    mul_148: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_518, mul_53);  view_518 = mul_53 = None
    sum_40: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1], True);  mul_148 = None
    view_519: "f32[512]" = torch.ops.aten.view.default(sum_40, [512]);  sum_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_149: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_147, add_65)
    mul_150: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_147, rsqrt_23);  mul_147 = rsqrt_23 = None
    sum_41: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [2], True);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_124: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_119, mul_150);  add_119 = mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_78: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    pow_49: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_78, 3);  alias_78 = None
    mul_151: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_41, -0.5);  sum_41 = None
    mul_152: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_151, pow_49);  mul_151 = pow_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_80: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_152, [4, 1024, 512]);  mul_152 = None
    div_30: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_80, 512);  expand_80 = None
    pow_50: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_65, 1.0);  add_65 = None
    mul_153: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
    mul_154: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_30, mul_153);  div_30 = mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_125: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_124, mul_154);  add_124 = mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_520: "f32[4096, 512]" = torch.ops.aten.view.default(add_125, [4096, 512])
    permute_342: "f32[512, 4096]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_151: "f32[512, 512]" = torch.ops.aten.mm.default(permute_342, view_296);  permute_342 = view_296 = None
    permute_343: "f32[512, 512]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    permute_344: "f32[512, 512]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    mm_152: "f32[4096, 512]" = torch.ops.aten.mm.default(view_520, permute_344);  view_520 = permute_344 = None
    view_521: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_152, [4, 1024, 512]);  mm_152 = None
    permute_345: "f32[512, 512]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_522: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_521, [4, 1024, 8, 64]);  view_521 = None
    permute_346: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_161: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
    view_523: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_161, [32, 1024, 64]);  clone_161 = None
    permute_347: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
    bmm_56: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_347, view_523);  permute_347 = None
    permute_348: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
    bmm_57: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_523, permute_348);  view_523 = permute_348 = None
    view_524: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_56, [4, 8, 1024, 64]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_126: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_15, view_524);  tangents_15 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_525: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_57, [4, 8, 1024, 1024]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_79: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    mul_155: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_525, alias_79);  view_525 = None
    sum_42: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [-1], True)
    mul_156: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_79, sum_42);  alias_79 = sum_42 = None
    sub_28: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_12: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_35: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_12, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_15: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_35, sub_28);  as_strided_35 = sub_28 = None
    as_strided_scatter_10: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_12, copy_15, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_12 = copy_15 = None
    as_strided_38: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_10, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_10 = None
    new_empty_strided_5: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_38, [32, 1024, 1024], [1048576, 1024, 1])
    copy_16: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_5, as_strided_38);  new_empty_strided_5 = as_strided_38 = None
    as_strided_40: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_16, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_162: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_40, memory_format = torch.contiguous_format)
    copy_17: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_40, clone_162);  as_strided_40 = None
    as_strided_scatter_11: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_16, copy_17, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_16 = copy_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_127: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_112, clone_162);  add_112 = clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_349: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    bmm_58: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_349, as_strided_scatter_11);  permute_349 = None
    permute_350: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    bmm_59: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_11, permute_350);  as_strided_scatter_11 = permute_350 = None
    view_526: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_58, [4, 8, 64, 1024]);  bmm_58 = None
    view_527: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_59, [4, 8, 1024, 64]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_351: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_526, [0, 1, 3, 2]);  view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_128: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_14, permute_351);  tangents_14 = permute_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_352: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_126, [0, 2, 1, 3]);  add_126 = None
    clone_163: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
    view_528: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_163, [4, 1024, 512]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_529: "f32[4096, 512]" = torch.ops.aten.view.default(view_528, [4096, 512]);  view_528 = None
    permute_353: "f32[512, 4096]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_153: "f32[512, 512]" = torch.ops.aten.mm.default(permute_353, view_284);  permute_353 = view_284 = None
    permute_354: "f32[512, 512]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    permute_355: "f32[512, 512]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_154: "f32[4096, 512]" = torch.ops.aten.mm.default(view_529, permute_355);  view_529 = permute_355 = None
    view_530: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_154, [4, 1024, 512]);  mm_154 = None
    permute_356: "f32[512, 512]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_357: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_128, [0, 2, 1, 3]);  add_128 = None
    clone_164: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    view_531: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_164, [4, 1024, 512]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_532: "f32[4096, 512]" = torch.ops.aten.view.default(view_531, [4096, 512]);  view_531 = None
    permute_358: "f32[512, 4096]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_155: "f32[512, 512]" = torch.ops.aten.mm.default(permute_358, view_281);  permute_358 = view_281 = None
    permute_359: "f32[512, 512]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    permute_360: "f32[512, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_156: "f32[4096, 512]" = torch.ops.aten.mm.default(view_532, permute_360);  view_532 = permute_360 = None
    view_533: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_156, [4, 1024, 512]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_129: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_530, view_533);  view_530 = view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_361: "f32[512, 512]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_362: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
    clone_165: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_362, memory_format = torch.contiguous_format);  permute_362 = None
    view_534: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_165, [4, 1024, 512]);  clone_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_535: "f32[4096, 512]" = torch.ops.aten.view.default(view_534, [4096, 512]);  view_534 = None
    permute_363: "f32[512, 4096]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_157: "f32[512, 512]" = torch.ops.aten.mm.default(permute_363, view_278);  permute_363 = view_278 = None
    permute_364: "f32[512, 512]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    permute_365: "f32[512, 512]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_158: "f32[4096, 512]" = torch.ops.aten.mm.default(view_535, permute_365);  view_535 = permute_365 = None
    view_536: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_158, [4, 1024, 512]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_130: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_129, view_536);  add_129 = view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_366: "f32[512, 512]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_157: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_130, primals_23);  primals_23 = None
    mul_158: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_130, mul_51);  add_130 = mul_51 = None
    sum_43: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True);  mul_158 = None
    view_537: "f32[512]" = torch.ops.aten.view.default(sum_43, [512]);  sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_159: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_157, add_62)
    mul_160: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_157, rsqrt_22);  mul_157 = rsqrt_22 = None
    sum_44: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_131: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_125, mul_160);  add_125 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_80: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    pow_51: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_80, 3);  alias_80 = None
    mul_161: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_44, -0.5);  sum_44 = None
    mul_162: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_161, pow_51);  mul_161 = pow_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_81: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_162, [4, 1024, 512]);  mul_162 = None
    div_31: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_81, 512);  expand_81 = None
    pow_52: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 1.0);  add_62 = None
    mul_163: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_52, 2.0);  pow_52 = None
    mul_164: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_31, mul_163);  div_31 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_132: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_131, mul_164);  add_131 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_538: "f32[4096, 512]" = torch.ops.aten.view.default(add_132, [4096, 512])
    permute_367: "f32[512, 4096]" = torch.ops.aten.permute.default(view_538, [1, 0])
    mm_159: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_367, view_276);  permute_367 = view_276 = None
    permute_368: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    permute_369: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    mm_160: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_538, permute_369);  view_538 = permute_369 = None
    view_539: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_160, [4, 1024, 2048]);  mm_160 = None
    permute_370: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_81: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_4: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_81, 0);  alias_81 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_4, scalar_tensor_3, view_539);  le_4 = scalar_tensor_3 = view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_540: "f32[4096, 2048]" = torch.ops.aten.view.default(where_5, [4096, 2048]);  where_5 = None
    permute_371: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_540, [1, 0])
    mm_161: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_371, view_274);  permute_371 = view_274 = None
    permute_372: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    permute_373: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    mm_162: "f32[4096, 512]" = torch.ops.aten.mm.default(view_540, permute_373);  view_540 = permute_373 = None
    view_541: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_162, [4, 1024, 512]);  mm_162 = None
    permute_374: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_165: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_541, primals_22);  primals_22 = None
    mul_166: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_541, mul_49);  view_541 = mul_49 = None
    sum_45: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 1], True);  mul_166 = None
    view_542: "f32[512]" = torch.ops.aten.view.default(sum_45, [512]);  sum_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_167: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_165, add_60)
    mul_168: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_165, rsqrt_21);  mul_165 = rsqrt_21 = None
    sum_46: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_133: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_132, mul_168);  add_132 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_82: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    pow_53: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_82, 3);  alias_82 = None
    mul_169: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_46, -0.5);  sum_46 = None
    mul_170: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_169, pow_53);  mul_169 = pow_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_82: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_170, [4, 1024, 512]);  mul_170 = None
    div_32: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_82, 512);  expand_82 = None
    pow_54: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 1.0);  add_60 = None
    mul_171: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_54, 2.0);  pow_54 = None
    mul_172: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_32, mul_171);  div_32 = mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_134: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_133, mul_172);  add_133 = mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_543: "f32[4096, 512]" = torch.ops.aten.view.default(add_134, [4096, 512])
    permute_375: "f32[512, 4096]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_163: "f32[512, 512]" = torch.ops.aten.mm.default(permute_375, view_272);  permute_375 = view_272 = None
    permute_376: "f32[512, 512]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    permute_377: "f32[512, 512]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_164: "f32[4096, 512]" = torch.ops.aten.mm.default(view_543, permute_377);  view_543 = permute_377 = None
    view_544: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_164, [4, 1024, 512]);  mm_164 = None
    permute_378: "f32[512, 512]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_545: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_544, [4, 1024, 8, 64]);  view_544 = None
    permute_379: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_166: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_379, memory_format = torch.contiguous_format);  permute_379 = None
    view_546: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_166, [32, 1024, 64]);  clone_166 = None
    permute_380: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
    bmm_60: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_380, view_546);  permute_380 = None
    permute_381: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
    bmm_61: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_546, permute_381);  view_546 = permute_381 = None
    view_547: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_60, [4, 8, 1024, 64]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_135: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_13, view_547);  tangents_13 = view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_548: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_61, [4, 8, 1024, 1024]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_83: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    mul_173: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_548, alias_83);  view_548 = None
    sum_47: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [-1], True)
    mul_174: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_83, sum_47);  alias_83 = sum_47 = None
    sub_29: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_13: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_42: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_13, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_18: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_42, sub_29);  as_strided_42 = sub_29 = None
    as_strided_scatter_12: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_13, copy_18, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_13 = copy_18 = None
    as_strided_45: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_12, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_12 = None
    new_empty_strided_6: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_45, [32, 1024, 1024], [1048576, 1024, 1])
    copy_19: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_6, as_strided_45);  new_empty_strided_6 = as_strided_45 = None
    as_strided_47: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_19, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_167: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_47, memory_format = torch.contiguous_format)
    copy_20: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_47, clone_167);  as_strided_47 = clone_167 = None
    as_strided_scatter_13: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_19, copy_20, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_19 = copy_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_382: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
    bmm_62: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_382, as_strided_scatter_13);  permute_382 = None
    permute_383: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    bmm_63: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_13, permute_383);  as_strided_scatter_13 = permute_383 = None
    view_549: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_62, [4, 8, 64, 1024]);  bmm_62 = None
    view_550: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_63, [4, 8, 1024, 64]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_384: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_549, [0, 1, 3, 2]);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_136: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_12, permute_384);  tangents_12 = permute_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_385: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_135, [0, 2, 1, 3]);  add_135 = None
    clone_168: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_551: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_168, [4, 1024, 512]);  clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_552: "f32[4096, 512]" = torch.ops.aten.view.default(view_551, [4096, 512]);  view_551 = None
    permute_386: "f32[512, 4096]" = torch.ops.aten.permute.default(view_552, [1, 0])
    mm_165: "f32[512, 512]" = torch.ops.aten.mm.default(permute_386, view_260);  permute_386 = view_260 = None
    permute_387: "f32[512, 512]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    permute_388: "f32[512, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_166: "f32[4096, 512]" = torch.ops.aten.mm.default(view_552, permute_388);  view_552 = permute_388 = None
    view_553: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_166, [4, 1024, 512]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_137: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_123, view_553);  add_123 = view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_389: "f32[512, 512]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_390: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_136, [0, 2, 1, 3]);  add_136 = None
    clone_169: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
    view_554: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_169, [4, 1024, 512]);  clone_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_555: "f32[4096, 512]" = torch.ops.aten.view.default(view_554, [4096, 512]);  view_554 = None
    permute_391: "f32[512, 4096]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_167: "f32[512, 512]" = torch.ops.aten.mm.default(permute_391, view_257);  permute_391 = view_257 = None
    permute_392: "f32[512, 512]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    permute_393: "f32[512, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_168: "f32[4096, 512]" = torch.ops.aten.mm.default(view_555, permute_393);  view_555 = permute_393 = None
    view_556: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_168, [4, 1024, 512]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_138: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_137, view_556);  add_137 = view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_394: "f32[512, 512]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_395: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
    clone_170: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    view_557: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_170, [4, 1024, 512]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_558: "f32[4096, 512]" = torch.ops.aten.view.default(view_557, [4096, 512]);  view_557 = None
    permute_396: "f32[512, 4096]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_169: "f32[512, 512]" = torch.ops.aten.mm.default(permute_396, view_254);  permute_396 = view_254 = None
    permute_397: "f32[512, 512]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    permute_398: "f32[512, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    mm_170: "f32[4096, 512]" = torch.ops.aten.mm.default(view_558, permute_398);  view_558 = permute_398 = None
    view_559: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_170, [4, 1024, 512]);  mm_170 = None
    permute_399: "f32[512, 512]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_175: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_559, primals_21);  primals_21 = None
    mul_176: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_559, mul_47);  view_559 = mul_47 = None
    sum_48: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1], True);  mul_176 = None
    view_560: "f32[512]" = torch.ops.aten.view.default(sum_48, [512]);  sum_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_177: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_175, add_57)
    mul_178: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_175, rsqrt_20);  mul_175 = rsqrt_20 = None
    sum_49: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_139: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_134, mul_178);  add_134 = mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_84: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    pow_55: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_84, 3);  alias_84 = None
    mul_179: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_49, -0.5);  sum_49 = None
    mul_180: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_179, pow_55);  mul_179 = pow_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_83: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_180, [4, 1024, 512]);  mul_180 = None
    div_33: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_83, 512);  expand_83 = None
    pow_56: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_57, 1.0);  add_57 = None
    mul_181: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_56, 2.0);  pow_56 = None
    mul_182: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_33, mul_181);  div_33 = mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_140: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_139, mul_182);  add_139 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_561: "f32[4096, 512]" = torch.ops.aten.view.default(add_140, [4096, 512])
    permute_400: "f32[512, 4096]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_171: "f32[512, 512]" = torch.ops.aten.mm.default(permute_400, view_252);  permute_400 = view_252 = None
    permute_401: "f32[512, 512]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    permute_402: "f32[512, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_172: "f32[4096, 512]" = torch.ops.aten.mm.default(view_561, permute_402);  view_561 = permute_402 = None
    view_562: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_172, [4, 1024, 512]);  mm_172 = None
    permute_403: "f32[512, 512]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_563: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_562, [4, 1024, 8, 64]);  view_562 = None
    permute_404: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_563, [0, 2, 1, 3]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_171: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_404, memory_format = torch.contiguous_format);  permute_404 = None
    view_564: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_171, [32, 1024, 64]);  clone_171 = None
    permute_405: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
    bmm_64: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_405, view_564);  permute_405 = None
    permute_406: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
    bmm_65: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_564, permute_406);  view_564 = permute_406 = None
    view_565: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_64, [4, 8, 1024, 64]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_141: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_11, view_565);  tangents_11 = view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_566: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_65, [4, 8, 1024, 1024]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_85: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    mul_183: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_566, alias_85);  view_566 = None
    sum_50: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [-1], True)
    mul_184: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_85, sum_50);  alias_85 = sum_50 = None
    sub_30: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_14: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_49: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_14, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_21: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_49, sub_30);  as_strided_49 = sub_30 = None
    as_strided_scatter_14: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_14, copy_21, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_14 = copy_21 = None
    as_strided_52: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_14, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_14 = None
    new_empty_strided_7: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_52, [32, 1024, 1024], [1048576, 1024, 1])
    copy_22: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_7, as_strided_52);  new_empty_strided_7 = as_strided_52 = None
    as_strided_54: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_22, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_172: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_54, memory_format = torch.contiguous_format)
    copy_23: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_54, clone_172);  as_strided_54 = None
    as_strided_scatter_15: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_22, copy_23, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_22 = copy_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_142: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_127, clone_172);  add_127 = clone_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_407: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
    bmm_66: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_407, as_strided_scatter_15);  permute_407 = None
    permute_408: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    bmm_67: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_15, permute_408);  as_strided_scatter_15 = permute_408 = None
    view_567: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_66, [4, 8, 64, 1024]);  bmm_66 = None
    view_568: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_67, [4, 8, 1024, 64]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_409: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_567, [0, 1, 3, 2]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_143: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_10, permute_409);  tangents_10 = permute_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_410: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_141, [0, 2, 1, 3]);  add_141 = None
    clone_173: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
    view_569: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_173, [4, 1024, 512]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_570: "f32[4096, 512]" = torch.ops.aten.view.default(view_569, [4096, 512]);  view_569 = None
    permute_411: "f32[512, 4096]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_173: "f32[512, 512]" = torch.ops.aten.mm.default(permute_411, view_240);  permute_411 = view_240 = None
    permute_412: "f32[512, 512]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    permute_413: "f32[512, 512]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_174: "f32[4096, 512]" = torch.ops.aten.mm.default(view_570, permute_413);  view_570 = permute_413 = None
    view_571: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_174, [4, 1024, 512]);  mm_174 = None
    permute_414: "f32[512, 512]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_415: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_143, [0, 2, 1, 3]);  add_143 = None
    clone_174: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_415, memory_format = torch.contiguous_format);  permute_415 = None
    view_572: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_174, [4, 1024, 512]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_573: "f32[4096, 512]" = torch.ops.aten.view.default(view_572, [4096, 512]);  view_572 = None
    permute_416: "f32[512, 4096]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_175: "f32[512, 512]" = torch.ops.aten.mm.default(permute_416, view_237);  permute_416 = view_237 = None
    permute_417: "f32[512, 512]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    permute_418: "f32[512, 512]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_176: "f32[4096, 512]" = torch.ops.aten.mm.default(view_573, permute_418);  view_573 = permute_418 = None
    view_574: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_176, [4, 1024, 512]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_144: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_571, view_574);  view_571 = view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_419: "f32[512, 512]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_420: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
    clone_175: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_575: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_175, [4, 1024, 512]);  clone_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_576: "f32[4096, 512]" = torch.ops.aten.view.default(view_575, [4096, 512]);  view_575 = None
    permute_421: "f32[512, 4096]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_177: "f32[512, 512]" = torch.ops.aten.mm.default(permute_421, view_234);  permute_421 = view_234 = None
    permute_422: "f32[512, 512]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    permute_423: "f32[512, 512]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_178: "f32[4096, 512]" = torch.ops.aten.mm.default(view_576, permute_423);  view_576 = permute_423 = None
    view_577: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_178, [4, 1024, 512]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_145: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_144, view_577);  add_144 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_424: "f32[512, 512]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_185: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_145, primals_20);  primals_20 = None
    mul_186: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_145, mul_45);  add_145 = mul_45 = None
    sum_51: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 1], True);  mul_186 = None
    view_578: "f32[512]" = torch.ops.aten.view.default(sum_51, [512]);  sum_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_187: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_185, add_54)
    mul_188: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_185, rsqrt_19);  mul_185 = rsqrt_19 = None
    sum_52: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_146: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_140, mul_188);  add_140 = mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_86: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    pow_57: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_86, 3);  alias_86 = None
    mul_189: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_52, -0.5);  sum_52 = None
    mul_190: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_189, pow_57);  mul_189 = pow_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_84: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_190, [4, 1024, 512]);  mul_190 = None
    div_34: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_84, 512);  expand_84 = None
    pow_58: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 1.0);  add_54 = None
    mul_191: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_58, 2.0);  pow_58 = None
    mul_192: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_34, mul_191);  div_34 = mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_147: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_146, mul_192);  add_146 = mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_579: "f32[4096, 512]" = torch.ops.aten.view.default(add_147, [4096, 512])
    permute_425: "f32[512, 4096]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_179: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_425, view_232);  permute_425 = view_232 = None
    permute_426: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    permute_427: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_180: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_579, permute_427);  view_579 = permute_427 = None
    view_580: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_180, [4, 1024, 2048]);  mm_180 = None
    permute_428: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_426, [1, 0]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_87: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le_5: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_87, 0);  alias_87 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_5, scalar_tensor_4, view_580);  le_5 = scalar_tensor_4 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_581: "f32[4096, 2048]" = torch.ops.aten.view.default(where_6, [4096, 2048]);  where_6 = None
    permute_429: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_581, [1, 0])
    mm_181: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_429, view_230);  permute_429 = view_230 = None
    permute_430: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    permute_431: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    mm_182: "f32[4096, 512]" = torch.ops.aten.mm.default(view_581, permute_431);  view_581 = permute_431 = None
    view_582: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_182, [4, 1024, 512]);  mm_182 = None
    permute_432: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_193: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_582, primals_19);  primals_19 = None
    mul_194: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_582, mul_43);  view_582 = mul_43 = None
    sum_53: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_194, [0, 1], True);  mul_194 = None
    view_583: "f32[512]" = torch.ops.aten.view.default(sum_53, [512]);  sum_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_195: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_193, add_52)
    mul_196: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_193, rsqrt_18);  mul_193 = rsqrt_18 = None
    sum_54: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True);  mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_148: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_147, mul_196);  add_147 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_88: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    pow_59: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_88, 3);  alias_88 = None
    mul_197: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_54, -0.5);  sum_54 = None
    mul_198: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_197, pow_59);  mul_197 = pow_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_85: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_198, [4, 1024, 512]);  mul_198 = None
    div_35: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_85, 512);  expand_85 = None
    pow_60: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 1.0);  add_52 = None
    mul_199: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
    mul_200: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_35, mul_199);  div_35 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_149: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_148, mul_200);  add_148 = mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_584: "f32[4096, 512]" = torch.ops.aten.view.default(add_149, [4096, 512])
    permute_433: "f32[512, 4096]" = torch.ops.aten.permute.default(view_584, [1, 0])
    mm_183: "f32[512, 512]" = torch.ops.aten.mm.default(permute_433, view_228);  permute_433 = view_228 = None
    permute_434: "f32[512, 512]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    permute_435: "f32[512, 512]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_184: "f32[4096, 512]" = torch.ops.aten.mm.default(view_584, permute_435);  view_584 = permute_435 = None
    view_585: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_184, [4, 1024, 512]);  mm_184 = None
    permute_436: "f32[512, 512]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_586: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_585, [4, 1024, 8, 64]);  view_585 = None
    permute_437: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_586, [0, 2, 1, 3]);  view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_176: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_437, memory_format = torch.contiguous_format);  permute_437 = None
    view_587: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_176, [32, 1024, 64]);  clone_176 = None
    permute_438: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
    bmm_68: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_438, view_587);  permute_438 = None
    permute_439: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
    bmm_69: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_587, permute_439);  view_587 = permute_439 = None
    view_588: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_68, [4, 8, 1024, 64]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_150: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_9, view_588);  tangents_9 = view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_589: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_69, [4, 8, 1024, 1024]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_89: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    mul_201: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_589, alias_89);  view_589 = None
    sum_55: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [-1], True)
    mul_202: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_89, sum_55);  alias_89 = sum_55 = None
    sub_31: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_15: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_56: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_15, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_24: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_56, sub_31);  as_strided_56 = sub_31 = None
    as_strided_scatter_16: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_15, copy_24, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_15 = copy_24 = None
    as_strided_59: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_16, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_16 = None
    new_empty_strided_8: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_59, [32, 1024, 1024], [1048576, 1024, 1])
    copy_25: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_8, as_strided_59);  new_empty_strided_8 = as_strided_59 = None
    as_strided_61: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_25, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_177: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_61, memory_format = torch.contiguous_format)
    copy_26: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_61, clone_177);  as_strided_61 = clone_177 = None
    as_strided_scatter_17: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_25, copy_26, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_25 = copy_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_440: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
    bmm_70: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_440, as_strided_scatter_17);  permute_440 = None
    permute_441: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
    bmm_71: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_17, permute_441);  as_strided_scatter_17 = permute_441 = None
    view_590: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_70, [4, 8, 64, 1024]);  bmm_70 = None
    view_591: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_71, [4, 8, 1024, 64]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_442: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_590, [0, 1, 3, 2]);  view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_151: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_8, permute_442);  tangents_8 = permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_443: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_150, [0, 2, 1, 3]);  add_150 = None
    clone_178: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_443, memory_format = torch.contiguous_format);  permute_443 = None
    view_592: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_178, [4, 1024, 512]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_593: "f32[4096, 512]" = torch.ops.aten.view.default(view_592, [4096, 512]);  view_592 = None
    permute_444: "f32[512, 4096]" = torch.ops.aten.permute.default(view_593, [1, 0])
    mm_185: "f32[512, 512]" = torch.ops.aten.mm.default(permute_444, view_216);  permute_444 = view_216 = None
    permute_445: "f32[512, 512]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    permute_446: "f32[512, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_186: "f32[4096, 512]" = torch.ops.aten.mm.default(view_593, permute_446);  view_593 = permute_446 = None
    view_594: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_186, [4, 1024, 512]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_152: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_138, view_594);  add_138 = view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_447: "f32[512, 512]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_448: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_151, [0, 2, 1, 3]);  add_151 = None
    clone_179: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
    view_595: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_179, [4, 1024, 512]);  clone_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_596: "f32[4096, 512]" = torch.ops.aten.view.default(view_595, [4096, 512]);  view_595 = None
    permute_449: "f32[512, 4096]" = torch.ops.aten.permute.default(view_596, [1, 0])
    mm_187: "f32[512, 512]" = torch.ops.aten.mm.default(permute_449, view_213);  permute_449 = view_213 = None
    permute_450: "f32[512, 512]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    permute_451: "f32[512, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_188: "f32[4096, 512]" = torch.ops.aten.mm.default(view_596, permute_451);  view_596 = permute_451 = None
    view_597: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_188, [4, 1024, 512]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_153: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_152, view_597);  add_152 = view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_452: "f32[512, 512]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_453: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_591, [0, 2, 1, 3]);  view_591 = None
    clone_180: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_598: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_180, [4, 1024, 512]);  clone_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_599: "f32[4096, 512]" = torch.ops.aten.view.default(view_598, [4096, 512]);  view_598 = None
    permute_454: "f32[512, 4096]" = torch.ops.aten.permute.default(view_599, [1, 0])
    mm_189: "f32[512, 512]" = torch.ops.aten.mm.default(permute_454, view_210);  permute_454 = view_210 = None
    permute_455: "f32[512, 512]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    permute_456: "f32[512, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_190: "f32[4096, 512]" = torch.ops.aten.mm.default(view_599, permute_456);  view_599 = permute_456 = None
    view_600: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_190, [4, 1024, 512]);  mm_190 = None
    permute_457: "f32[512, 512]" = torch.ops.aten.permute.default(permute_455, [1, 0]);  permute_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_203: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_600, primals_18);  primals_18 = None
    mul_204: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_600, mul_41);  view_600 = mul_41 = None
    sum_56: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 1], True);  mul_204 = None
    view_601: "f32[512]" = torch.ops.aten.view.default(sum_56, [512]);  sum_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_205: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_203, add_49)
    mul_206: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_203, rsqrt_17);  mul_203 = rsqrt_17 = None
    sum_57: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [2], True);  mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_154: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_149, mul_206);  add_149 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_90: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    pow_61: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_90, 3);  alias_90 = None
    mul_207: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_57, -0.5);  sum_57 = None
    mul_208: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_207, pow_61);  mul_207 = pow_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_86: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_208, [4, 1024, 512]);  mul_208 = None
    div_36: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_86, 512);  expand_86 = None
    pow_62: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 1.0);  add_49 = None
    mul_209: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_62, 2.0);  pow_62 = None
    mul_210: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_36, mul_209);  div_36 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_155: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_154, mul_210);  add_154 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_602: "f32[4096, 512]" = torch.ops.aten.view.default(add_155, [4096, 512])
    permute_458: "f32[512, 4096]" = torch.ops.aten.permute.default(view_602, [1, 0])
    mm_191: "f32[512, 512]" = torch.ops.aten.mm.default(permute_458, view_208);  permute_458 = view_208 = None
    permute_459: "f32[512, 512]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    permute_460: "f32[512, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_192: "f32[4096, 512]" = torch.ops.aten.mm.default(view_602, permute_460);  view_602 = permute_460 = None
    view_603: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_192, [4, 1024, 512]);  mm_192 = None
    permute_461: "f32[512, 512]" = torch.ops.aten.permute.default(permute_459, [1, 0]);  permute_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_604: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_603, [4, 1024, 8, 64]);  view_603 = None
    permute_462: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_181: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_462, memory_format = torch.contiguous_format);  permute_462 = None
    view_605: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_181, [32, 1024, 64]);  clone_181 = None
    permute_463: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    bmm_72: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_463, view_605);  permute_463 = None
    permute_464: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    bmm_73: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_605, permute_464);  view_605 = permute_464 = None
    view_606: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_72, [4, 8, 1024, 64]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_156: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_7, view_606);  tangents_7 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_607: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_73, [4, 8, 1024, 1024]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_91: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    mul_211: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_607, alias_91);  view_607 = None
    sum_58: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [-1], True)
    mul_212: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_91, sum_58);  alias_91 = sum_58 = None
    sub_32: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_16: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_63: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_16, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_27: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_63, sub_32);  as_strided_63 = sub_32 = None
    as_strided_scatter_18: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_16, copy_27, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_16 = copy_27 = None
    as_strided_66: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_18, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_18 = None
    new_empty_strided_9: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_66, [32, 1024, 1024], [1048576, 1024, 1])
    copy_28: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_9, as_strided_66);  new_empty_strided_9 = as_strided_66 = None
    as_strided_68: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_28, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_182: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_68, memory_format = torch.contiguous_format)
    copy_29: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_68, clone_182);  as_strided_68 = None
    as_strided_scatter_19: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_28, copy_29, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_28 = copy_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_157: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_142, clone_182);  add_142 = clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_465: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
    bmm_74: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_465, as_strided_scatter_19);  permute_465 = None
    permute_466: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
    bmm_75: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_19, permute_466);  as_strided_scatter_19 = permute_466 = None
    view_608: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_74, [4, 8, 64, 1024]);  bmm_74 = None
    view_609: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_75, [4, 8, 1024, 64]);  bmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_467: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_608, [0, 1, 3, 2]);  view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_158: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_6, permute_467);  tangents_6 = permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_468: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_156, [0, 2, 1, 3]);  add_156 = None
    clone_183: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    view_610: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_183, [4, 1024, 512]);  clone_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_611: "f32[4096, 512]" = torch.ops.aten.view.default(view_610, [4096, 512]);  view_610 = None
    permute_469: "f32[512, 4096]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_193: "f32[512, 512]" = torch.ops.aten.mm.default(permute_469, view_196);  permute_469 = view_196 = None
    permute_470: "f32[512, 512]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    permute_471: "f32[512, 512]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_194: "f32[4096, 512]" = torch.ops.aten.mm.default(view_611, permute_471);  view_611 = permute_471 = None
    view_612: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_194, [4, 1024, 512]);  mm_194 = None
    permute_472: "f32[512, 512]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_473: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_158, [0, 2, 1, 3]);  add_158 = None
    clone_184: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
    view_613: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_184, [4, 1024, 512]);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_614: "f32[4096, 512]" = torch.ops.aten.view.default(view_613, [4096, 512]);  view_613 = None
    permute_474: "f32[512, 4096]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_195: "f32[512, 512]" = torch.ops.aten.mm.default(permute_474, view_193);  permute_474 = view_193 = None
    permute_475: "f32[512, 512]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    permute_476: "f32[512, 512]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_196: "f32[4096, 512]" = torch.ops.aten.mm.default(view_614, permute_476);  view_614 = permute_476 = None
    view_615: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_196, [4, 1024, 512]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_159: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_612, view_615);  view_612 = view_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_477: "f32[512, 512]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_478: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_609, [0, 2, 1, 3]);  view_609 = None
    clone_185: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
    view_616: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_185, [4, 1024, 512]);  clone_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_617: "f32[4096, 512]" = torch.ops.aten.view.default(view_616, [4096, 512]);  view_616 = None
    permute_479: "f32[512, 4096]" = torch.ops.aten.permute.default(view_617, [1, 0])
    mm_197: "f32[512, 512]" = torch.ops.aten.mm.default(permute_479, view_190);  permute_479 = view_190 = None
    permute_480: "f32[512, 512]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_198: "f32[4096, 512]" = torch.ops.aten.mm.default(view_617, permute_481);  view_617 = permute_481 = None
    view_618: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_198, [4, 1024, 512]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_160: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_159, view_618);  add_159 = view_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_482: "f32[512, 512]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_213: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_160, primals_17);  primals_17 = None
    mul_214: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_160, mul_39);  add_160 = mul_39 = None
    sum_59: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1], True);  mul_214 = None
    view_619: "f32[512]" = torch.ops.aten.view.default(sum_59, [512]);  sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_215: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_213, add_46)
    mul_216: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_213, rsqrt_16);  mul_213 = rsqrt_16 = None
    sum_60: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_161: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_155, mul_216);  add_155 = mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_92: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    pow_63: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_92, 3);  alias_92 = None
    mul_217: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_60, -0.5);  sum_60 = None
    mul_218: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_217, pow_63);  mul_217 = pow_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_87: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_218, [4, 1024, 512]);  mul_218 = None
    div_37: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_87, 512);  expand_87 = None
    pow_64: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 1.0);  add_46 = None
    mul_219: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
    mul_220: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_37, mul_219);  div_37 = mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_162: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_161, mul_220);  add_161 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_620: "f32[4096, 512]" = torch.ops.aten.view.default(add_162, [4096, 512])
    permute_483: "f32[512, 4096]" = torch.ops.aten.permute.default(view_620, [1, 0])
    mm_199: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_483, view_188);  permute_483 = view_188 = None
    permute_484: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    permute_485: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_200: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_620, permute_485);  view_620 = permute_485 = None
    view_621: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_200, [4, 1024, 2048]);  mm_200 = None
    permute_486: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_484, [1, 0]);  permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_93: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_6: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_93, 0);  alias_93 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_6, scalar_tensor_5, view_621);  le_6 = scalar_tensor_5 = view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_622: "f32[4096, 2048]" = torch.ops.aten.view.default(where_7, [4096, 2048]);  where_7 = None
    permute_487: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_622, [1, 0])
    mm_201: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_487, view_186);  permute_487 = view_186 = None
    permute_488: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    permute_489: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_202: "f32[4096, 512]" = torch.ops.aten.mm.default(view_622, permute_489);  view_622 = permute_489 = None
    view_623: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_202, [4, 1024, 512]);  mm_202 = None
    permute_490: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_488, [1, 0]);  permute_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_221: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_623, primals_16);  primals_16 = None
    mul_222: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_623, mul_37);  view_623 = mul_37 = None
    sum_61: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 1], True);  mul_222 = None
    view_624: "f32[512]" = torch.ops.aten.view.default(sum_61, [512]);  sum_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_223: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_221, add_44)
    mul_224: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_221, rsqrt_15);  mul_221 = rsqrt_15 = None
    sum_62: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True);  mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_163: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_162, mul_224);  add_162 = mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_94: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    pow_65: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_94, 3);  alias_94 = None
    mul_225: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_62, -0.5);  sum_62 = None
    mul_226: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_225, pow_65);  mul_225 = pow_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_88: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_226, [4, 1024, 512]);  mul_226 = None
    div_38: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_88, 512);  expand_88 = None
    pow_66: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 1.0);  add_44 = None
    mul_227: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
    mul_228: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_38, mul_227);  div_38 = mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_164: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_163, mul_228);  add_163 = mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_625: "f32[4096, 512]" = torch.ops.aten.view.default(add_164, [4096, 512])
    permute_491: "f32[512, 4096]" = torch.ops.aten.permute.default(view_625, [1, 0])
    mm_203: "f32[512, 512]" = torch.ops.aten.mm.default(permute_491, view_184);  permute_491 = view_184 = None
    permute_492: "f32[512, 512]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    permute_493: "f32[512, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_204: "f32[4096, 512]" = torch.ops.aten.mm.default(view_625, permute_493);  view_625 = permute_493 = None
    view_626: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_204, [4, 1024, 512]);  mm_204 = None
    permute_494: "f32[512, 512]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_627: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_626, [4, 1024, 8, 64]);  view_626 = None
    permute_495: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_186: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_495, memory_format = torch.contiguous_format);  permute_495 = None
    view_628: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_186, [32, 1024, 64]);  clone_186 = None
    permute_496: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_180, [0, 2, 1]);  view_180 = None
    bmm_76: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_496, view_628);  permute_496 = None
    permute_497: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
    bmm_77: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_628, permute_497);  view_628 = permute_497 = None
    view_629: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_76, [4, 8, 1024, 64]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_165: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_5, view_629);  tangents_5 = view_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_630: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_77, [4, 8, 1024, 1024]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_95: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    mul_229: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_630, alias_95);  view_630 = None
    sum_63: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [-1], True)
    mul_230: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_95, sum_63);  alias_95 = sum_63 = None
    sub_33: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_17: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_70: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_17, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_30: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_70, sub_33);  as_strided_70 = sub_33 = None
    as_strided_scatter_20: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_17, copy_30, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_17 = copy_30 = None
    as_strided_73: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_20, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_20 = None
    new_empty_strided_10: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_73, [32, 1024, 1024], [1048576, 1024, 1])
    copy_31: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_10, as_strided_73);  new_empty_strided_10 = as_strided_73 = None
    as_strided_75: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_31, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_187: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_75, memory_format = torch.contiguous_format)
    copy_32: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_75, clone_187);  as_strided_75 = clone_187 = None
    as_strided_scatter_21: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_31, copy_32, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_31 = copy_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_498: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
    bmm_78: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_498, as_strided_scatter_21);  permute_498 = None
    permute_499: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
    bmm_79: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_21, permute_499);  as_strided_scatter_21 = permute_499 = None
    view_631: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_78, [4, 8, 64, 1024]);  bmm_78 = None
    view_632: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_79, [4, 8, 1024, 64]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_500: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_631, [0, 1, 3, 2]);  view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_166: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_4, permute_500);  tangents_4 = permute_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_501: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_165, [0, 2, 1, 3]);  add_165 = None
    clone_188: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
    view_633: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_188, [4, 1024, 512]);  clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_634: "f32[4096, 512]" = torch.ops.aten.view.default(view_633, [4096, 512]);  view_633 = None
    permute_502: "f32[512, 4096]" = torch.ops.aten.permute.default(view_634, [1, 0])
    mm_205: "f32[512, 512]" = torch.ops.aten.mm.default(permute_502, view_172);  permute_502 = view_172 = None
    permute_503: "f32[512, 512]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    permute_504: "f32[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_206: "f32[4096, 512]" = torch.ops.aten.mm.default(view_634, permute_504);  view_634 = permute_504 = None
    view_635: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_206, [4, 1024, 512]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_167: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_153, view_635);  add_153 = view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_505: "f32[512, 512]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_506: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_166, [0, 2, 1, 3]);  add_166 = None
    clone_189: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_506, memory_format = torch.contiguous_format);  permute_506 = None
    view_636: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_189, [4, 1024, 512]);  clone_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    view_637: "f32[4096, 512]" = torch.ops.aten.view.default(view_636, [4096, 512]);  view_636 = None
    permute_507: "f32[512, 4096]" = torch.ops.aten.permute.default(view_637, [1, 0])
    mm_207: "f32[512, 512]" = torch.ops.aten.mm.default(permute_507, view_169);  permute_507 = view_169 = None
    permute_508: "f32[512, 512]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    permute_509: "f32[512, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_208: "f32[4096, 512]" = torch.ops.aten.mm.default(view_637, permute_509);  view_637 = permute_509 = None
    view_638: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_208, [4, 1024, 512]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    add_168: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_167, view_638);  add_167 = view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_510: "f32[512, 512]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_511: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_632, [0, 2, 1, 3]);  view_632 = None
    clone_190: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_511, memory_format = torch.contiguous_format);  permute_511 = None
    view_639: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_190, [4, 1024, 512]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_640: "f32[4096, 512]" = torch.ops.aten.view.default(view_639, [4096, 512]);  view_639 = None
    permute_512: "f32[512, 4096]" = torch.ops.aten.permute.default(view_640, [1, 0])
    mm_209: "f32[512, 512]" = torch.ops.aten.mm.default(permute_512, view_166);  permute_512 = view_166 = None
    permute_513: "f32[512, 512]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    permute_514: "f32[512, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_210: "f32[4096, 512]" = torch.ops.aten.mm.default(view_640, permute_514);  view_640 = permute_514 = None
    view_641: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_210, [4, 1024, 512]);  mm_210 = None
    permute_515: "f32[512, 512]" = torch.ops.aten.permute.default(permute_513, [1, 0]);  permute_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_231: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_641, primals_15);  primals_15 = None
    mul_232: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_641, mul_35);  view_641 = mul_35 = None
    sum_64: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1], True);  mul_232 = None
    view_642: "f32[512]" = torch.ops.aten.view.default(sum_64, [512]);  sum_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_233: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_231, add_40)
    mul_234: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_231, rsqrt_14);  mul_231 = rsqrt_14 = None
    sum_65: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [2], True);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_169: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_164, mul_234);  add_164 = mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_96: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    pow_67: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_96, 3);  alias_96 = None
    mul_235: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_65, -0.5);  sum_65 = None
    mul_236: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_235, pow_67);  mul_235 = pow_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_89: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_236, [4, 1024, 512]);  mul_236 = None
    div_39: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_89, 512);  expand_89 = None
    pow_68: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_40, 1.0);  add_40 = None
    mul_237: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
    mul_238: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_39, mul_237);  div_39 = mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_170: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_169, mul_238);  add_169 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_643: "f32[4096, 512]" = torch.ops.aten.view.default(add_170, [4096, 512])
    permute_516: "f32[512, 4096]" = torch.ops.aten.permute.default(view_643, [1, 0])
    mm_211: "f32[512, 512]" = torch.ops.aten.mm.default(permute_516, view_164);  permute_516 = view_164 = None
    permute_517: "f32[512, 512]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    permute_518: "f32[512, 512]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_212: "f32[4096, 512]" = torch.ops.aten.mm.default(view_643, permute_518);  view_643 = permute_518 = None
    view_644: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_212, [4, 1024, 512]);  mm_212 = None
    permute_519: "f32[512, 512]" = torch.ops.aten.permute.default(permute_517, [1, 0]);  permute_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_645: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_644, [4, 1024, 8, 64]);  view_644 = None
    permute_520: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_191: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
    view_646: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_191, [32, 1024, 64]);  clone_191 = None
    permute_521: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
    bmm_80: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_521, view_646);  permute_521 = None
    permute_522: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    bmm_81: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_646, permute_522);  view_646 = permute_522 = None
    view_647: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_80, [4, 8, 1024, 64]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_171: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_647);  tangents_3 = view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_648: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_81, [4, 8, 1024, 1024]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_97: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    mul_239: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_648, alias_97);  view_648 = None
    sum_66: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [-1], True)
    mul_240: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_97, sum_66);  alias_97 = sum_66 = None
    sub_34: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_18: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_77: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_18, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_33: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_77, sub_34);  as_strided_77 = sub_34 = None
    as_strided_scatter_22: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_18, copy_33, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_18 = copy_33 = None
    as_strided_80: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_22, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_22 = None
    new_empty_strided_11: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_80, [32, 1024, 1024], [1048576, 1024, 1])
    copy_34: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_11, as_strided_80);  new_empty_strided_11 = as_strided_80 = None
    as_strided_82: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_34, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_192: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_82, memory_format = torch.contiguous_format)
    copy_35: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_82, clone_192);  as_strided_82 = None
    as_strided_scatter_23: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_34, copy_35, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_34 = copy_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_172: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_157, clone_192);  add_157 = clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    sum_67: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_172, [0], True);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    squeeze: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_67, 0);  sum_67 = None
    permute_523: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    eq: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_37, -1)
    unsqueeze_17: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_17, scalar_tensor_6, permute_523);  unsqueeze_17 = scalar_tensor_6 = permute_523 = None
    clone_193: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_8, memory_format = torch.contiguous_format);  where_8 = None
    full_19: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[32, 8]" = torch.ops.aten._unsafe_index_put.default(full_19, [add_37], clone_193, True);  full_19 = add_37 = clone_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_524: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    bmm_82: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_524, as_strided_scatter_23);  permute_524 = None
    permute_525: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None
    bmm_83: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_23, permute_525);  as_strided_scatter_23 = permute_525 = None
    view_649: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_82, [4, 8, 64, 1024]);  bmm_82 = None
    view_650: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_83, [4, 8, 1024, 64]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_526: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_649, [0, 1, 3, 2]);  view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    add_173: "f32[4, 8, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_2, permute_526);  tangents_2 = permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_527: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_171, [0, 2, 1, 3]);  add_171 = None
    clone_194: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_527, memory_format = torch.contiguous_format);  permute_527 = None
    view_651: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_194, [4, 1024, 512]);  clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_652: "f32[4096, 512]" = torch.ops.aten.view.default(view_651, [4096, 512]);  view_651 = None
    permute_528: "f32[512, 4096]" = torch.ops.aten.permute.default(view_652, [1, 0])
    mm_213: "f32[512, 512]" = torch.ops.aten.mm.default(permute_528, view_152);  permute_528 = view_152 = None
    permute_529: "f32[512, 512]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    permute_530: "f32[512, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_214: "f32[4096, 512]" = torch.ops.aten.mm.default(view_652, permute_530);  view_652 = permute_530 = None
    view_653: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_214, [4, 1024, 512]);  mm_214 = None
    permute_531: "f32[512, 512]" = torch.ops.aten.permute.default(permute_529, [1, 0]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_532: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(add_173, [0, 2, 1, 3]);  add_173 = None
    clone_195: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_532, memory_format = torch.contiguous_format);  permute_532 = None
    view_654: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_195, [4, 1024, 512]);  clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_655: "f32[4096, 512]" = torch.ops.aten.view.default(view_654, [4096, 512]);  view_654 = None
    permute_533: "f32[512, 4096]" = torch.ops.aten.permute.default(view_655, [1, 0])
    mm_215: "f32[512, 512]" = torch.ops.aten.mm.default(permute_533, view_149);  permute_533 = view_149 = None
    permute_534: "f32[512, 512]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    permute_535: "f32[512, 512]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_216: "f32[4096, 512]" = torch.ops.aten.mm.default(view_655, permute_535);  view_655 = permute_535 = None
    view_656: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_216, [4, 1024, 512]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_174: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_653, view_656);  view_653 = view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_536: "f32[512, 512]" = torch.ops.aten.permute.default(permute_534, [1, 0]);  permute_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_537: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_650, [0, 2, 1, 3]);  view_650 = None
    clone_196: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_537, memory_format = torch.contiguous_format);  permute_537 = None
    view_657: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_196, [4, 1024, 512]);  clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_658: "f32[4096, 512]" = torch.ops.aten.view.default(view_657, [4096, 512]);  view_657 = None
    permute_538: "f32[512, 4096]" = torch.ops.aten.permute.default(view_658, [1, 0])
    mm_217: "f32[512, 512]" = torch.ops.aten.mm.default(permute_538, view_146);  permute_538 = view_146 = None
    permute_539: "f32[512, 512]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    permute_540: "f32[512, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_218: "f32[4096, 512]" = torch.ops.aten.mm.default(view_658, permute_540);  view_658 = permute_540 = None
    view_659: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_218, [4, 1024, 512]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_175: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_174, view_659);  add_174 = view_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_541: "f32[512, 512]" = torch.ops.aten.permute.default(permute_539, [1, 0]);  permute_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_241: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_175, primals_14);  primals_14 = None
    mul_242: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_175, mul_32);  add_175 = mul_32 = None
    sum_68: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1], True);  mul_242 = None
    view_660: "f32[512]" = torch.ops.aten.view.default(sum_68, [512]);  sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_243: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_241, clone_50)
    mul_244: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_241, rsqrt_13);  mul_241 = rsqrt_13 = None
    sum_69: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_176: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_170, mul_244);  add_170 = mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_98: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    pow_69: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_98, 3);  alias_98 = None
    mul_245: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_69, -0.5);  sum_69 = None
    mul_246: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_245, pow_69);  mul_245 = pow_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_90: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_246, [4, 1024, 512]);  mul_246 = None
    div_40: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_90, 512);  expand_90 = None
    pow_70: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(clone_50, 1.0);  clone_50 = None
    mul_247: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
    mul_248: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_40, mul_247);  div_40 = mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_177: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_176, mul_248);  add_176 = mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    eq_1: "b8[4, 1024]" = torch.ops.aten.eq.Scalar(view_145, -1)
    unsqueeze_18: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[4, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_18, scalar_tensor_7, add_177);  unsqueeze_18 = scalar_tensor_7 = add_177 = None
    full_20: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[32128, 512]" = torch.ops.aten._unsafe_index_put.default(full_20, [view_145], where_9, True);  full_20 = view_145 = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_249: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_168, primals_13);  primals_13 = None
    mul_250: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_168, mul_27);  add_168 = mul_27 = None
    sum_70: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1], True);  mul_250 = None
    view_661: "f32[512]" = torch.ops.aten.view.default(sum_70, [512]);  sum_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_251: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_249, add_33)
    mul_252: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_249, rsqrt_12);  mul_249 = rsqrt_12 = None
    sum_71: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
    alias_99: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    pow_71: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_99, 3);  alias_99 = None
    mul_253: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_71, -0.5);  sum_71 = None
    mul_254: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_253, pow_71);  mul_253 = pow_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_91: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_254, [4, 1024, 512]);  mul_254 = None
    div_41: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_91, 512);  expand_91 = None
    pow_72: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_33, 1.0);  add_33 = None
    mul_255: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
    mul_256: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_41, mul_255);  div_41 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_178: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(mul_252, mul_256);  mul_252 = mul_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_662: "f32[4096, 512]" = torch.ops.aten.view.default(add_178, [4096, 512])
    permute_542: "f32[512, 4096]" = torch.ops.aten.permute.default(view_662, [1, 0])
    mm_219: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_542, view_143);  permute_542 = view_143 = None
    permute_543: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    permute_544: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_220: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_662, permute_544);  view_662 = permute_544 = None
    view_663: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_220, [4, 1024, 2048]);  mm_220 = None
    permute_545: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_100: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    le_7: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_7, scalar_tensor_8, view_663);  le_7 = scalar_tensor_8 = view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_664: "f32[4096, 2048]" = torch.ops.aten.view.default(where_10, [4096, 2048]);  where_10 = None
    permute_546: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_664, [1, 0])
    mm_221: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_546, view_141);  permute_546 = view_141 = None
    permute_547: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    permute_548: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_222: "f32[4096, 512]" = torch.ops.aten.mm.default(view_664, permute_548);  view_664 = permute_548 = None
    view_665: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_222, [4, 1024, 512]);  mm_222 = None
    permute_549: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_547, [1, 0]);  permute_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_257: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_665, primals_12);  primals_12 = None
    mul_258: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_665, mul_25);  view_665 = mul_25 = None
    sum_72: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1], True);  mul_258 = None
    view_666: "f32[512]" = torch.ops.aten.view.default(sum_72, [512]);  sum_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_259: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_257, add_31)
    mul_260: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_257, rsqrt_11);  mul_257 = rsqrt_11 = None
    sum_73: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_179: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_178, mul_260);  add_178 = mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_101: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    pow_73: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_101, 3);  alias_101 = None
    mul_261: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_73, -0.5);  sum_73 = None
    mul_262: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_261, pow_73);  mul_261 = pow_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_92: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_262, [4, 1024, 512]);  mul_262 = None
    div_42: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_92, 512);  expand_92 = None
    pow_74: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 1.0);  add_31 = None
    mul_263: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
    mul_264: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_42, mul_263);  div_42 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_180: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_179, mul_264);  add_179 = mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_667: "f32[4096, 512]" = torch.ops.aten.view.default(add_180, [4096, 512])
    permute_550: "f32[512, 4096]" = torch.ops.aten.permute.default(view_667, [1, 0])
    mm_223: "f32[512, 512]" = torch.ops.aten.mm.default(permute_550, view_139);  permute_550 = view_139 = None
    permute_551: "f32[512, 512]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    permute_552: "f32[512, 512]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_224: "f32[4096, 512]" = torch.ops.aten.mm.default(view_667, permute_552);  view_667 = permute_552 = None
    view_668: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_224, [4, 1024, 512]);  mm_224 = None
    permute_553: "f32[512, 512]" = torch.ops.aten.permute.default(permute_551, [1, 0]);  permute_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_669: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_668, [4, 1024, 8, 64]);  view_668 = None
    permute_554: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_669, [0, 2, 1, 3]);  view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_197: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_554, memory_format = torch.contiguous_format);  permute_554 = None
    view_670: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_197, [32, 1024, 64]);  clone_197 = None
    permute_555: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    bmm_84: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_555, view_670);  permute_555 = None
    permute_556: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    bmm_85: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_670, permute_556);  view_670 = permute_556 = None
    view_671: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_84, [4, 8, 1024, 64]);  bmm_84 = None
    view_672: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_85, [4, 8, 1024, 1024]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_102: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_265: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_672, alias_102);  view_672 = None
    sum_74: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [-1], True)
    mul_266: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_102, sum_74);  alias_102 = sum_74 = None
    sub_35: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_21: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_84: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_21, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_36: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_84, sub_35);  as_strided_84 = sub_35 = None
    as_strided_scatter_24: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_21, copy_36, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_21 = copy_36 = None
    as_strided_87: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_24, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_24 = None
    new_empty_strided_12: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_87, [32, 1024, 1024], [1048576, 1024, 1])
    copy_37: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_12, as_strided_87);  new_empty_strided_12 = as_strided_87 = None
    as_strided_89: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_37, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_198: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_89, memory_format = torch.contiguous_format)
    copy_38: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_89, clone_198);  as_strided_89 = None
    as_strided_scatter_25: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_37, copy_38, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_37 = copy_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_557: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
    bmm_86: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_557, as_strided_scatter_25);  permute_557 = None
    permute_558: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
    bmm_87: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_25, permute_558);  as_strided_scatter_25 = permute_558 = None
    view_673: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_86, [4, 8, 64, 1024]);  bmm_86 = None
    view_674: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_87, [4, 8, 1024, 64]);  bmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_559: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_673, [0, 1, 3, 2]);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_560: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None
    clone_199: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format);  permute_560 = None
    view_675: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_199, [4, 1024, 512]);  clone_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_676: "f32[4096, 512]" = torch.ops.aten.view.default(view_675, [4096, 512]);  view_675 = None
    permute_561: "f32[512, 4096]" = torch.ops.aten.permute.default(view_676, [1, 0])
    mm_225: "f32[512, 512]" = torch.ops.aten.mm.default(permute_561, view_127);  permute_561 = view_127 = None
    permute_562: "f32[512, 512]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    permute_563: "f32[512, 512]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_226: "f32[4096, 512]" = torch.ops.aten.mm.default(view_676, permute_563);  view_676 = permute_563 = None
    view_677: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_226, [4, 1024, 512]);  mm_226 = None
    permute_564: "f32[512, 512]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_565: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_559, [0, 2, 1, 3]);  permute_559 = None
    view_678: "f32[4, 1024, 512]" = torch.ops.aten.view.default(permute_565, [4, 1024, 512]);  permute_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    clone_200: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_678, memory_format = torch.contiguous_format);  view_678 = None
    view_679: "f32[4096, 512]" = torch.ops.aten.view.default(clone_200, [4096, 512]);  clone_200 = None
    permute_566: "f32[512, 4096]" = torch.ops.aten.permute.default(view_679, [1, 0])
    mm_227: "f32[512, 512]" = torch.ops.aten.mm.default(permute_566, view_124);  permute_566 = view_124 = None
    permute_567: "f32[512, 512]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    permute_568: "f32[512, 512]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_228: "f32[4096, 512]" = torch.ops.aten.mm.default(view_679, permute_568);  view_679 = permute_568 = None
    view_680: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_228, [4, 1024, 512]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_181: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_677, view_680);  view_677 = view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_569: "f32[512, 512]" = torch.ops.aten.permute.default(permute_567, [1, 0]);  permute_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_570: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_674, [0, 2, 1, 3]);  view_674 = None
    clone_201: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_570, memory_format = torch.contiguous_format);  permute_570 = None
    view_681: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_201, [4, 1024, 512]);  clone_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_682: "f32[4096, 512]" = torch.ops.aten.view.default(view_681, [4096, 512]);  view_681 = None
    permute_571: "f32[512, 4096]" = torch.ops.aten.permute.default(view_682, [1, 0])
    mm_229: "f32[512, 512]" = torch.ops.aten.mm.default(permute_571, view_121);  permute_571 = view_121 = None
    permute_572: "f32[512, 512]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    permute_573: "f32[512, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_230: "f32[4096, 512]" = torch.ops.aten.mm.default(view_682, permute_573);  view_682 = permute_573 = None
    view_683: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_230, [4, 1024, 512]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_182: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_181, view_683);  add_181 = view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_574: "f32[512, 512]" = torch.ops.aten.permute.default(permute_572, [1, 0]);  permute_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_267: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_182, primals_11);  primals_11 = None
    mul_268: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_182, mul_23);  add_182 = mul_23 = None
    sum_75: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1], True);  mul_268 = None
    view_684: "f32[512]" = torch.ops.aten.view.default(sum_75, [512]);  sum_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_269: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_267, add_28)
    mul_270: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_267, rsqrt_10);  mul_267 = rsqrt_10 = None
    sum_76: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_183: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_180, mul_270);  add_180 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_103: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    pow_75: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_103, 3);  alias_103 = None
    mul_271: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_76, -0.5);  sum_76 = None
    mul_272: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_271, pow_75);  mul_271 = pow_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_93: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_272, [4, 1024, 512]);  mul_272 = None
    div_43: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_93, 512);  expand_93 = None
    pow_76: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_28, 1.0);  add_28 = None
    mul_273: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_76, 2.0);  pow_76 = None
    mul_274: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_43, mul_273);  div_43 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_184: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_183, mul_274);  add_183 = mul_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_685: "f32[4096, 512]" = torch.ops.aten.view.default(add_184, [4096, 512])
    permute_575: "f32[512, 4096]" = torch.ops.aten.permute.default(view_685, [1, 0])
    mm_231: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_575, view_119);  permute_575 = view_119 = None
    permute_576: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    permute_577: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_232: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_685, permute_577);  view_685 = permute_577 = None
    view_686: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_232, [4, 1024, 2048]);  mm_232 = None
    permute_578: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_576, [1, 0]);  permute_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_104: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    le_8: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_104, 0);  alias_104 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_8, scalar_tensor_9, view_686);  le_8 = scalar_tensor_9 = view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_687: "f32[4096, 2048]" = torch.ops.aten.view.default(where_11, [4096, 2048]);  where_11 = None
    permute_579: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_687, [1, 0])
    mm_233: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_579, view_117);  permute_579 = view_117 = None
    permute_580: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    permute_581: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_234: "f32[4096, 512]" = torch.ops.aten.mm.default(view_687, permute_581);  view_687 = permute_581 = None
    view_688: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_234, [4, 1024, 512]);  mm_234 = None
    permute_582: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_580, [1, 0]);  permute_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_275: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_688, primals_10);  primals_10 = None
    mul_276: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_688, mul_21);  view_688 = mul_21 = None
    sum_77: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1], True);  mul_276 = None
    view_689: "f32[512]" = torch.ops.aten.view.default(sum_77, [512]);  sum_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_277: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_275, add_26)
    mul_278: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_275, rsqrt_9);  mul_275 = rsqrt_9 = None
    sum_78: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_185: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_184, mul_278);  add_184 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_105: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    pow_77: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_105, 3);  alias_105 = None
    mul_279: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_78, -0.5);  sum_78 = None
    mul_280: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_279, pow_77);  mul_279 = pow_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_94: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_280, [4, 1024, 512]);  mul_280 = None
    div_44: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_94, 512);  expand_94 = None
    pow_78: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 1.0);  add_26 = None
    mul_281: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_78, 2.0);  pow_78 = None
    mul_282: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_44, mul_281);  div_44 = mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_186: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_185, mul_282);  add_185 = mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_690: "f32[4096, 512]" = torch.ops.aten.view.default(add_186, [4096, 512])
    permute_583: "f32[512, 4096]" = torch.ops.aten.permute.default(view_690, [1, 0])
    mm_235: "f32[512, 512]" = torch.ops.aten.mm.default(permute_583, view_115);  permute_583 = view_115 = None
    permute_584: "f32[512, 512]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    permute_585: "f32[512, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_236: "f32[4096, 512]" = torch.ops.aten.mm.default(view_690, permute_585);  view_690 = permute_585 = None
    view_691: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_236, [4, 1024, 512]);  mm_236 = None
    permute_586: "f32[512, 512]" = torch.ops.aten.permute.default(permute_584, [1, 0]);  permute_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_692: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_691, [4, 1024, 8, 64]);  view_691 = None
    permute_587: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_202: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
    view_693: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_202, [32, 1024, 64]);  clone_202 = None
    permute_588: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    bmm_88: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_588, view_693);  permute_588 = None
    permute_589: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    bmm_89: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_693, permute_589);  view_693 = permute_589 = None
    view_694: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_88, [4, 8, 1024, 64]);  bmm_88 = None
    view_695: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_89, [4, 8, 1024, 1024]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_106: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_283: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_695, alias_106);  view_695 = None
    sum_79: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [-1], True)
    mul_284: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_106, sum_79);  alias_106 = sum_79 = None
    sub_36: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_22: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_91: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_22, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_39: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_91, sub_36);  as_strided_91 = sub_36 = None
    as_strided_scatter_26: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_22, copy_39, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_22 = copy_39 = None
    as_strided_94: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_26, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_26 = None
    new_empty_strided_13: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_94, [32, 1024, 1024], [1048576, 1024, 1])
    copy_40: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_13, as_strided_94);  new_empty_strided_13 = as_strided_94 = None
    as_strided_96: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_40, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_203: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_96, memory_format = torch.contiguous_format)
    copy_41: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_96, clone_203);  as_strided_96 = None
    as_strided_scatter_27: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_40, copy_41, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_40 = copy_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_187: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(clone_198, clone_203);  clone_198 = clone_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_590: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
    bmm_90: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_590, as_strided_scatter_27);  permute_590 = None
    permute_591: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    bmm_91: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_27, permute_591);  as_strided_scatter_27 = permute_591 = None
    view_696: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_90, [4, 8, 64, 1024]);  bmm_90 = None
    view_697: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_91, [4, 8, 1024, 64]);  bmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_592: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_696, [0, 1, 3, 2]);  view_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_593: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
    clone_204: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_593, memory_format = torch.contiguous_format);  permute_593 = None
    view_698: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_204, [4, 1024, 512]);  clone_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_699: "f32[4096, 512]" = torch.ops.aten.view.default(view_698, [4096, 512]);  view_698 = None
    permute_594: "f32[512, 4096]" = torch.ops.aten.permute.default(view_699, [1, 0])
    mm_237: "f32[512, 512]" = torch.ops.aten.mm.default(permute_594, view_103);  permute_594 = view_103 = None
    permute_595: "f32[512, 512]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    permute_596: "f32[512, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_238: "f32[4096, 512]" = torch.ops.aten.mm.default(view_699, permute_596);  view_699 = permute_596 = None
    view_700: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_238, [4, 1024, 512]);  mm_238 = None
    permute_597: "f32[512, 512]" = torch.ops.aten.permute.default(permute_595, [1, 0]);  permute_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_598: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_592, [0, 2, 1, 3]);  permute_592 = None
    view_701: "f32[4, 1024, 512]" = torch.ops.aten.view.default(permute_598, [4, 1024, 512]);  permute_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    clone_205: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_701, memory_format = torch.contiguous_format);  view_701 = None
    view_702: "f32[4096, 512]" = torch.ops.aten.view.default(clone_205, [4096, 512]);  clone_205 = None
    permute_599: "f32[512, 4096]" = torch.ops.aten.permute.default(view_702, [1, 0])
    mm_239: "f32[512, 512]" = torch.ops.aten.mm.default(permute_599, view_100);  permute_599 = view_100 = None
    permute_600: "f32[512, 512]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    permute_601: "f32[512, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_240: "f32[4096, 512]" = torch.ops.aten.mm.default(view_702, permute_601);  view_702 = permute_601 = None
    view_703: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_240, [4, 1024, 512]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_188: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_700, view_703);  view_700 = view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_602: "f32[512, 512]" = torch.ops.aten.permute.default(permute_600, [1, 0]);  permute_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_603: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_697, [0, 2, 1, 3]);  view_697 = None
    clone_206: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
    view_704: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_206, [4, 1024, 512]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_705: "f32[4096, 512]" = torch.ops.aten.view.default(view_704, [4096, 512]);  view_704 = None
    permute_604: "f32[512, 4096]" = torch.ops.aten.permute.default(view_705, [1, 0])
    mm_241: "f32[512, 512]" = torch.ops.aten.mm.default(permute_604, view_97);  permute_604 = view_97 = None
    permute_605: "f32[512, 512]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    permute_606: "f32[512, 512]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_242: "f32[4096, 512]" = torch.ops.aten.mm.default(view_705, permute_606);  view_705 = permute_606 = None
    view_706: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_242, [4, 1024, 512]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_189: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_188, view_706);  add_188 = view_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_607: "f32[512, 512]" = torch.ops.aten.permute.default(permute_605, [1, 0]);  permute_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_285: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_189, primals_9);  primals_9 = None
    mul_286: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_189, mul_19);  add_189 = mul_19 = None
    sum_80: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 1], True);  mul_286 = None
    view_707: "f32[512]" = torch.ops.aten.view.default(sum_80, [512]);  sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_287: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_285, add_23)
    mul_288: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_285, rsqrt_8);  mul_285 = rsqrt_8 = None
    sum_81: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_190: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_186, mul_288);  add_186 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_107: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    pow_79: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_107, 3);  alias_107 = None
    mul_289: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_81, -0.5);  sum_81 = None
    mul_290: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_289, pow_79);  mul_289 = pow_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_95: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_290, [4, 1024, 512]);  mul_290 = None
    div_45: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_95, 512);  expand_95 = None
    pow_80: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 1.0);  add_23 = None
    mul_291: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_80, 2.0);  pow_80 = None
    mul_292: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_45, mul_291);  div_45 = mul_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_191: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_190, mul_292);  add_190 = mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_708: "f32[4096, 512]" = torch.ops.aten.view.default(add_191, [4096, 512])
    permute_608: "f32[512, 4096]" = torch.ops.aten.permute.default(view_708, [1, 0])
    mm_243: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_608, view_95);  permute_608 = view_95 = None
    permute_609: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    permute_610: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_244: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_708, permute_610);  view_708 = permute_610 = None
    view_709: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_244, [4, 1024, 2048]);  mm_244 = None
    permute_611: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_609, [1, 0]);  permute_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_108: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    le_9: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_108, 0);  alias_108 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_9, scalar_tensor_10, view_709);  le_9 = scalar_tensor_10 = view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_710: "f32[4096, 2048]" = torch.ops.aten.view.default(where_12, [4096, 2048]);  where_12 = None
    permute_612: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_710, [1, 0])
    mm_245: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_612, view_93);  permute_612 = view_93 = None
    permute_613: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    permute_614: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_246: "f32[4096, 512]" = torch.ops.aten.mm.default(view_710, permute_614);  view_710 = permute_614 = None
    view_711: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_246, [4, 1024, 512]);  mm_246 = None
    permute_615: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_613, [1, 0]);  permute_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_293: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_711, primals_8);  primals_8 = None
    mul_294: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_711, mul_17);  view_711 = mul_17 = None
    sum_82: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 1], True);  mul_294 = None
    view_712: "f32[512]" = torch.ops.aten.view.default(sum_82, [512]);  sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_295: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_293, add_21)
    mul_296: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_293, rsqrt_7);  mul_293 = rsqrt_7 = None
    sum_83: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True);  mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_192: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_191, mul_296);  add_191 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_109: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    pow_81: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_109, 3);  alias_109 = None
    mul_297: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_83, -0.5);  sum_83 = None
    mul_298: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_297, pow_81);  mul_297 = pow_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_96: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_298, [4, 1024, 512]);  mul_298 = None
    div_46: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_96, 512);  expand_96 = None
    pow_82: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 1.0);  add_21 = None
    mul_299: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_82, 2.0);  pow_82 = None
    mul_300: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_46, mul_299);  div_46 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_193: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_192, mul_300);  add_192 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_713: "f32[4096, 512]" = torch.ops.aten.view.default(add_193, [4096, 512])
    permute_616: "f32[512, 4096]" = torch.ops.aten.permute.default(view_713, [1, 0])
    mm_247: "f32[512, 512]" = torch.ops.aten.mm.default(permute_616, view_91);  permute_616 = view_91 = None
    permute_617: "f32[512, 512]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    permute_618: "f32[512, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_248: "f32[4096, 512]" = torch.ops.aten.mm.default(view_713, permute_618);  view_713 = permute_618 = None
    view_714: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_248, [4, 1024, 512]);  mm_248 = None
    permute_619: "f32[512, 512]" = torch.ops.aten.permute.default(permute_617, [1, 0]);  permute_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_715: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_714, [4, 1024, 8, 64]);  view_714 = None
    permute_620: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_715, [0, 2, 1, 3]);  view_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_207: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_620, memory_format = torch.contiguous_format);  permute_620 = None
    view_716: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_207, [32, 1024, 64]);  clone_207 = None
    permute_621: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
    bmm_92: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_621, view_716);  permute_621 = None
    permute_622: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
    bmm_93: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_716, permute_622);  view_716 = permute_622 = None
    view_717: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_92, [4, 8, 1024, 64]);  bmm_92 = None
    view_718: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_93, [4, 8, 1024, 1024]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_110: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_301: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_718, alias_110);  view_718 = None
    sum_84: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
    mul_302: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_110, sum_84);  alias_110 = sum_84 = None
    sub_37: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_23: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_98: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_23, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_42: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_98, sub_37);  as_strided_98 = sub_37 = None
    as_strided_scatter_28: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_23, copy_42, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_23 = copy_42 = None
    as_strided_101: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_28, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_28 = None
    new_empty_strided_14: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_101, [32, 1024, 1024], [1048576, 1024, 1])
    copy_43: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_14, as_strided_101);  new_empty_strided_14 = as_strided_101 = None
    as_strided_103: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_43, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_208: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_103, memory_format = torch.contiguous_format)
    copy_44: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_103, clone_208);  as_strided_103 = None
    as_strided_scatter_29: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_43, copy_44, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_43 = copy_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_194: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_187, clone_208);  add_187 = clone_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_623: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    bmm_94: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_623, as_strided_scatter_29);  permute_623 = None
    permute_624: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    bmm_95: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_29, permute_624);  as_strided_scatter_29 = permute_624 = None
    view_719: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_94, [4, 8, 64, 1024]);  bmm_94 = None
    view_720: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_95, [4, 8, 1024, 64]);  bmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_625: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_719, [0, 1, 3, 2]);  view_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_626: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_717, [0, 2, 1, 3]);  view_717 = None
    clone_209: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_626, memory_format = torch.contiguous_format);  permute_626 = None
    view_721: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_209, [4, 1024, 512]);  clone_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_722: "f32[4096, 512]" = torch.ops.aten.view.default(view_721, [4096, 512]);  view_721 = None
    permute_627: "f32[512, 4096]" = torch.ops.aten.permute.default(view_722, [1, 0])
    mm_249: "f32[512, 512]" = torch.ops.aten.mm.default(permute_627, view_79);  permute_627 = view_79 = None
    permute_628: "f32[512, 512]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    permute_629: "f32[512, 512]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    mm_250: "f32[4096, 512]" = torch.ops.aten.mm.default(view_722, permute_629);  view_722 = permute_629 = None
    view_723: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_250, [4, 1024, 512]);  mm_250 = None
    permute_630: "f32[512, 512]" = torch.ops.aten.permute.default(permute_628, [1, 0]);  permute_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_631: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_625, [0, 2, 1, 3]);  permute_625 = None
    view_724: "f32[4, 1024, 512]" = torch.ops.aten.view.default(permute_631, [4, 1024, 512]);  permute_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    clone_210: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_724, memory_format = torch.contiguous_format);  view_724 = None
    view_725: "f32[4096, 512]" = torch.ops.aten.view.default(clone_210, [4096, 512]);  clone_210 = None
    permute_632: "f32[512, 4096]" = torch.ops.aten.permute.default(view_725, [1, 0])
    mm_251: "f32[512, 512]" = torch.ops.aten.mm.default(permute_632, view_76);  permute_632 = view_76 = None
    permute_633: "f32[512, 512]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    permute_634: "f32[512, 512]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_252: "f32[4096, 512]" = torch.ops.aten.mm.default(view_725, permute_634);  view_725 = permute_634 = None
    view_726: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_252, [4, 1024, 512]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_195: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_723, view_726);  view_723 = view_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_635: "f32[512, 512]" = torch.ops.aten.permute.default(permute_633, [1, 0]);  permute_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_636: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
    clone_211: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_636, memory_format = torch.contiguous_format);  permute_636 = None
    view_727: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_211, [4, 1024, 512]);  clone_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_728: "f32[4096, 512]" = torch.ops.aten.view.default(view_727, [4096, 512]);  view_727 = None
    permute_637: "f32[512, 4096]" = torch.ops.aten.permute.default(view_728, [1, 0])
    mm_253: "f32[512, 512]" = torch.ops.aten.mm.default(permute_637, view_73);  permute_637 = view_73 = None
    permute_638: "f32[512, 512]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    permute_639: "f32[512, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_254: "f32[4096, 512]" = torch.ops.aten.mm.default(view_728, permute_639);  view_728 = permute_639 = None
    view_729: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_254, [4, 1024, 512]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_196: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_195, view_729);  add_195 = view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_640: "f32[512, 512]" = torch.ops.aten.permute.default(permute_638, [1, 0]);  permute_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_303: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_196, primals_7);  primals_7 = None
    mul_304: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_196, mul_15);  add_196 = mul_15 = None
    sum_85: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1], True);  mul_304 = None
    view_730: "f32[512]" = torch.ops.aten.view.default(sum_85, [512]);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_305: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_303, add_18)
    mul_306: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_303, rsqrt_6);  mul_303 = rsqrt_6 = None
    sum_86: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_197: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_193, mul_306);  add_193 = mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_111: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    pow_83: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_111, 3);  alias_111 = None
    mul_307: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_86, -0.5);  sum_86 = None
    mul_308: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_307, pow_83);  mul_307 = pow_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_97: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_308, [4, 1024, 512]);  mul_308 = None
    div_47: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_97, 512);  expand_97 = None
    pow_84: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_18, 1.0);  add_18 = None
    mul_309: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
    mul_310: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_47, mul_309);  div_47 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_198: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_197, mul_310);  add_197 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_731: "f32[4096, 512]" = torch.ops.aten.view.default(add_198, [4096, 512])
    permute_641: "f32[512, 4096]" = torch.ops.aten.permute.default(view_731, [1, 0])
    mm_255: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_641, view_71);  permute_641 = view_71 = None
    permute_642: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    permute_643: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_256: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_731, permute_643);  view_731 = permute_643 = None
    view_732: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_256, [4, 1024, 2048]);  mm_256 = None
    permute_644: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_642, [1, 0]);  permute_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_112: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    le_10: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_112, 0);  alias_112 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_10, scalar_tensor_11, view_732);  le_10 = scalar_tensor_11 = view_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_733: "f32[4096, 2048]" = torch.ops.aten.view.default(where_13, [4096, 2048]);  where_13 = None
    permute_645: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_733, [1, 0])
    mm_257: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_645, view_69);  permute_645 = view_69 = None
    permute_646: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    permute_647: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_258: "f32[4096, 512]" = torch.ops.aten.mm.default(view_733, permute_647);  view_733 = permute_647 = None
    view_734: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_258, [4, 1024, 512]);  mm_258 = None
    permute_648: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_646, [1, 0]);  permute_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_311: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_734, primals_6);  primals_6 = None
    mul_312: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_734, mul_13);  view_734 = mul_13 = None
    sum_87: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 1], True);  mul_312 = None
    view_735: "f32[512]" = torch.ops.aten.view.default(sum_87, [512]);  sum_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_313: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_311, add_16)
    mul_314: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_311, rsqrt_5);  mul_311 = rsqrt_5 = None
    sum_88: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_199: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_198, mul_314);  add_198 = mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_113: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    pow_85: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_113, 3);  alias_113 = None
    mul_315: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_88, -0.5);  sum_88 = None
    mul_316: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_315, pow_85);  mul_315 = pow_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_98: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_316, [4, 1024, 512]);  mul_316 = None
    div_48: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_98, 512);  expand_98 = None
    pow_86: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 1.0);  add_16 = None
    mul_317: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
    mul_318: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_48, mul_317);  div_48 = mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_200: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_199, mul_318);  add_199 = mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_736: "f32[4096, 512]" = torch.ops.aten.view.default(add_200, [4096, 512])
    permute_649: "f32[512, 4096]" = torch.ops.aten.permute.default(view_736, [1, 0])
    mm_259: "f32[512, 512]" = torch.ops.aten.mm.default(permute_649, view_67);  permute_649 = view_67 = None
    permute_650: "f32[512, 512]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    permute_651: "f32[512, 512]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_260: "f32[4096, 512]" = torch.ops.aten.mm.default(view_736, permute_651);  view_736 = permute_651 = None
    view_737: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_260, [4, 1024, 512]);  mm_260 = None
    permute_652: "f32[512, 512]" = torch.ops.aten.permute.default(permute_650, [1, 0]);  permute_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_738: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_737, [4, 1024, 8, 64]);  view_737 = None
    permute_653: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_212: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_653, memory_format = torch.contiguous_format);  permute_653 = None
    view_739: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_212, [32, 1024, 64]);  clone_212 = None
    permute_654: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    bmm_96: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_654, view_739);  permute_654 = None
    permute_655: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    bmm_97: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_739, permute_655);  view_739 = permute_655 = None
    view_740: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_96, [4, 8, 1024, 64]);  bmm_96 = None
    view_741: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_97, [4, 8, 1024, 1024]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_114: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_319: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_741, alias_114);  view_741 = None
    sum_89: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [-1], True)
    mul_320: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_114, sum_89);  alias_114 = sum_89 = None
    sub_38: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_24: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_105: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_24, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_45: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_105, sub_38);  as_strided_105 = sub_38 = None
    as_strided_scatter_30: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_24, copy_45, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_24 = copy_45 = None
    as_strided_108: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_30, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_30 = None
    new_empty_strided_15: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_108, [32, 1024, 1024], [1048576, 1024, 1])
    copy_46: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_15, as_strided_108);  new_empty_strided_15 = as_strided_108 = None
    as_strided_110: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_46, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_213: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_110, memory_format = torch.contiguous_format)
    copy_47: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_110, clone_213);  as_strided_110 = None
    as_strided_scatter_31: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_46, copy_47, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_46 = copy_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_201: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_194, clone_213);  add_194 = clone_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_656: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_98: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_656, as_strided_scatter_31);  permute_656 = None
    permute_657: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_99: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_31, permute_657);  as_strided_scatter_31 = permute_657 = None
    view_742: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_98, [4, 8, 64, 1024]);  bmm_98 = None
    view_743: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_99, [4, 8, 1024, 64]);  bmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_658: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_742, [0, 1, 3, 2]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_659: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_740, [0, 2, 1, 3]);  view_740 = None
    clone_214: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_659, memory_format = torch.contiguous_format);  permute_659 = None
    view_744: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_214, [4, 1024, 512]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_745: "f32[4096, 512]" = torch.ops.aten.view.default(view_744, [4096, 512]);  view_744 = None
    permute_660: "f32[512, 4096]" = torch.ops.aten.permute.default(view_745, [1, 0])
    mm_261: "f32[512, 512]" = torch.ops.aten.mm.default(permute_660, view_55);  permute_660 = view_55 = None
    permute_661: "f32[512, 512]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    permute_662: "f32[512, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm_262: "f32[4096, 512]" = torch.ops.aten.mm.default(view_745, permute_662);  view_745 = permute_662 = None
    view_746: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_262, [4, 1024, 512]);  mm_262 = None
    permute_663: "f32[512, 512]" = torch.ops.aten.permute.default(permute_661, [1, 0]);  permute_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_664: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_658, [0, 2, 1, 3]);  permute_658 = None
    view_747: "f32[4, 1024, 512]" = torch.ops.aten.view.default(permute_664, [4, 1024, 512]);  permute_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    clone_215: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_747, memory_format = torch.contiguous_format);  view_747 = None
    view_748: "f32[4096, 512]" = torch.ops.aten.view.default(clone_215, [4096, 512]);  clone_215 = None
    permute_665: "f32[512, 4096]" = torch.ops.aten.permute.default(view_748, [1, 0])
    mm_263: "f32[512, 512]" = torch.ops.aten.mm.default(permute_665, view_52);  permute_665 = view_52 = None
    permute_666: "f32[512, 512]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    permute_667: "f32[512, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_264: "f32[4096, 512]" = torch.ops.aten.mm.default(view_748, permute_667);  view_748 = permute_667 = None
    view_749: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_264, [4, 1024, 512]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_202: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_746, view_749);  view_746 = view_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_668: "f32[512, 512]" = torch.ops.aten.permute.default(permute_666, [1, 0]);  permute_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_669: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_743, [0, 2, 1, 3]);  view_743 = None
    clone_216: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_669, memory_format = torch.contiguous_format);  permute_669 = None
    view_750: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_216, [4, 1024, 512]);  clone_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_751: "f32[4096, 512]" = torch.ops.aten.view.default(view_750, [4096, 512]);  view_750 = None
    permute_670: "f32[512, 4096]" = torch.ops.aten.permute.default(view_751, [1, 0])
    mm_265: "f32[512, 512]" = torch.ops.aten.mm.default(permute_670, view_49);  permute_670 = view_49 = None
    permute_671: "f32[512, 512]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    permute_672: "f32[512, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_266: "f32[4096, 512]" = torch.ops.aten.mm.default(view_751, permute_672);  view_751 = permute_672 = None
    view_752: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_266, [4, 1024, 512]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_203: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_202, view_752);  add_202 = view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_673: "f32[512, 512]" = torch.ops.aten.permute.default(permute_671, [1, 0]);  permute_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_321: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_203, primals_5);  primals_5 = None
    mul_322: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_203, mul_11);  add_203 = mul_11 = None
    sum_90: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 1], True);  mul_322 = None
    view_753: "f32[512]" = torch.ops.aten.view.default(sum_90, [512]);  sum_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_323: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_321, add_13)
    mul_324: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_321, rsqrt_4);  mul_321 = rsqrt_4 = None
    sum_91: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_204: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_200, mul_324);  add_200 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_115: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    pow_87: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_115, 3);  alias_115 = None
    mul_325: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_91, -0.5);  sum_91 = None
    mul_326: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_325, pow_87);  mul_325 = pow_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_99: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_326, [4, 1024, 512]);  mul_326 = None
    div_49: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_99, 512);  expand_99 = None
    pow_88: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 1.0);  add_13 = None
    mul_327: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_88, 2.0);  pow_88 = None
    mul_328: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_49, mul_327);  div_49 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_205: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_204, mul_328);  add_204 = mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_754: "f32[4096, 512]" = torch.ops.aten.view.default(add_205, [4096, 512])
    permute_674: "f32[512, 4096]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_267: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_674, view_47);  permute_674 = view_47 = None
    permute_675: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    permute_676: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_268: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_754, permute_676);  view_754 = permute_676 = None
    view_755: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_268, [4, 1024, 2048]);  mm_268 = None
    permute_677: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_675, [1, 0]);  permute_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_116: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    le_11: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_116, 0);  alias_116 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_11, scalar_tensor_12, view_755);  le_11 = scalar_tensor_12 = view_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_756: "f32[4096, 2048]" = torch.ops.aten.view.default(where_14, [4096, 2048]);  where_14 = None
    permute_678: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_756, [1, 0])
    mm_269: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_678, view_45);  permute_678 = view_45 = None
    permute_679: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    permute_680: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_270: "f32[4096, 512]" = torch.ops.aten.mm.default(view_756, permute_680);  view_756 = permute_680 = None
    view_757: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_270, [4, 1024, 512]);  mm_270 = None
    permute_681: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_679, [1, 0]);  permute_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_329: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_757, primals_4);  primals_4 = None
    mul_330: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_757, mul_9);  view_757 = mul_9 = None
    sum_92: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1], True);  mul_330 = None
    view_758: "f32[512]" = torch.ops.aten.view.default(sum_92, [512]);  sum_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_331: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_329, add_11)
    mul_332: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_329, rsqrt_3);  mul_329 = rsqrt_3 = None
    sum_93: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_206: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_205, mul_332);  add_205 = mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_117: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    pow_89: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_117, 3);  alias_117 = None
    mul_333: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_93, -0.5);  sum_93 = None
    mul_334: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_333, pow_89);  mul_333 = pow_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_100: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_334, [4, 1024, 512]);  mul_334 = None
    div_50: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_100, 512);  expand_100 = None
    pow_90: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 1.0);  add_11 = None
    mul_335: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_90, 2.0);  pow_90 = None
    mul_336: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_50, mul_335);  div_50 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_207: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_206, mul_336);  add_206 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_759: "f32[4096, 512]" = torch.ops.aten.view.default(add_207, [4096, 512])
    permute_682: "f32[512, 4096]" = torch.ops.aten.permute.default(view_759, [1, 0])
    mm_271: "f32[512, 512]" = torch.ops.aten.mm.default(permute_682, view_43);  permute_682 = view_43 = None
    permute_683: "f32[512, 512]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    permute_684: "f32[512, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_272: "f32[4096, 512]" = torch.ops.aten.mm.default(view_759, permute_684);  view_759 = permute_684 = None
    view_760: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_272, [4, 1024, 512]);  mm_272 = None
    permute_685: "f32[512, 512]" = torch.ops.aten.permute.default(permute_683, [1, 0]);  permute_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_761: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_760, [4, 1024, 8, 64]);  view_760 = None
    permute_686: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_761, [0, 2, 1, 3]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_217: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_686, memory_format = torch.contiguous_format);  permute_686 = None
    view_762: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_217, [32, 1024, 64]);  clone_217 = None
    permute_687: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_39, [0, 2, 1]);  view_39 = None
    bmm_100: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_687, view_762);  permute_687 = None
    permute_688: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_40, [0, 2, 1]);  view_40 = None
    bmm_101: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_762, permute_688);  view_762 = permute_688 = None
    view_763: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_100, [4, 8, 1024, 64]);  bmm_100 = None
    view_764: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_101, [4, 8, 1024, 1024]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_118: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_337: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_764, alias_118);  view_764 = None
    sum_94: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [-1], True)
    mul_338: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_118, sum_94);  alias_118 = sum_94 = None
    sub_39: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_25: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_112: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_25, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_48: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_112, sub_39);  as_strided_112 = sub_39 = None
    as_strided_scatter_32: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_25, copy_48, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_25 = copy_48 = None
    as_strided_115: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_32, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_32 = None
    new_empty_strided_16: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_115, [32, 1024, 1024], [1048576, 1024, 1])
    copy_49: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_16, as_strided_115);  new_empty_strided_16 = as_strided_115 = None
    as_strided_117: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_49, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_218: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_117, memory_format = torch.contiguous_format)
    copy_50: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_117, clone_218);  as_strided_117 = None
    as_strided_scatter_33: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_49, copy_50, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_49 = copy_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_208: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_201, clone_218);  add_201 = clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_689: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_102: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_689, as_strided_scatter_33);  permute_689 = None
    permute_690: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    bmm_103: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_33, permute_690);  as_strided_scatter_33 = permute_690 = None
    view_765: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_102, [4, 8, 64, 1024]);  bmm_102 = None
    view_766: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_103, [4, 8, 1024, 64]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_691: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_765, [0, 1, 3, 2]);  view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_692: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_763, [0, 2, 1, 3]);  view_763 = None
    clone_219: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_692, memory_format = torch.contiguous_format);  permute_692 = None
    view_767: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_219, [4, 1024, 512]);  clone_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_768: "f32[4096, 512]" = torch.ops.aten.view.default(view_767, [4096, 512]);  view_767 = None
    permute_693: "f32[512, 4096]" = torch.ops.aten.permute.default(view_768, [1, 0])
    mm_273: "f32[512, 512]" = torch.ops.aten.mm.default(permute_693, view_31);  permute_693 = view_31 = None
    permute_694: "f32[512, 512]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    permute_695: "f32[512, 512]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm_274: "f32[4096, 512]" = torch.ops.aten.mm.default(view_768, permute_695);  view_768 = permute_695 = None
    view_769: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_274, [4, 1024, 512]);  mm_274 = None
    permute_696: "f32[512, 512]" = torch.ops.aten.permute.default(permute_694, [1, 0]);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_697: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_691, [0, 2, 1, 3]);  permute_691 = None
    view_770: "f32[4, 1024, 512]" = torch.ops.aten.view.default(permute_697, [4, 1024, 512]);  permute_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    clone_220: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_770, memory_format = torch.contiguous_format);  view_770 = None
    view_771: "f32[4096, 512]" = torch.ops.aten.view.default(clone_220, [4096, 512]);  clone_220 = None
    permute_698: "f32[512, 4096]" = torch.ops.aten.permute.default(view_771, [1, 0])
    mm_275: "f32[512, 512]" = torch.ops.aten.mm.default(permute_698, view_28);  permute_698 = view_28 = None
    permute_699: "f32[512, 512]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    permute_700: "f32[512, 512]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_276: "f32[4096, 512]" = torch.ops.aten.mm.default(view_771, permute_700);  view_771 = permute_700 = None
    view_772: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_276, [4, 1024, 512]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_209: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_769, view_772);  view_769 = view_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_701: "f32[512, 512]" = torch.ops.aten.permute.default(permute_699, [1, 0]);  permute_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_702: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
    clone_221: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
    view_773: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_221, [4, 1024, 512]);  clone_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_774: "f32[4096, 512]" = torch.ops.aten.view.default(view_773, [4096, 512]);  view_773 = None
    permute_703: "f32[512, 4096]" = torch.ops.aten.permute.default(view_774, [1, 0])
    mm_277: "f32[512, 512]" = torch.ops.aten.mm.default(permute_703, view_25);  permute_703 = view_25 = None
    permute_704: "f32[512, 512]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    permute_705: "f32[512, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_278: "f32[4096, 512]" = torch.ops.aten.mm.default(view_774, permute_705);  view_774 = permute_705 = None
    view_775: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_278, [4, 1024, 512]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_210: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_209, view_775);  add_209 = view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_706: "f32[512, 512]" = torch.ops.aten.permute.default(permute_704, [1, 0]);  permute_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_339: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_210, primals_3);  primals_3 = None
    mul_340: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_210, mul_7);  add_210 = mul_7 = None
    sum_95: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1], True);  mul_340 = None
    view_776: "f32[512]" = torch.ops.aten.view.default(sum_95, [512]);  sum_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_341: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_339, add_8)
    mul_342: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_339, rsqrt_2);  mul_339 = rsqrt_2 = None
    sum_96: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_211: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_207, mul_342);  add_207 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_119: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    pow_91: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_119, 3);  alias_119 = None
    mul_343: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_96, -0.5);  sum_96 = None
    mul_344: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_343, pow_91);  mul_343 = pow_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_101: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_344, [4, 1024, 512]);  mul_344 = None
    div_51: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_101, 512);  expand_101 = None
    pow_92: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_8, 1.0);  add_8 = None
    mul_345: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_92, 2.0);  pow_92 = None
    mul_346: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_51, mul_345);  div_51 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_212: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_211, mul_346);  add_211 = mul_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    view_777: "f32[4096, 512]" = torch.ops.aten.view.default(add_212, [4096, 512])
    permute_707: "f32[512, 4096]" = torch.ops.aten.permute.default(view_777, [1, 0])
    mm_279: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_707, view_23);  permute_707 = view_23 = None
    permute_708: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    permute_709: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_280: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_777, permute_709);  view_777 = permute_709 = None
    view_778: "f32[4, 1024, 2048]" = torch.ops.aten.view.default(mm_280, [4, 1024, 2048]);  mm_280 = None
    permute_710: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_708, [1, 0]);  permute_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_120: "f32[4, 1024, 2048]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    le_12: "b8[4, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_120, 0);  alias_120 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[4, 1024, 2048]" = torch.ops.aten.where.self(le_12, scalar_tensor_13, view_778);  le_12 = scalar_tensor_13 = view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    view_779: "f32[4096, 2048]" = torch.ops.aten.view.default(where_15, [4096, 2048]);  where_15 = None
    permute_711: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_779, [1, 0])
    mm_281: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_711, view_21);  permute_711 = view_21 = None
    permute_712: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    permute_713: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_282: "f32[4096, 512]" = torch.ops.aten.mm.default(view_779, permute_713);  view_779 = permute_713 = None
    view_780: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_282, [4, 1024, 512]);  mm_282 = None
    permute_714: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_712, [1, 0]);  permute_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_347: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_780, primals_2);  primals_2 = None
    mul_348: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_780, mul_5);  view_780 = mul_5 = None
    sum_97: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 1], True);  mul_348 = None
    view_781: "f32[512]" = torch.ops.aten.view.default(sum_97, [512]);  sum_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_349: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_347, add_6)
    mul_350: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_347, rsqrt_1);  mul_347 = rsqrt_1 = None
    sum_98: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_213: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_212, mul_350);  add_212 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_121: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    pow_93: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_121, 3);  alias_121 = None
    mul_351: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_98, -0.5);  sum_98 = None
    mul_352: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_351, pow_93);  mul_351 = pow_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_102: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_352, [4, 1024, 512]);  mul_352 = None
    div_52: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_102, 512);  expand_102 = None
    pow_94: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 1.0);  add_6 = None
    mul_353: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_94, 2.0);  pow_94 = None
    mul_354: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_52, mul_353);  div_52 = mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_214: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_213, mul_354);  add_213 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    view_782: "f32[4096, 512]" = torch.ops.aten.view.default(add_214, [4096, 512])
    permute_715: "f32[512, 4096]" = torch.ops.aten.permute.default(view_782, [1, 0])
    mm_283: "f32[512, 512]" = torch.ops.aten.mm.default(permute_715, view_19);  permute_715 = view_19 = None
    permute_716: "f32[512, 512]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    permute_717: "f32[512, 512]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_284: "f32[4096, 512]" = torch.ops.aten.mm.default(view_782, permute_717);  view_782 = permute_717 = None
    view_783: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_284, [4, 1024, 512]);  mm_284 = None
    permute_718: "f32[512, 512]" = torch.ops.aten.permute.default(permute_716, [1, 0]);  permute_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_784: "f32[4, 1024, 8, 64]" = torch.ops.aten.view.default(view_783, [4, 1024, 8, 64]);  view_783 = None
    permute_719: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_784, [0, 2, 1, 3]);  view_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    clone_222: "f32[4, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_719, memory_format = torch.contiguous_format);  permute_719 = None
    view_785: "f32[32, 1024, 64]" = torch.ops.aten.view.default(clone_222, [32, 1024, 64]);  clone_222 = None
    permute_720: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    bmm_104: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(permute_720, view_785);  permute_720 = None
    permute_721: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
    bmm_105: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(view_785, permute_721);  view_785 = permute_721 = None
    view_786: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_104, [4, 8, 1024, 64]);  bmm_104 = None
    view_787: "f32[4, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_105, [4, 8, 1024, 1024]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_122: "f32[4, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_355: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_787, alias_122);  view_787 = None
    sum_99: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [-1], True)
    mul_356: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_122, sum_99);  alias_122 = sum_99 = None
    sub_40: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_355, mul_356);  mul_355 = mul_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    full_26: "f32[33554432]" = torch.ops.aten.full.default([33554432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_119: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(full_26, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    copy_51: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_119, sub_40);  as_strided_119 = sub_40 = None
    as_strided_scatter_34: "f32[33554432]" = torch.ops.aten.as_strided_scatter.default(full_26, copy_51, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  full_26 = copy_51 = None
    as_strided_122: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided.default(as_strided_scatter_34, [32, 1024, 1024], [1048576, 1024, 1], 0);  as_strided_scatter_34 = None
    new_empty_strided_17: "f32[32, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(as_strided_122, [32, 1024, 1024], [1048576, 1024, 1])
    copy_52: "f32[32, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided_17, as_strided_122);  new_empty_strided_17 = as_strided_122 = None
    as_strided_124: "f32[4, 8, 1024, 1024]" = torch.ops.aten.as_strided.default(copy_52, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0)
    clone_223: "f32[4, 8, 1024, 1024]" = torch.ops.aten.clone.default(as_strided_124, memory_format = torch.contiguous_format)
    copy_53: "f32[4, 8, 1024, 1024]" = torch.ops.aten.copy.default(as_strided_124, clone_223);  as_strided_124 = None
    as_strided_scatter_35: "f32[32, 1024, 1024]" = torch.ops.aten.as_strided_scatter.default(copy_52, copy_53, [4, 8, 1024, 1024], [8388608, 1048576, 1024, 1], 0);  copy_52 = copy_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_215: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_208, clone_223);  add_208 = clone_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    sum_100: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_215, [0], True);  add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    squeeze_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_100, 0);  sum_100 = None
    permute_722: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    eq_2: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_3, -1)
    unsqueeze_19: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_19, scalar_tensor_14, permute_722);  unsqueeze_19 = scalar_tensor_14 = permute_722 = None
    clone_224: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_16, memory_format = torch.contiguous_format);  where_16 = None
    full_27: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[32, 8]" = torch.ops.aten._unsafe_index_put.default(full_27, [add_3], clone_224, True);  full_27 = add_3 = clone_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_723: "f32[32, 64, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_106: "f32[32, 64, 1024]" = torch.ops.aten.bmm.default(permute_723, as_strided_scatter_35);  permute_723 = None
    permute_724: "f32[32, 1024, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_107: "f32[32, 1024, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_35, permute_724);  as_strided_scatter_35 = permute_724 = None
    view_788: "f32[4, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_106, [4, 8, 64, 1024]);  bmm_106 = None
    view_789: "f32[4, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_107, [4, 8, 1024, 64]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_725: "f32[4, 8, 1024, 64]" = torch.ops.aten.permute.default(view_788, [0, 1, 3, 2]);  view_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_726: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_786, [0, 2, 1, 3]);  view_786 = None
    clone_225: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_726, memory_format = torch.contiguous_format);  permute_726 = None
    view_790: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_225, [4, 1024, 512]);  clone_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    view_791: "f32[4096, 512]" = torch.ops.aten.view.default(view_790, [4096, 512]);  view_790 = None
    permute_727: "f32[512, 4096]" = torch.ops.aten.permute.default(view_791, [1, 0])
    mm_285: "f32[512, 512]" = torch.ops.aten.mm.default(permute_727, view_7);  permute_727 = view_7 = None
    permute_728: "f32[512, 512]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    permute_729: "f32[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_286: "f32[4096, 512]" = torch.ops.aten.mm.default(view_791, permute_729);  view_791 = permute_729 = None
    view_792: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_286, [4, 1024, 512]);  mm_286 = None
    permute_730: "f32[512, 512]" = torch.ops.aten.permute.default(permute_728, [1, 0]);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_731: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_725, [0, 2, 1, 3]);  permute_725 = None
    view_793: "f32[4, 1024, 512]" = torch.ops.aten.view.default(permute_731, [4, 1024, 512]);  permute_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    clone_226: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(view_793, memory_format = torch.contiguous_format);  view_793 = None
    view_794: "f32[4096, 512]" = torch.ops.aten.view.default(clone_226, [4096, 512]);  clone_226 = None
    permute_732: "f32[512, 4096]" = torch.ops.aten.permute.default(view_794, [1, 0])
    mm_287: "f32[512, 512]" = torch.ops.aten.mm.default(permute_732, view_4);  permute_732 = view_4 = None
    permute_733: "f32[512, 512]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    permute_734: "f32[512, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_288: "f32[4096, 512]" = torch.ops.aten.mm.default(view_794, permute_734);  view_794 = permute_734 = None
    view_795: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_288, [4, 1024, 512]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    add_216: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_792, view_795);  view_792 = view_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_735: "f32[512, 512]" = torch.ops.aten.permute.default(permute_733, [1, 0]);  permute_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_736: "f32[4, 1024, 8, 64]" = torch.ops.aten.permute.default(view_789, [0, 2, 1, 3]);  view_789 = None
    clone_227: "f32[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_736, memory_format = torch.contiguous_format);  permute_736 = None
    view_796: "f32[4, 1024, 512]" = torch.ops.aten.view.default(clone_227, [4, 1024, 512]);  clone_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_797: "f32[4096, 512]" = torch.ops.aten.view.default(view_796, [4096, 512]);  view_796 = None
    permute_737: "f32[512, 4096]" = torch.ops.aten.permute.default(view_797, [1, 0])
    mm_289: "f32[512, 512]" = torch.ops.aten.mm.default(permute_737, view_1);  permute_737 = view_1 = None
    permute_738: "f32[512, 512]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    permute_739: "f32[512, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_290: "f32[4096, 512]" = torch.ops.aten.mm.default(view_797, permute_739);  view_797 = permute_739 = None
    view_798: "f32[4, 1024, 512]" = torch.ops.aten.view.default(mm_290, [4, 1024, 512]);  mm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_217: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_216, view_798);  add_216 = view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_740: "f32[512, 512]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_357: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_217, primals_1);  primals_1 = None
    mul_358: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_217, mul_1);  add_217 = mul_1 = None
    sum_101: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1], True);  mul_358 = None
    view_799: "f32[512]" = torch.ops.aten.view.default(sum_101, [512]);  sum_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_359: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_357, clone)
    mul_360: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_357, rsqrt);  mul_357 = rsqrt = None
    sum_102: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_218: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_214, mul_360);  add_214 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    alias_123: "f32[4, 1024, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    pow_95: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(alias_123, 3);  alias_123 = None
    mul_361: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_102, -0.5);  sum_102 = None
    mul_362: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_361, pow_95);  mul_361 = pow_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_103: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_362, [4, 1024, 512]);  mul_362 = None
    div_53: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_103, 512);  expand_103 = None
    pow_96: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(clone, 1.0);  clone = None
    mul_363: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_96, 2.0);  pow_96 = None
    mul_364: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_53, mul_363);  div_53 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_219: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_218, mul_364);  add_218 = mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    eq_3: "b8[4, 1024]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_20: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[4, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_20, scalar_tensor_15, add_219);  unsqueeze_20 = scalar_tensor_15 = add_219 = None
    full_28: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_3: "f32[32128, 512]" = torch.ops.aten._unsafe_index_put.default(full_28, [view], where_17, True);  full_28 = view = where_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    add_220: "f32[32128, 512]" = torch.ops.aten.add.Tensor(_unsafe_index_put_1, _unsafe_index_put_3);  _unsafe_index_put_1 = _unsafe_index_put_3 = None
    return pytree.tree_unflatten([view_411, permute_70, permute_72, permute_80, permute_82, permute_91, permute_93, permute_100, permute_102, permute_111, permute_113, permute_120, permute_122, permute_131, permute_133, permute_140, permute_142, permute_151, permute_153, permute_160, permute_162, permute_171, permute_173, permute_180, permute_182, clone_49, view_799, view_781, view_776, view_758, view_753, view_735, view_730, view_712, view_707, view_689, view_684, view_666, view_661, view_660, view_642, view_624, view_619, view_601, view_583, view_578, view_560, view_542, view_537, view_519, view_501, view_496, view_478, view_460, view_455, view_437, view_419, view_414, add_220, permute_740, permute_735, permute_730, _unsafe_index_put_2, permute_718, permute_714, permute_710, permute_706, permute_701, permute_696, permute_685, permute_681, permute_677, permute_673, permute_668, permute_663, permute_652, permute_648, permute_644, permute_640, permute_635, permute_630, permute_619, permute_615, permute_611, permute_607, permute_602, permute_597, permute_586, permute_582, permute_578, permute_574, permute_569, permute_564, permute_553, permute_549, permute_545, permute_541, permute_536, permute_531, _unsafe_index_put, permute_519, permute_515, permute_510, permute_505, permute_494, permute_490, permute_486, permute_482, permute_477, permute_472, permute_461, permute_457, permute_452, permute_447, permute_436, permute_432, permute_428, permute_424, permute_419, permute_414, permute_403, permute_399, permute_394, permute_389, permute_378, permute_374, permute_370, permute_366, permute_361, permute_356, permute_345, permute_341, permute_336, permute_331, permute_320, permute_316, permute_312, permute_308, permute_303, permute_298, permute_287, permute_283, permute_278, permute_273, permute_262, permute_258, permute_254, permute_250, permute_245, permute_240, permute_229, permute_225, permute_220, permute_215, permute_204, permute_200, permute_196, permute_192, None, None], self._out_spec)
    