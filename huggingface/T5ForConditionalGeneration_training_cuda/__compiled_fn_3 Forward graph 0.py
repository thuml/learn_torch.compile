from __future__ import annotations



def forward(self, primals_1: "f32[512]", primals_2: "f32[512]", primals_3: "f32[512]", primals_4: "f32[512]", primals_5: "f32[512]", primals_6: "f32[512]", primals_7: "f32[512]", primals_8: "f32[512]", primals_9: "f32[512]", primals_10: "f32[512]", primals_11: "f32[512]", primals_12: "f32[512]", primals_13: "f32[512]", primals_14: "f32[512]", primals_15: "f32[512]", primals_16: "f32[512]", primals_17: "f32[512]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "f32[512]", primals_21: "f32[512]", primals_22: "f32[512]", primals_23: "f32[512]", primals_24: "f32[512]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512]", primals_29: "f32[512]", primals_30: "f32[512]", primals_31: "f32[512]", primals_32: "f32[512]", primals_33: "f32[32128, 512]", primals_34: "f32[512, 512]", primals_35: "f32[512, 512]", primals_36: "f32[512, 512]", primals_37: "f32[32, 8]", primals_38: "f32[512, 512]", primals_39: "f32[2048, 512]", primals_40: "f32[512, 2048]", primals_41: "f32[512, 512]", primals_42: "f32[512, 512]", primals_43: "f32[512, 512]", primals_44: "f32[512, 512]", primals_45: "f32[2048, 512]", primals_46: "f32[512, 2048]", primals_47: "f32[512, 512]", primals_48: "f32[512, 512]", primals_49: "f32[512, 512]", primals_50: "f32[512, 512]", primals_51: "f32[2048, 512]", primals_52: "f32[512, 2048]", primals_53: "f32[512, 512]", primals_54: "f32[512, 512]", primals_55: "f32[512, 512]", primals_56: "f32[512, 512]", primals_57: "f32[2048, 512]", primals_58: "f32[512, 2048]", primals_59: "f32[512, 512]", primals_60: "f32[512, 512]", primals_61: "f32[512, 512]", primals_62: "f32[512, 512]", primals_63: "f32[2048, 512]", primals_64: "f32[512, 2048]", primals_65: "f32[512, 512]", primals_66: "f32[512, 512]", primals_67: "f32[512, 512]", primals_68: "f32[512, 512]", primals_69: "f32[2048, 512]", primals_70: "f32[512, 2048]", primals_71: "f32[512, 512]", primals_72: "f32[512, 512]", primals_73: "f32[512, 512]", primals_74: "f32[32, 8]", primals_75: "f32[512, 512]", primals_76: "f32[512, 512]", primals_77: "f32[512, 512]", primals_78: "f32[512, 512]", primals_79: "f32[512, 512]", primals_80: "f32[2048, 512]", primals_81: "f32[512, 2048]", primals_82: "f32[512, 512]", primals_83: "f32[512, 512]", primals_84: "f32[512, 512]", primals_85: "f32[512, 512]", primals_86: "f32[512, 512]", primals_87: "f32[512, 512]", primals_88: "f32[512, 512]", primals_89: "f32[512, 512]", primals_90: "f32[2048, 512]", primals_91: "f32[512, 2048]", primals_92: "f32[512, 512]", primals_93: "f32[512, 512]", primals_94: "f32[512, 512]", primals_95: "f32[512, 512]", primals_96: "f32[512, 512]", primals_97: "f32[512, 512]", primals_98: "f32[512, 512]", primals_99: "f32[512, 512]", primals_100: "f32[2048, 512]", primals_101: "f32[512, 2048]", primals_102: "f32[512, 512]", primals_103: "f32[512, 512]", primals_104: "f32[512, 512]", primals_105: "f32[512, 512]", primals_106: "f32[512, 512]", primals_107: "f32[512, 512]", primals_108: "f32[512, 512]", primals_109: "f32[512, 512]", primals_110: "f32[2048, 512]", primals_111: "f32[512, 2048]", primals_112: "f32[512, 512]", primals_113: "f32[512, 512]", primals_114: "f32[512, 512]", primals_115: "f32[512, 512]", primals_116: "f32[512, 512]", primals_117: "f32[512, 512]", primals_118: "f32[512, 512]", primals_119: "f32[512, 512]", primals_120: "f32[2048, 512]", primals_121: "f32[512, 2048]", primals_122: "f32[512, 512]", primals_123: "f32[512, 512]", primals_124: "f32[512, 512]", primals_125: "f32[512, 512]", primals_126: "f32[512, 512]", primals_127: "f32[512, 512]", primals_128: "f32[512, 512]", primals_129: "f32[512, 512]", primals_130: "f32[2048, 512]", primals_131: "f32[512, 2048]", primals_132: "f32[32128, 512]", primals_133: "i64[1, 1024]", primals_134: "i64[1, 1024]", primals_135: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1011, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 1024]" = torch.ops.aten.view.default(primals_133, [-1, 1024]);  primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding: "f32[1, 1024, 512]" = torch.ops.aten.embedding.default(primals_33, view)
    
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
    mul_1: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(getitem, rsqrt)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_2: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_1, mul_1);  mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute: "f32[512, 512]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    view_1: "f32[1024, 512]" = torch.ops.aten.view.default(mul_2, [1024, 512]);  mul_2 = None
    mm: "f32[1024, 512]" = torch.ops.aten.mm.default(view_1, permute)
    view_2: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm, [1, 1024, 512]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_3: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_2, [1, -1, 8, 64]);  view_2 = None
    permute_1: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    mm_1: "f32[1024, 512]" = torch.ops.aten.mm.default(view_1, permute_2)
    view_5: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_1, [1, 1024, 512]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_6: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_5, [1, -1, 8, 64]);  view_5 = None
    permute_3: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    mm_2: "f32[1024, 512]" = torch.ops.aten.mm.default(view_1, permute_4)
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
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    slice_3: "i64[1024]" = torch.ops.aten.slice.Tensor(iota, 0, 0, 9223372036854775807)
    unsqueeze_2: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:442, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    unsqueeze_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    slice_4: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 1, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:443, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    sub_1: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(slice_4, unsqueeze_2);  unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:414, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    gt: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(sub_1, 0)
    convert_element_type: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
    mul_3: "i64[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
    add_1: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(mul_3, 0);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:415, code: relative_position = torch.abs(relative_position)
    abs_1: "i64[1024, 1024]" = torch.ops.aten.abs.default(sub_1)
    
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
    full_default_1: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:430, code: relative_position_if_large = torch.min(
    minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_2, full_default_1);  add_2 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:434, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
    add_3: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    embedding_1: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(primals_37, add_3);  primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_7: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    unsqueeze_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_12, unsqueeze_4);  view_12 = None
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
    mul_5: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_6: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_2, mul_5);  mul_5 = None
    
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
    mul_7: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_8, rsqrt_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_8: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_3, mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    view_25: "f32[1024, 512]" = torch.ops.aten.view.default(mul_8, [1024, 512]);  mul_8 = None
    mm_6: "f32[1024, 512]" = torch.ops.aten.mm.default(view_25, permute_12)
    view_26: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_6, [1, 1024, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_27: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_26, [1, -1, 8, 64]);  view_26 = None
    permute_13: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_14: "f32[512, 512]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    mm_7: "f32[1024, 512]" = torch.ops.aten.mm.default(view_25, permute_14)
    view_29: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_7, [1, 1024, 512]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_30: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_29, [1, -1, 8, 64]);  view_29 = None
    permute_15: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_16: "f32[512, 512]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    mm_8: "f32[1024, 512]" = torch.ops.aten.mm.default(view_25, permute_16)
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
    add_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_36, unsqueeze_4);  view_36 = None
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
    mul_9: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_10: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_4, mul_9);  mul_9 = None
    
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
    mul_11: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_12: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_5, mul_11);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_23: "f32[512, 512]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    view_49: "f32[1024, 512]" = torch.ops.aten.view.default(mul_12, [1024, 512]);  mul_12 = None
    mm_12: "f32[1024, 512]" = torch.ops.aten.mm.default(view_49, permute_23)
    view_50: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_12, [1, 1024, 512]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_51: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_50, [1, -1, 8, 64]);  view_50 = None
    permute_24: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    mm_13: "f32[1024, 512]" = torch.ops.aten.mm.default(view_49, permute_25)
    view_53: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_13, [1, 1024, 512]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_54: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_53, [1, -1, 8, 64]);  view_53 = None
    permute_26: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_27: "f32[512, 512]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    mm_14: "f32[1024, 512]" = torch.ops.aten.mm.default(view_49, permute_27)
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
    add_15: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_60, unsqueeze_4);  view_60 = None
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
    mul_13: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_14: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_6, mul_13);  mul_13 = None
    
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
    mul_15: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_18, rsqrt_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_16: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_7, mul_15);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    view_73: "f32[1024, 512]" = torch.ops.aten.view.default(mul_16, [1024, 512]);  mul_16 = None
    mm_18: "f32[1024, 512]" = torch.ops.aten.mm.default(view_73, permute_34)
    view_74: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_18, [1, 1024, 512]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_75: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_74, [1, -1, 8, 64]);  view_74 = None
    permute_35: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_36: "f32[512, 512]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    mm_19: "f32[1024, 512]" = torch.ops.aten.mm.default(view_73, permute_36)
    view_77: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_19, [1, 1024, 512]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_78: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_77, [1, -1, 8, 64]);  view_77 = None
    permute_37: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_38: "f32[512, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    mm_20: "f32[1024, 512]" = torch.ops.aten.mm.default(view_73, permute_38)
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
    add_20: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_84, unsqueeze_4);  view_84 = None
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
    mul_17: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_18: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_8, mul_17);  mul_17 = None
    
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
    mul_19: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_23, rsqrt_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_20: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_9, mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_45: "f32[512, 512]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    view_97: "f32[1024, 512]" = torch.ops.aten.view.default(mul_20, [1024, 512]);  mul_20 = None
    mm_24: "f32[1024, 512]" = torch.ops.aten.mm.default(view_97, permute_45)
    view_98: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_24, [1, 1024, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_99: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_98, [1, -1, 8, 64]);  view_98 = None
    permute_46: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_47: "f32[512, 512]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    mm_25: "f32[1024, 512]" = torch.ops.aten.mm.default(view_97, permute_47)
    view_101: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_25, [1, 1024, 512]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_102: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_101, [1, -1, 8, 64]);  view_101 = None
    permute_48: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_49: "f32[512, 512]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    mm_26: "f32[1024, 512]" = torch.ops.aten.mm.default(view_97, permute_49)
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
    add_25: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_108, unsqueeze_4);  view_108 = None
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
    mul_21: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_22: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_10, mul_21);  mul_21 = None
    
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
    mul_23: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_28, rsqrt_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_24: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_11, mul_23);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_56: "f32[512, 512]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    view_121: "f32[1024, 512]" = torch.ops.aten.view.default(mul_24, [1024, 512]);  mul_24 = None
    mm_30: "f32[1024, 512]" = torch.ops.aten.mm.default(view_121, permute_56)
    view_122: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_30, [1, 1024, 512]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_123: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_122, [1, -1, 8, 64]);  view_122 = None
    permute_57: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_58: "f32[512, 512]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    mm_31: "f32[1024, 512]" = torch.ops.aten.mm.default(view_121, permute_58)
    view_125: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_31, [1, 1024, 512]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_126: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_125, [1, -1, 8, 64]);  view_125 = None
    permute_59: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_60: "f32[512, 512]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    mm_32: "f32[1024, 512]" = torch.ops.aten.mm.default(view_121, permute_60)
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
    add_30: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_132, unsqueeze_4);  view_132 = unsqueeze_4 = None
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
    mul_25: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_26: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_12, mul_25);  mul_25 = None
    
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
    mul_27: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_33, rsqrt_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_28: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_13, mul_27);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    native_dropout_25 = torch.ops.aten.native_dropout.default(mul_28, 0.1, True);  mul_28 = None
    getitem_50: "f32[1, 1024, 512]" = native_dropout_25[0]
    getitem_51: "b8[1, 1024, 512]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1011, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_145: "i64[1, 1024]" = torch.ops.aten.view.default(primals_135, [-1, 1024]);  primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding_2: "f32[1, 1024, 512]" = torch.ops.aten.embedding.default(primals_33, view_145);  primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:861, code: causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    unsqueeze_6: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
    slice_5: "i64[1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_6, 2, 0, 9223372036854775807);  unsqueeze_6 = None
    repeat: "i64[1, 1024, 1024]" = torch.ops.aten.repeat.default(slice_5, [1, 1024, 1]);  slice_5 = None
    unsqueeze_8: "i64[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_4, 2);  slice_4 = None
    le: "b8[1, 1024, 1024]" = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:864, code: causal_mask = causal_mask.to(attention_mask.dtype)
    convert_element_type_3: "f32[1, 1024, 1024]" = torch.ops.prims.convert_element_type.default(le, torch.float32);  le = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:876, code: extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    slice_7: "f32[1, 1024, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_3, 0, 0, 9223372036854775807);  convert_element_type_3 = None
    unsqueeze_9: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(slice_7, 1);  slice_7 = None
    slice_8: "f32[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_9, 2, 0, 9223372036854775807);  unsqueeze_9 = None
    slice_9: "f32[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 9223372036854775807);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub_8: "f32[1, 1, 1024, 1024]" = torch.ops.aten.sub.Tensor(1.0, slice_9);  slice_9 = None
    mul_30: "f32[1, 1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_8, -3.4028234663852886e+38);  sub_8 = None
    
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
    mul_32: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(getitem_52, rsqrt_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_33: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_14, mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    view_146: "f32[1024, 512]" = torch.ops.aten.view.default(mul_33, [1024, 512]);  mul_33 = None
    mm_36: "f32[1024, 512]" = torch.ops.aten.mm.default(view_146, permute_67)
    view_147: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_36, [1, 1024, 512]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_148: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_147, [1, -1, 8, 64]);  view_147 = None
    permute_68: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_69: "f32[512, 512]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    mm_37: "f32[1024, 512]" = torch.ops.aten.mm.default(view_146, permute_69)
    view_150: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_37, [1, 1024, 512]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_151: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_150, [1, -1, 8, 64]);  view_150 = None
    permute_70: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_71: "f32[512, 512]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    mm_38: "f32[1024, 512]" = torch.ops.aten.mm.default(view_146, permute_71)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:417, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    full_default_3: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub_1, full_default_3);  sub_1 = full_default_3 = None
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
    full_default_4: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:430, code: relative_position_if_large = torch.min(
    minimum_2: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_36, full_default_4);  add_36 = full_default_4 = None
    
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
    mul_35: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_40, rsqrt_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_36: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_15, mul_35);  mul_35 = None
    
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
    mm_42: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_81)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    view_177: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, [1, 8, 1024, 1024]);  bmm_14 = None
    view_178: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(view_177, [8, 1024, 1024]);  view_177 = None
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
    mul_37: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_38: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_16, mul_37);  mul_37 = None
    
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
    mul_39: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_40: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_17, mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    view_190: "f32[1024, 512]" = torch.ops.aten.view.default(mul_40, [1024, 512]);  mul_40 = None
    mm_46: "f32[1024, 512]" = torch.ops.aten.mm.default(view_190, permute_88)
    view_191: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_46, [1, 1024, 512]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_192: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_191, [1, -1, 8, 64]);  view_191 = None
    permute_89: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_90: "f32[512, 512]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    mm_47: "f32[1024, 512]" = torch.ops.aten.mm.default(view_190, permute_90)
    view_194: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_47, [1, 1024, 512]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_195: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_194, [1, -1, 8, 64]);  view_194 = None
    permute_91: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_92: "f32[512, 512]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    mm_48: "f32[1024, 512]" = torch.ops.aten.mm.default(view_190, permute_92)
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
    mul_41: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_42: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_18, mul_41);  mul_41 = None
    
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
    mm_51: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_99)
    view_214: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_51, [1, 1024, 512]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_215: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_214, [1, -1, 8, 64]);  view_214 = None
    permute_100: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_101: "f32[512, 512]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    mm_52: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_101)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    view_221: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, [1, 8, 1024, 1024]);  bmm_18 = None
    view_222: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(view_221, [8, 1024, 1024]);  view_221 = None
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
    mul_43: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_44: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_19, mul_43);  mul_43 = None
    
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
    mul_45: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_46: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_20, mul_45);  mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_108: "f32[512, 512]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    view_234: "f32[1024, 512]" = torch.ops.aten.view.default(mul_46, [1024, 512]);  mul_46 = None
    mm_56: "f32[1024, 512]" = torch.ops.aten.mm.default(view_234, permute_108)
    view_235: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_56, [1, 1024, 512]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_236: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_235, [1, -1, 8, 64]);  view_235 = None
    permute_109: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_110: "f32[512, 512]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    mm_57: "f32[1024, 512]" = torch.ops.aten.mm.default(view_234, permute_110)
    view_238: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_57, [1, 1024, 512]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_239: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_238, [1, -1, 8, 64]);  view_238 = None
    permute_111: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_112: "f32[512, 512]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    mm_58: "f32[1024, 512]" = torch.ops.aten.mm.default(view_234, permute_112)
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
    mul_47: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_57, rsqrt_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_48: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_21, mul_47);  mul_47 = None
    
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
    mm_61: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_119)
    view_258: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_61, [1, 1024, 512]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_259: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_258, [1, -1, 8, 64]);  view_258 = None
    permute_120: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_121: "f32[512, 512]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    mm_62: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_121)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    view_265: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, [1, 8, 1024, 1024]);  bmm_22 = None
    view_266: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(view_265, [8, 1024, 1024]);  view_265 = None
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
    mul_49: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_50: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_22, mul_49);  mul_49 = None
    
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
    mul_51: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_52: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_23, mul_51);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_128: "f32[512, 512]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    view_278: "f32[1024, 512]" = torch.ops.aten.view.default(mul_52, [1024, 512]);  mul_52 = None
    mm_66: "f32[1024, 512]" = torch.ops.aten.mm.default(view_278, permute_128)
    view_279: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_66, [1, 1024, 512]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_280: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_279, [1, -1, 8, 64]);  view_279 = None
    permute_129: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    mm_67: "f32[1024, 512]" = torch.ops.aten.mm.default(view_278, permute_130)
    view_282: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_67, [1, 1024, 512]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_283: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_282, [1, -1, 8, 64]);  view_282 = None
    permute_131: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_132: "f32[512, 512]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    mm_68: "f32[1024, 512]" = torch.ops.aten.mm.default(view_278, permute_132)
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
    mul_53: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_54: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_24, mul_53);  mul_53 = None
    
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
    mm_71: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_139)
    view_302: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_71, [1, 1024, 512]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_303: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_302, [1, -1, 8, 64]);  view_302 = None
    permute_140: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_141: "f32[512, 512]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    mm_72: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_141)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    view_309: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_26, [1, 8, 1024, 1024]);  bmm_26 = None
    view_310: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(view_309, [8, 1024, 1024]);  view_309 = None
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
    mul_55: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_68, rsqrt_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_56: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_25, mul_55);  mul_55 = None
    
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
    mul_57: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_58: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_26, mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_148: "f32[512, 512]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    view_322: "f32[1024, 512]" = torch.ops.aten.view.default(mul_58, [1024, 512]);  mul_58 = None
    mm_76: "f32[1024, 512]" = torch.ops.aten.mm.default(view_322, permute_148)
    view_323: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_76, [1, 1024, 512]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_324: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_323, [1, -1, 8, 64]);  view_323 = None
    permute_149: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_150: "f32[512, 512]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    mm_77: "f32[1024, 512]" = torch.ops.aten.mm.default(view_322, permute_150)
    view_326: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_77, [1, 1024, 512]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_327: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_326, [1, -1, 8, 64]);  view_326 = None
    permute_151: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_152: "f32[512, 512]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    mm_78: "f32[1024, 512]" = torch.ops.aten.mm.default(view_322, permute_152)
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
    mul_59: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_73, rsqrt_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_60: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_27, mul_59);  mul_59 = None
    
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
    mm_81: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_159)
    view_346: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_81, [1, 1024, 512]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_347: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_346, [1, -1, 8, 64]);  view_346 = None
    permute_160: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_161: "f32[512, 512]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    mm_82: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_161)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    view_353: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_30, [1, 8, 1024, 1024]);  bmm_30 = None
    view_354: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(view_353, [8, 1024, 1024]);  view_353 = None
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
    mul_61: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_62: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_28, mul_61);  mul_61 = None
    
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
    mul_63: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_64: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_29, mul_63);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_168: "f32[512, 512]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    view_366: "f32[1024, 512]" = torch.ops.aten.view.default(mul_64, [1024, 512]);  mul_64 = None
    mm_86: "f32[1024, 512]" = torch.ops.aten.mm.default(view_366, permute_168)
    view_367: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_86, [1, 1024, 512]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_368: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_367, [1, -1, 8, 64]);  view_367 = None
    permute_169: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_170: "f32[512, 512]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    mm_87: "f32[1024, 512]" = torch.ops.aten.mm.default(view_366, permute_170)
    view_370: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_87, [1, 1024, 512]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_371: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_370, [1, -1, 8, 64]);  view_370 = None
    permute_171: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_172: "f32[512, 512]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    mm_88: "f32[1024, 512]" = torch.ops.aten.mm.default(view_366, permute_172)
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
    mul_65: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_66: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_30, mul_65);  mul_65 = None
    
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
    mm_91: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_179)
    view_390: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_91, [1, 1024, 512]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_391: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_390, [1, -1, 8, 64]);  view_390 = None
    permute_180: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_181: "f32[512, 512]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    mm_92: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_181)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    view_397: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_34, [1, 8, 1024, 1024]);  bmm_34 = None
    view_398: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(view_397, [8, 1024, 1024]);  view_397 = None
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
    mul_67: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_68: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_31, mul_67);  mul_67 = None
    
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
    mul_69: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_70: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_32, mul_69);  mul_69 = None
    
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
    view_413: "i64[1024]" = torch.ops.aten.view.default(primals_134, [-1])
    amax_18: "f32[1024, 1]" = torch.ops.aten.amax.default(view_412, [1], True)
    sub_23: "f32[1024, 32128]" = torch.ops.aten.sub.Tensor(view_412, amax_18);  view_412 = amax_18 = None
    exp_18: "f32[1024, 32128]" = torch.ops.aten.exp.default(sub_23)
    sum_19: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
    log_2: "f32[1024, 1]" = torch.ops.aten.log.default(sum_19);  sum_19 = None
    sub_24: "f32[1024, 32128]" = torch.ops.aten.sub.Tensor(sub_23, log_2);  sub_23 = log_2 = None
    ne: "b8[1024]" = torch.ops.aten.ne.Scalar(view_413, -100)
    full_default_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_2: "i64[1024]" = torch.ops.aten.where.self(ne, view_413, full_default_6);  view_413 = full_default_6 = None
    unsqueeze_17: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_24, 1, unsqueeze_17);  unsqueeze_17 = None
    squeeze: "f32[1024]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg_1: "f32[1024]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_3: "f32[1024]" = torch.ops.aten.where.self(ne, neg_1, full_default_7);  neg_1 = full_default_7 = None
    sum_20: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type_7: "f32[]" = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
    sum_21: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_22: "f32[]" = torch.ops.aten.div.Tensor(sum_21, convert_element_type_7);  sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1774, code: lm_logits = self.lm_head(sequence_output)
    permute_191: "f32[32128, 512]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_195: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_65: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_1: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_65, 0);  alias_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_199: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_203: "f32[512, 512]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_206: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_400, [0, 2, 1]);  view_400 = None
    permute_207: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_401, [0, 2, 1]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_67: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_208: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_395, [0, 2, 1]);  view_395 = None
    permute_209: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_396, [0, 2, 1]);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_214: "f32[512, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_219: "f32[512, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_224: "f32[512, 512]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_228: "f32[512, 512]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_231: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None
    permute_232: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_69: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_233: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
    permute_234: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_239: "f32[512, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_244: "f32[512, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_249: "f32[512, 512]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_253: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_71: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_2: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_71, 0);  alias_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_257: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_261: "f32[512, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_264: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_356, [0, 2, 1]);  view_356 = None
    permute_265: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_357, [0, 2, 1]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_73: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_266: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
    permute_267: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_272: "f32[512, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_277: "f32[512, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_282: "f32[512, 512]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_286: "f32[512, 512]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_289: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None
    permute_290: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_75: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_291: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
    permute_292: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_297: "f32[512, 512]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_302: "f32[512, 512]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_307: "f32[512, 512]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_311: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_77: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_3: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_77, 0);  alias_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_315: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_319: "f32[512, 512]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_322: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
    permute_323: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_79: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_324: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
    permute_325: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_330: "f32[512, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_335: "f32[512, 512]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_340: "f32[512, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_344: "f32[512, 512]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_347: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
    permute_348: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_81: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_349: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    permute_350: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_355: "f32[512, 512]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_360: "f32[512, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_365: "f32[512, 512]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_369: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_83: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_4: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_83, 0);  alias_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_373: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_377: "f32[512, 512]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_380: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
    permute_381: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_85: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_382: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
    permute_383: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_388: "f32[512, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_393: "f32[512, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_398: "f32[512, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_402: "f32[512, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_405: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
    permute_406: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_87: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_407: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
    permute_408: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_413: "f32[512, 512]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_418: "f32[512, 512]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_423: "f32[512, 512]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_427: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_89: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le_5: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_89, 0);  alias_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_431: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_435: "f32[512, 512]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_438: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
    permute_439: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_91: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_440: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
    permute_441: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_446: "f32[512, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_451: "f32[512, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_456: "f32[512, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_460: "f32[512, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_463: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    permute_464: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_93: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_465: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
    permute_466: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_471: "f32[512, 512]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_476: "f32[512, 512]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_485: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_95: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_6: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_95, 0);  alias_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_489: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_493: "f32[512, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_496: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_180, [0, 2, 1]);  view_180 = None
    permute_497: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_97: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_498: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
    permute_499: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_504: "f32[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_509: "f32[512, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_514: "f32[512, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_518: "f32[512, 512]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_521: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
    permute_522: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_99: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_524: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    permute_525: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_530: "f32[512, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_535: "f32[512, 512]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_540: "f32[512, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_544: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_102: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    le_7: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_102, 0);  alias_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_548: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_552: "f32[512, 512]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_555: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    permute_556: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_104: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_557: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
    permute_558: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_563: "f32[512, 512]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_568: "f32[512, 512]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_573: "f32[512, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_577: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_106: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    le_8: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_581: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_585: "f32[512, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_588: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    permute_589: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_108: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_590: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
    permute_591: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_596: "f32[512, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_601: "f32[512, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_606: "f32[512, 512]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_610: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_110: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    le_9: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_110, 0);  alias_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_614: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_618: "f32[512, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_621: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
    permute_622: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_112: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_623: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    permute_624: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_629: "f32[512, 512]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_634: "f32[512, 512]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_639: "f32[512, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_643: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_114: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    le_10: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_114, 0);  alias_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_647: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_651: "f32[512, 512]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_654: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    permute_655: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_116: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_656: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    permute_657: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_662: "f32[512, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_667: "f32[512, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_672: "f32[512, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_676: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_118: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    le_11: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_680: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_684: "f32[512, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_687: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_39, [0, 2, 1]);  view_39 = None
    permute_688: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_40, [0, 2, 1]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_120: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_689: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    permute_690: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_695: "f32[512, 512]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_700: "f32[512, 512]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_705: "f32[512, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_709: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    alias_122: "f32[1, 1024, 2048]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    le_12: "b8[1, 1024, 2048]" = torch.ops.aten.le.Scalar(alias_122, 0);  alias_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_713: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_717: "f32[512, 512]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_720: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    permute_721: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_124: "f32[1, 8, 1024, 1024]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    permute_723: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    permute_724: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_729: "f32[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_734: "f32[512, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_739: "f32[512, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [div_22, view_411, permute_70, permute_72, permute_80, permute_82, permute_91, permute_93, permute_100, permute_102, permute_111, permute_113, permute_120, permute_122, permute_131, permute_133, permute_140, permute_142, permute_151, permute_153, permute_160, permute_162, permute_171, permute_173, permute_180, permute_182, getitem_50, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_134, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, getitem_7, view_23, getitem_9, add_8, rsqrt_2, view_25, getitem_11, view_43, getitem_13, add_11, rsqrt_3, view_45, getitem_15, view_47, getitem_17, add_13, rsqrt_4, view_49, getitem_19, view_67, getitem_21, add_16, rsqrt_5, view_69, getitem_23, view_71, getitem_25, add_18, rsqrt_6, view_73, getitem_27, view_91, getitem_29, add_21, rsqrt_7, view_93, getitem_31, view_95, getitem_33, add_23, rsqrt_8, view_97, getitem_35, view_115, getitem_37, add_26, rsqrt_9, view_117, getitem_39, view_119, getitem_41, add_28, rsqrt_10, view_121, getitem_43, view_139, getitem_45, add_31, rsqrt_11, view_141, getitem_47, view_143, getitem_49, add_33, rsqrt_12, getitem_51, view_145, getitem_52, getitem_53, rsqrt_13, view_146, add_37, getitem_55, view_164, getitem_57, add_40, rsqrt_14, view_166, view_169, getitem_59, view_184, getitem_61, add_44, rsqrt_15, view_186, getitem_63, view_188, getitem_65, add_46, rsqrt_16, view_190, getitem_67, view_208, getitem_69, add_49, rsqrt_17, view_210, getitem_71, view_228, getitem_73, add_52, rsqrt_18, view_230, getitem_75, view_232, getitem_77, add_54, rsqrt_19, view_234, getitem_79, view_252, getitem_81, add_57, rsqrt_20, view_254, getitem_83, view_272, getitem_85, add_60, rsqrt_21, view_274, getitem_87, view_276, getitem_89, add_62, rsqrt_22, view_278, getitem_91, view_296, getitem_93, add_65, rsqrt_23, view_298, getitem_95, view_316, getitem_97, add_68, rsqrt_24, view_318, getitem_99, view_320, getitem_101, add_70, rsqrt_25, view_322, getitem_103, view_340, getitem_105, add_73, rsqrt_26, view_342, getitem_107, view_360, getitem_109, add_76, rsqrt_27, view_362, getitem_111, view_364, getitem_113, add_78, rsqrt_28, view_366, getitem_115, view_384, getitem_117, add_81, rsqrt_29, view_386, getitem_119, view_404, getitem_121, add_84, rsqrt_30, view_406, getitem_123, view_408, getitem_125, add_86, rsqrt_31, getitem_127, view_410, sub_24, convert_element_type_7, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, alias_67, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, alias_69, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, alias_73, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, alias_75, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, alias_79, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, alias_81, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, alias_85, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, alias_87, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, alias_91, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, alias_93, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, alias_97, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, alias_99, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, alias_104, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, alias_108, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, alias_112, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, alias_116, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, alias_120, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, alias_124, permute_723, permute_724, permute_729, permute_734, permute_739]
    