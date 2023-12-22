from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[30522, 768]"; primals_2: "f32[512, 768]"; primals_3: "f32[768]"; primals_4: "f32[768]"; primals_5: "f32[768, 768]"; primals_6: "f32[768]"; primals_7: "f32[768, 768]"; primals_8: "f32[768]"; primals_9: "f32[768, 768]"; primals_10: "f32[768]"; primals_11: "f32[768, 768]"; primals_12: "f32[768]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[3072, 768]"; primals_16: "f32[3072]"; primals_17: "f32[768, 3072]"; primals_18: "f32[768]"; primals_19: "f32[768]"; primals_20: "f32[768]"; primals_21: "f32[768, 768]"; primals_22: "f32[768]"; primals_23: "f32[768, 768]"; primals_24: "f32[768]"; primals_25: "f32[768, 768]"; primals_26: "f32[768]"; primals_27: "f32[768, 768]"; primals_28: "f32[768]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[3072, 768]"; primals_32: "f32[3072]"; primals_33: "f32[768, 3072]"; primals_34: "f32[768]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[768, 768]"; primals_38: "f32[768]"; primals_39: "f32[768, 768]"; primals_40: "f32[768]"; primals_41: "f32[768, 768]"; primals_42: "f32[768]"; primals_43: "f32[768, 768]"; primals_44: "f32[768]"; primals_45: "f32[768]"; primals_46: "f32[768]"; primals_47: "f32[3072, 768]"; primals_48: "f32[3072]"; primals_49: "f32[768, 3072]"; primals_50: "f32[768]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768, 768]"; primals_54: "f32[768]"; primals_55: "f32[768, 768]"; primals_56: "f32[768]"; primals_57: "f32[768, 768]"; primals_58: "f32[768]"; primals_59: "f32[768, 768]"; primals_60: "f32[768]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[3072, 768]"; primals_64: "f32[3072]"; primals_65: "f32[768, 3072]"; primals_66: "f32[768]"; primals_67: "f32[768]"; primals_68: "f32[768]"; primals_69: "f32[768, 768]"; primals_70: "f32[768]"; primals_71: "f32[768, 768]"; primals_72: "f32[768]"; primals_73: "f32[768, 768]"; primals_74: "f32[768]"; primals_75: "f32[768, 768]"; primals_76: "f32[768]"; primals_77: "f32[768]"; primals_78: "f32[768]"; primals_79: "f32[3072, 768]"; primals_80: "f32[3072]"; primals_81: "f32[768, 3072]"; primals_82: "f32[768]"; primals_83: "f32[768]"; primals_84: "f32[768]"; primals_85: "f32[768, 768]"; primals_86: "f32[768]"; primals_87: "f32[768, 768]"; primals_88: "f32[768]"; primals_89: "f32[768, 768]"; primals_90: "f32[768]"; primals_91: "f32[768, 768]"; primals_92: "f32[768]"; primals_93: "f32[768]"; primals_94: "f32[768]"; primals_95: "f32[3072, 768]"; primals_96: "f32[3072]"; primals_97: "f32[768, 3072]"; primals_98: "f32[768]"; primals_99: "f32[768]"; primals_100: "f32[768]"; primals_101: "f32[768, 768]"; primals_102: "f32[768]"; primals_103: "f32[768]"; primals_104: "f32[768]"; primals_105: "f32[30522, 768]"; primals_106: "f32[30522]"; primals_107: "i64[1, 512]"; primals_108: "i64[1, 128]"; primals_109: "i64[1, 128]"; tangents_1: "f32[]"; tangents_2: "f32[1, 128, 30522]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:602, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
    full: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:120, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
    embedding: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(primals_1, primals_108, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:128, code: position_ids = self.position_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_107, 0, 0, 9223372036854775807);  primals_107 = None
    slice_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 128);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:133, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
    embedding_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(primals_2, slice_2);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:135, code: embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
    add: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:136, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1)
    mul: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul, primals_3);  mul = None
    add_2: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:137, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
    native_dropout = torch.ops.aten.native_dropout.default(add_2, 0.1, True);  add_2 = None
    getitem_2: "f32[1, 128, 768]" = native_dropout[0]
    getitem_3: "b8[1, 128, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view: "f32[128, 768]" = torch.ops.aten.view.default(getitem_2, [128, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_6, view, permute);  primals_6 = None
    view_1: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm, [1, 128, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_2: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_1, [1, -1, 12, 64]);  view_1 = None
    permute_1: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_3: "f32[128, 768]" = torch.ops.aten.view.default(getitem_2, [128, 768])
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_1: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_8, view_3, permute_2);  primals_8 = None
    view_4: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_1, [1, 128, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_5: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_4, [1, -1, 12, 64]);  view_4 = None
    permute_3: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_6: "f32[128, 768]" = torch.ops.aten.view.default(getitem_2, [128, 768])
    permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_2: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_10, view_6, permute_4);  primals_10 = None
    view_7: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_2, [1, 128, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_8: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_7, [1, -1, 12, 64]);  view_7 = None
    permute_5: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_1, 8.0);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_6: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    expand: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div, [1, 12, 128, 64]);  div = None
    view_9: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand, [12, 128, 64]);  expand = None
    expand_1: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 128]);  permute_6 = None
    view_10: "f32[12, 64, 128]" = torch.ops.aten.view.default(expand_1, [12, 64, 128]);  expand_1 = None
    bmm: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm, [1, 12, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_12: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq, [1, 1, 1, 128]);  eq = None
    expand_2: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_12, [1, 12, 128, 128]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, lift_fresh_copy, view_11);  lift_fresh_copy = view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_1, 0.1, True);  div_1 = None
    getitem_4: "f32[1, 12, 128, 128]" = native_dropout_1[0]
    getitem_5: "b8[1, 12, 128, 128]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_3: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_4, [1, 12, 128, 128]);  getitem_4 = None
    view_13: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_3, [12, 128, 128]);  expand_3 = None
    expand_4: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 128, 64]);  permute_5 = None
    view_14: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_4, [12, 128, 64]);  expand_4 = None
    bmm_1: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_13, view_14)
    view_15: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_7: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
    clone: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_16: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone, [1, -1, 768]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_17: "f32[128, 768]" = torch.ops.aten.view.default(view_16, [128, 768]);  view_16 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_3: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_12, view_17, permute_8);  primals_12 = None
    view_18: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_3, [1, 128, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_3: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_18, getitem_2);  view_18 = getitem_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_7)
    mul_2: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_3: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_13);  mul_2 = None
    add_5: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_14);  mul_3 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_19: "f32[128, 768]" = torch.ops.aten.view.default(add_5, [128, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_4: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_16, view_19, permute_9);  primals_16 = None
    view_20: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 128, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_4: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, 0.5)
    mul_5: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476)
    erf: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_21: "f32[128, 3072]" = torch.ops.aten.view.default(mul_6, [128, 3072]);  mul_6 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_5: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_18, view_21, permute_10);  primals_18 = None
    view_22: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_5, [1, 128, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_22, 0.1, True);  view_22 = None
    getitem_8: "f32[1, 128, 768]" = native_dropout_2[0]
    getitem_9: "b8[1, 128, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_7: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_8, add_5);  getitem_8 = add_5 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_11)
    mul_7: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_8: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_19);  mul_7 = None
    add_9: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_8, primals_20);  mul_8 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_23: "f32[128, 768]" = torch.ops.aten.view.default(add_9, [128, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_6: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_22, view_23, permute_11);  primals_22 = None
    view_24: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_6, [1, 128, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_25: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_24, [1, -1, 12, 64]);  view_24 = None
    permute_12: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_26: "f32[128, 768]" = torch.ops.aten.view.default(add_9, [128, 768])
    permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_7: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_24, view_26, permute_13);  primals_24 = None
    view_27: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_7, [1, 128, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_28: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_27, [1, -1, 12, 64]);  view_27 = None
    permute_14: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_29: "f32[128, 768]" = torch.ops.aten.view.default(add_9, [128, 768])
    permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_8: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_26, view_29, permute_15);  primals_26 = None
    view_30: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_8, [1, 128, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_31: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_30, [1, -1, 12, 64]);  view_30 = None
    permute_16: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_2: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_12, 8.0);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_17: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
    expand_5: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_2, [1, 12, 128, 64]);  div_2 = None
    view_32: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_5, [12, 128, 64]);  expand_5 = None
    expand_6: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 128]);  permute_17 = None
    view_33: "f32[12, 64, 128]" = torch.ops.aten.view.default(expand_6, [12, 64, 128]);  expand_6 = None
    bmm_2: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_32, view_33)
    view_34: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_2, [1, 12, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_1: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_35: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_1, [1, 1, 1, 128]);  eq_1 = None
    expand_7: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_35, [1, 12, 128, 128]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant1 = self._tensor_constant1
    lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_1: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_7, lift_fresh_copy_1, view_34);  lift_fresh_copy_1 = view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_1: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    native_dropout_3 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_12: "f32[1, 12, 128, 128]" = native_dropout_3[0]
    getitem_13: "b8[1, 12, 128, 128]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_8: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_12, [1, 12, 128, 128]);  getitem_12 = None
    view_36: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_8, [12, 128, 128]);  expand_8 = None
    expand_9: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_16, [1, 12, 128, 64]);  permute_16 = None
    view_37: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_9, [12, 128, 64]);  expand_9 = None
    bmm_3: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_18: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_1: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_39: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_1, [1, -1, 768]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_40: "f32[128, 768]" = torch.ops.aten.view.default(view_39, [128, 768]);  view_39 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_9: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_28, view_40, permute_19);  primals_28 = None
    view_41: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_9, [1, 128, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_10: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_41, add_9);  view_41 = add_9 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_5: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_15)
    mul_9: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_10: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_29);  mul_9 = None
    add_12: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_30);  mul_10 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_42: "f32[128, 768]" = torch.ops.aten.view.default(add_12, [128, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_10: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_32, view_42, permute_20);  primals_32 = None
    view_43: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 128, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_11: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_12: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_1: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_44: "f32[128, 3072]" = torch.ops.aten.view.default(mul_13, [128, 3072]);  mul_13 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    addmm_11: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_34, view_44, permute_21);  primals_34 = None
    view_45: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_11, [1, 128, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    native_dropout_4 = torch.ops.aten.native_dropout.default(view_45, 0.1, True);  view_45 = None
    getitem_16: "f32[1, 128, 768]" = native_dropout_4[0]
    getitem_17: "b8[1, 128, 768]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_14: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_16, add_12);  getitem_16 = add_12 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_19)
    mul_14: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_15: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_35);  mul_14 = None
    add_16: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_36);  mul_15 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_46: "f32[128, 768]" = torch.ops.aten.view.default(add_16, [128, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_12: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_38, view_46, permute_22);  primals_38 = None
    view_47: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_12, [1, 128, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_48: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_47, [1, -1, 12, 64]);  view_47 = None
    permute_23: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_49: "f32[128, 768]" = torch.ops.aten.view.default(add_16, [128, 768])
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    addmm_13: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_40, view_49, permute_24);  primals_40 = None
    view_50: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_13, [1, 128, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_51: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_50, [1, -1, 12, 64]);  view_50 = None
    permute_25: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_52: "f32[128, 768]" = torch.ops.aten.view.default(add_16, [128, 768])
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    addmm_14: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_42, view_52, permute_26);  primals_42 = None
    view_53: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_14, [1, 128, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_54: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_53, [1, -1, 12, 64]);  view_53 = None
    permute_27: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_4: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_23, 8.0);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_28: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
    expand_10: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_4, [1, 12, 128, 64]);  div_4 = None
    view_55: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_10, [12, 128, 64]);  expand_10 = None
    expand_11: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 128]);  permute_28 = None
    view_56: "f32[12, 64, 128]" = torch.ops.aten.view.default(expand_11, [12, 64, 128]);  expand_11 = None
    bmm_4: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56)
    view_57: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_4, [1, 12, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_2: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_58: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_2, [1, 1, 1, 128]);  eq_2 = None
    expand_12: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_58, [1, 12, 128, 128]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant2 = self._tensor_constant2
    lift_fresh_copy_2: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_2: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_12, lift_fresh_copy_2, view_57);  lift_fresh_copy_2 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_2: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    native_dropout_5 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_20: "f32[1, 12, 128, 128]" = native_dropout_5[0]
    getitem_21: "b8[1, 12, 128, 128]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_13: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_20, [1, 12, 128, 128]);  getitem_20 = None
    view_59: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_13, [12, 128, 128]);  expand_13 = None
    expand_14: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 128, 64]);  permute_27 = None
    view_60: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_14, [12, 128, 64]);  expand_14 = None
    bmm_5: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_59, view_60)
    view_61: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_29: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
    clone_2: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_62: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_2, [1, -1, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_63: "f32[128, 768]" = torch.ops.aten.view.default(view_62, [128, 768]);  view_62 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_15: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_44, view_63, permute_30);  primals_44 = None
    view_64: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_15, [1, 128, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_17: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_64, add_16);  view_64 = add_16 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_8: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_23)
    mul_16: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_17: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_45);  mul_16 = None
    add_19: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_46);  mul_17 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_65: "f32[128, 768]" = torch.ops.aten.view.default(add_19, [128, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_16: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_48, view_65, permute_31);  primals_48 = None
    view_66: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 128, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_18: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
    mul_19: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476)
    erf_2: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_67: "f32[128, 3072]" = torch.ops.aten.view.default(mul_20, [128, 3072]);  mul_20 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_17: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_50, view_67, permute_32);  primals_50 = None
    view_68: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_17, [1, 128, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_68, 0.1, True);  view_68 = None
    getitem_24: "f32[1, 128, 768]" = native_dropout_6[0]
    getitem_25: "b8[1, 128, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_21: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_24, add_19);  getitem_24 = add_19 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_22: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_9: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_27)
    mul_21: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_22: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_51);  mul_21 = None
    add_23: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_52);  mul_22 = primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_69: "f32[128, 768]" = torch.ops.aten.view.default(add_23, [128, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_18: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_54, view_69, permute_33);  primals_54 = None
    view_70: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_18, [1, 128, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_71: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_70, [1, -1, 12, 64]);  view_70 = None
    permute_34: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_72: "f32[128, 768]" = torch.ops.aten.view.default(add_23, [128, 768])
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_19: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_56, view_72, permute_35);  primals_56 = None
    view_73: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_19, [1, 128, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_74: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_73, [1, -1, 12, 64]);  view_73 = None
    permute_36: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_75: "f32[128, 768]" = torch.ops.aten.view.default(add_23, [128, 768])
    permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_20: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_58, view_75, permute_37);  primals_58 = None
    view_76: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_20, [1, 128, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_77: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_76, [1, -1, 12, 64]);  view_76 = None
    permute_38: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_6: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_34, 8.0);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_39: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
    expand_15: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_6, [1, 12, 128, 64]);  div_6 = None
    view_78: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_15, [12, 128, 64]);  expand_15 = None
    expand_16: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_39, [1, 12, 64, 128]);  permute_39 = None
    view_79: "f32[12, 64, 128]" = torch.ops.aten.view.default(expand_16, [12, 64, 128]);  expand_16 = None
    bmm_6: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_6, [1, 12, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_3: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_81: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_3, [1, 1, 1, 128]);  eq_3 = None
    expand_17: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_81, [1, 12, 128, 128]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant3 = self._tensor_constant3
    lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_3: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_17, lift_fresh_copy_3, view_80);  lift_fresh_copy_3 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_3: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    native_dropout_7 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_28: "f32[1, 12, 128, 128]" = native_dropout_7[0]
    getitem_29: "b8[1, 12, 128, 128]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_18: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_28, [1, 12, 128, 128]);  getitem_28 = None
    view_82: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_18, [12, 128, 128]);  expand_18 = None
    expand_19: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_38, [1, 12, 128, 64]);  permute_38 = None
    view_83: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_19, [12, 128, 64]);  expand_19 = None
    bmm_7: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_40: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    clone_3: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_85: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_3, [1, -1, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_86: "f32[128, 768]" = torch.ops.aten.view.default(view_85, [128, 768]);  view_85 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_21: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_60, view_86, permute_41);  primals_60 = None
    view_87: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_21, [1, 128, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_24: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_87, add_23);  view_87 = add_23 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_11: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_31)
    mul_23: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_24: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_61);  mul_23 = None
    add_26: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_62);  mul_24 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_88: "f32[128, 768]" = torch.ops.aten.view.default(add_26, [128, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    addmm_22: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_64, view_88, permute_42);  primals_64 = None
    view_89: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 128, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_25: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    mul_26: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.7071067811865476)
    erf_3: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_27: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_27: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_90: "f32[128, 3072]" = torch.ops.aten.view.default(mul_27, [128, 3072]);  mul_27 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    addmm_23: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_66, view_90, permute_43);  primals_66 = None
    view_91: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_23, [1, 128, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_91, 0.1, True);  view_91 = None
    getitem_32: "f32[1, 128, 768]" = native_dropout_8[0]
    getitem_33: "b8[1, 128, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_28: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_32, add_26);  getitem_32 = add_26 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    add_29: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_12: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_35)
    mul_28: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_29: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_67);  mul_28 = None
    add_30: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_68);  mul_29 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_92: "f32[128, 768]" = torch.ops.aten.view.default(add_30, [128, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    addmm_24: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_70, view_92, permute_44);  primals_70 = None
    view_93: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_24, [1, 128, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_94: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_93, [1, -1, 12, 64]);  view_93 = None
    permute_45: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_95: "f32[128, 768]" = torch.ops.aten.view.default(add_30, [128, 768])
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_25: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_72, view_95, permute_46);  primals_72 = None
    view_96: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_25, [1, 128, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_97: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_96, [1, -1, 12, 64]);  view_96 = None
    permute_47: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_98: "f32[128, 768]" = torch.ops.aten.view.default(add_30, [128, 768])
    permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_26: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_74, view_98, permute_48);  primals_74 = None
    view_99: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_26, [1, 128, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_100: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_99, [1, -1, 12, 64]);  view_99 = None
    permute_49: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_8: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_45, 8.0);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_50: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
    expand_20: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_8, [1, 12, 128, 64]);  div_8 = None
    view_101: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_20, [12, 128, 64]);  expand_20 = None
    expand_21: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_50, [1, 12, 64, 128]);  permute_50 = None
    view_102: "f32[12, 64, 128]" = torch.ops.aten.view.default(expand_21, [12, 64, 128]);  expand_21 = None
    bmm_8: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_101, view_102)
    view_103: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_8, [1, 12, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_4: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_104: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_4, [1, 1, 1, 128]);  eq_4 = None
    expand_22: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_104, [1, 12, 128, 128]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant4 = self._tensor_constant4
    lift_fresh_copy_4: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_4: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_22, lift_fresh_copy_4, view_103);  lift_fresh_copy_4 = view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_4: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    native_dropout_9 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_36: "f32[1, 12, 128, 128]" = native_dropout_9[0]
    getitem_37: "b8[1, 12, 128, 128]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_23: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_36, [1, 12, 128, 128]);  getitem_36 = None
    view_105: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_23, [12, 128, 128]);  expand_23 = None
    expand_24: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_49, [1, 12, 128, 64]);  permute_49 = None
    view_106: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_24, [12, 128, 64]);  expand_24 = None
    bmm_9: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_105, view_106)
    view_107: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_51: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    clone_4: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_108: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_4, [1, -1, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_109: "f32[128, 768]" = torch.ops.aten.view.default(view_108, [128, 768]);  view_108 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_27: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_76, view_109, permute_52);  primals_76 = None
    view_110: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_27, [1, 128, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_31: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_110, add_30);  view_110 = add_30 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_14: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_39)
    mul_30: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_31: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_77);  mul_30 = None
    add_33: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_78);  mul_31 = primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_111: "f32[128, 768]" = torch.ops.aten.view.default(add_33, [128, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_28: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_80, view_111, permute_53);  primals_80 = None
    view_112: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 128, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_32: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.5)
    mul_33: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.7071067811865476)
    erf_4: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_34: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_34: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_113: "f32[128, 3072]" = torch.ops.aten.view.default(mul_34, [128, 3072]);  mul_34 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_29: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_82, view_113, permute_54);  primals_82 = None
    view_114: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_29, [1, 128, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    native_dropout_10 = torch.ops.aten.native_dropout.default(view_114, 0.1, True);  view_114 = None
    getitem_40: "f32[1, 128, 768]" = native_dropout_10[0]
    getitem_41: "b8[1, 128, 768]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_35: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_40, add_33);  getitem_40 = add_33 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_15: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_43)
    mul_35: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_36: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_83);  mul_35 = None
    add_37: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_84);  mul_36 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_115: "f32[128, 768]" = torch.ops.aten.view.default(add_37, [128, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_30: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_86, view_115, permute_55);  primals_86 = None
    view_116: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_30, [1, 128, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_117: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_116, [1, -1, 12, 64]);  view_116 = None
    permute_56: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_118: "f32[128, 768]" = torch.ops.aten.view.default(add_37, [128, 768])
    permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_31: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_88, view_118, permute_57);  primals_88 = None
    view_119: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_31, [1, 128, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_120: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_119, [1, -1, 12, 64]);  view_119 = None
    permute_58: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_121: "f32[128, 768]" = torch.ops.aten.view.default(add_37, [128, 768])
    permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_32: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_90, view_121, permute_59);  primals_90 = None
    view_122: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_32, [1, 128, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_123: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_122, [1, -1, 12, 64]);  view_122 = None
    permute_60: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_10: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_56, 8.0);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_61: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
    expand_25: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_10, [1, 12, 128, 64]);  div_10 = None
    view_124: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_25, [12, 128, 64]);  expand_25 = None
    expand_26: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_61, [1, 12, 64, 128]);  permute_61 = None
    view_125: "f32[12, 64, 128]" = torch.ops.aten.view.default(expand_26, [12, 64, 128]);  expand_26 = None
    bmm_10: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_124, view_125)
    view_126: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_10, [1, 12, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_5: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0);  full = None
    view_127: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_5, [1, 1, 1, 128]);  eq_5 = None
    expand_27: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_127, [1, 12, 128, 128]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant5 = self._tensor_constant5
    lift_fresh_copy_5: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_5: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_27, lift_fresh_copy_5, view_126);  lift_fresh_copy_5 = view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_5: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    native_dropout_11 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_44: "f32[1, 12, 128, 128]" = native_dropout_11[0]
    getitem_45: "b8[1, 12, 128, 128]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_28: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(getitem_44, [1, 12, 128, 128]);  getitem_44 = None
    view_128: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_28, [12, 128, 128]);  expand_28 = None
    expand_29: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_60, [1, 12, 128, 64]);  permute_60 = None
    view_129: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_29, [12, 128, 64]);  expand_29 = None
    bmm_11: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_128, view_129)
    view_130: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_62: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    clone_5: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_131: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_5, [1, -1, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_132: "f32[128, 768]" = torch.ops.aten.view.default(view_131, [128, 768]);  view_131 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_33: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_92, view_132, permute_63);  primals_92 = None
    view_133: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_33, [1, 128, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_38: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_133, add_37);  view_133 = add_37 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_17: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_47)
    mul_37: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_38: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_93);  mul_37 = None
    add_40: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_94);  mul_38 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_134: "f32[128, 768]" = torch.ops.aten.view.default(add_40, [128, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_34: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_96, view_134, permute_64);  primals_96 = None
    view_135: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 128, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_39: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, 0.5)
    mul_40: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476)
    erf_5: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_41: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_41: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_136: "f32[128, 3072]" = torch.ops.aten.view.default(mul_41, [128, 3072]);  mul_41 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_35: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_98, view_136, permute_65);  primals_98 = None
    view_137: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_35, [1, 128, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_137, 0.1, True);  view_137 = None
    getitem_48: "f32[1, 128, 768]" = native_dropout_12[0]
    getitem_49: "b8[1, 128, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_42: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(getitem_48, add_40);  getitem_48 = add_40 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_43: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_18: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_51)
    mul_42: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_43: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_99);  mul_42 = None
    add_44: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_100);  mul_43 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:702, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
    view_138: "f32[128, 768]" = torch.ops.aten.view.default(add_44, [128, 768]);  add_44 = None
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_36: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_102, view_138, permute_66);  primals_102 = None
    view_139: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_36, [1, 128, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_44: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, 0.5)
    mul_45: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, 0.7071067811865476)
    erf_6: "f32[1, 128, 768]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
    add_45: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_46: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_44, add_45);  mul_44 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:704, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
    var_mean_13 = torch.ops.aten.var_mean.correction(mul_46, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_19: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_46, getitem_53)
    mul_47: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = None
    mul_48: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_47, primals_103);  mul_47 = None
    add_47: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_48, primals_104);  mul_48 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:705, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
    view_140: "f32[128, 768]" = torch.ops.aten.view.default(add_47, [128, 768]);  add_47 = None
    permute_67: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_37: "f32[128, 30522]" = torch.ops.aten.addmm.default(primals_106, view_140, permute_67);  primals_106 = None
    view_141: "f32[1, 128, 30522]" = torch.ops.aten.view.default(addmm_37, [1, 128, 30522]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:709, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
    view_142: "f32[128, 30522]" = torch.ops.aten.view.default(view_141, [-1, 30522])
    view_143: "i64[128]" = torch.ops.aten.view.default(primals_109, [-1]);  primals_109 = None
    amax_6: "f32[128, 1]" = torch.ops.aten.amax.default(view_142, [1], True)
    sub_20: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(view_142, amax_6);  view_142 = amax_6 = None
    exp_6: "f32[128, 30522]" = torch.ops.aten.exp.default(sub_20)
    sum_7: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
    sub_21: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(sub_20, log);  sub_20 = log = None
    alias_6: "f32[128, 30522]" = torch.ops.aten.alias.default(sub_21)
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_143, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "i64[128]" = torch.ops.aten.where.self(ne, view_143, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_21, 1, unsqueeze);  sub_21 = unsqueeze = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_143, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[128]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_143, -100)
    sum_8: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
    sum_9: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
    div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type);  sum_9 = None
    div_13: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_1: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(view_143, 1);  view_143 = None
    ne_3: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "i64[128, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full_1: "f32[128, 30522]" = torch.ops.aten.full.default([128, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[128, 30522]" = torch.ops.aten.scatter.value(full_1, 1, where_8, -1.0);  full_1 = where_8 = None
    ne_4: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100);  unsqueeze_1 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[128, 1]" = torch.ops.aten.where.self(ne_4, div_13, scalar_tensor_3);  ne_4 = div_13 = scalar_tensor_3 = None
    mul_49: "f32[128, 30522]" = torch.ops.aten.mul.Tensor(scatter, where_9);  scatter = where_9 = None
    alias_7: "f32[128, 30522]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    exp_7: "f32[128, 30522]" = torch.ops.aten.exp.default(alias_7);  alias_7 = None
    sum_10: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_49, [1], True)
    mul_50: "f32[128, 30522]" = torch.ops.aten.mul.Tensor(exp_7, sum_10);  exp_7 = sum_10 = None
    sub_22: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
    view_144: "f32[1, 128, 30522]" = torch.ops.aten.view.default(sub_22, [1, 128, 30522]);  sub_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:709, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
    add_48: "f32[1, 128, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_144);  tangents_2 = view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:705, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
    view_145: "f32[128, 30522]" = torch.ops.aten.view.default(add_48, [128, 30522]);  add_48 = None
    permute_68: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm: "f32[128, 768]" = torch.ops.aten.mm.default(view_145, permute_68);  permute_68 = None
    permute_69: "f32[30522, 128]" = torch.ops.aten.permute.default(view_145, [1, 0])
    mm_1: "f32[30522, 768]" = torch.ops.aten.mm.default(permute_69, view_140);  permute_69 = view_140 = None
    permute_70: "f32[768, 30522]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_11: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_145, [0], True);  view_145 = None
    view_146: "f32[30522]" = torch.ops.aten.view.default(sum_11, [30522]);  sum_11 = None
    permute_71: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    view_147: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm, [1, 128, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:704, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
    sub_23: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_46, getitem_53);  mul_46 = getitem_53 = None
    mul_51: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_13);  sub_23 = None
    mul_52: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_147, primals_103);  primals_103 = None
    mul_53: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_52, 768)
    sum_12: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_52, [2], True)
    mul_54: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_52, mul_51);  mul_52 = None
    sum_13: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_54, [2], True);  mul_54 = None
    mul_55: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_51, sum_13);  sum_13 = None
    sub_24: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_53, sum_12);  mul_53 = sum_12 = None
    sub_25: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_24, mul_55);  sub_24 = mul_55 = None
    div_14: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_56: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_25);  div_14 = sub_25 = None
    mul_57: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_147, mul_51);  mul_51 = None
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_57, [0, 1]);  mul_57 = None
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_147, [0, 1]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_58: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, 0.7071067811865476)
    erf_7: "f32[1, 128, 768]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_49: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_59: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
    mul_60: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, view_139)
    mul_61: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_60, -0.5);  mul_60 = None
    exp_8: "f32[1, 128, 768]" = torch.ops.aten.exp.default(mul_61);  mul_61 = None
    mul_62: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_63: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, mul_62);  view_139 = mul_62 = None
    add_50: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_59, mul_63);  mul_59 = mul_63 = None
    mul_64: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_56, add_50);  mul_56 = add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:702, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
    view_148: "f32[128, 768]" = torch.ops.aten.view.default(mul_64, [128, 768]);  mul_64 = None
    permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_2: "f32[128, 768]" = torch.ops.aten.mm.default(view_148, permute_72);  permute_72 = None
    permute_73: "f32[768, 128]" = torch.ops.aten.permute.default(view_148, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_73, view_138);  permute_73 = view_138 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_148, [0], True);  view_148 = None
    view_149: "f32[768]" = torch.ops.aten.view.default(sum_16, [768]);  sum_16 = None
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    view_150: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_2, [1, 128, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    sub_26: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_51);  add_42 = getitem_51 = None
    mul_65: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_12);  sub_26 = None
    mul_66: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_150, primals_99);  primals_99 = None
    mul_67: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_66, 768)
    sum_17: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_66, [2], True)
    mul_68: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_66, mul_65);  mul_66 = None
    sum_18: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_68, [2], True);  mul_68 = None
    mul_69: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_18);  sum_18 = None
    sub_27: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_67, sum_17);  mul_67 = sum_17 = None
    sub_28: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_27, mul_69);  sub_27 = mul_69 = None
    div_15: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_70: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_28);  div_15 = sub_28 = None
    mul_71: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_150, mul_65);  mul_65 = None
    sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_71, [0, 1]);  mul_71 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_150, [0, 1]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_1: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_72: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_73: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_70, mul_72);  mul_72 = None
    clone_6: "f32[1, 128, 768]" = torch.ops.aten.clone.default(mul_73, memory_format = torch.contiguous_format);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_151: "f32[128, 768]" = torch.ops.aten.view.default(clone_6, [128, 768]);  clone_6 = None
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_4: "f32[128, 3072]" = torch.ops.aten.mm.default(view_151, permute_76);  permute_76 = None
    permute_77: "f32[768, 128]" = torch.ops.aten.permute.default(view_151, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_77, view_136);  permute_77 = view_136 = None
    permute_78: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_151, [0], True);  view_151 = None
    view_152: "f32[768]" = torch.ops.aten.view.default(sum_21, [768]);  sum_21 = None
    permute_79: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    view_153: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_4, [1, 128, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_74: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476)
    erf_8: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_51: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_75: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_51, 0.5);  add_51 = None
    mul_76: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, view_135)
    mul_77: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_76, -0.5);  mul_76 = None
    exp_9: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_77);  mul_77 = None
    mul_78: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_79: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, mul_78);  view_135 = mul_78 = None
    add_52: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_75, mul_79);  mul_75 = mul_79 = None
    mul_80: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_153, add_52);  view_153 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_154: "f32[128, 3072]" = torch.ops.aten.view.default(mul_80, [128, 3072]);  mul_80 = None
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_6: "f32[128, 768]" = torch.ops.aten.mm.default(view_154, permute_80);  permute_80 = None
    permute_81: "f32[3072, 128]" = torch.ops.aten.permute.default(view_154, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_81, view_134);  permute_81 = view_134 = None
    permute_82: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_22: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
    view_155: "f32[3072]" = torch.ops.aten.view.default(sum_22, [3072]);  sum_22 = None
    permute_83: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    view_156: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_6, [1, 128, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_53: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_70, view_156);  mul_70 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    sub_29: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_47);  add_38 = getitem_47 = None
    mul_81: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_11);  sub_29 = None
    mul_82: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_53, primals_93);  primals_93 = None
    mul_83: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_82, 768)
    sum_23: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_82, [2], True)
    mul_84: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_82, mul_81);  mul_82 = None
    sum_24: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_84, [2], True);  mul_84 = None
    mul_85: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_81, sum_24);  sum_24 = None
    sub_30: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_83, sum_23);  mul_83 = sum_23 = None
    sub_31: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_30, mul_85);  sub_30 = mul_85 = None
    div_16: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_86: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_31);  div_16 = sub_31 = None
    mul_87: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_53, mul_81);  mul_81 = None
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_87, [0, 1]);  mul_87 = None
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_53, [0, 1]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_157: "f32[128, 768]" = torch.ops.aten.view.default(mul_86, [128, 768])
    permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_8: "f32[128, 768]" = torch.ops.aten.mm.default(view_157, permute_84);  permute_84 = None
    permute_85: "f32[768, 128]" = torch.ops.aten.permute.default(view_157, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_85, view_132);  permute_85 = view_132 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_157, [0], True);  view_157 = None
    view_158: "f32[768]" = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
    permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    view_159: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_8, [1, 128, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_160: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_159, [1, 128, 12, 64]);  view_159 = None
    permute_88: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_161: "f32[12, 128, 64]" = torch.ops.aten.view.default(permute_88, [12, 128, 64]);  permute_88 = None
    permute_89: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
    bmm_12: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_89, view_161);  permute_89 = None
    permute_90: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
    bmm_13: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_161, permute_90);  view_161 = permute_90 = None
    view_162: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_12, [1, 12, 128, 64]);  bmm_12 = None
    view_163: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_13, [1, 12, 128, 128]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_2: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_88: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_89: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_163, mul_88);  view_163 = mul_88 = None
    clone_7: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(mul_89, memory_format = torch.contiguous_format);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_8: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_90: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(clone_7, alias_8);  clone_7 = None
    sum_28: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_90, [-1], True)
    mul_91: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_8, sum_28);  alias_8 = sum_28 = None
    sub_32: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_27, scalar_tensor_4, sub_32);  expand_27 = scalar_tensor_4 = sub_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_164: "f32[12, 128, 128]" = torch.ops.aten.view.default(where_10, [12, 128, 128]);  where_10 = None
    permute_91: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    bmm_14: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_91, view_164);  permute_91 = None
    permute_92: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    bmm_15: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_164, permute_92);  view_164 = permute_92 = None
    view_165: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_14, [1, 12, 64, 128]);  bmm_14 = None
    view_166: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_15, [1, 12, 128, 64]);  bmm_15 = None
    permute_93: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_165, [0, 1, 3, 2]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_17: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_166, 8.0);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_94: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_8: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    view_167: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_8, [1, 128, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_168: "f32[128, 768]" = torch.ops.aten.view.default(view_167, [128, 768]);  view_167 = None
    permute_95: "f32[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_10: "f32[128, 768]" = torch.ops.aten.mm.default(view_168, permute_95);  permute_95 = None
    permute_96: "f32[768, 128]" = torch.ops.aten.permute.default(view_168, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_96, view_121);  permute_96 = view_121 = None
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
    view_169: "f32[768]" = torch.ops.aten.view.default(sum_29, [768]);  sum_29 = None
    permute_98: "f32[768, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    view_170: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_10, [1, 128, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_54: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_86, view_170);  mul_86 = view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_99: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_93, [0, 2, 1, 3]);  permute_93 = None
    view_171: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_99, [1, 128, 768]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_172: "f32[128, 768]" = torch.ops.aten.view.default(view_171, [128, 768]);  view_171 = None
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_12: "f32[128, 768]" = torch.ops.aten.mm.default(view_172, permute_100);  permute_100 = None
    permute_101: "f32[768, 128]" = torch.ops.aten.permute.default(view_172, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_101, view_118);  permute_101 = view_118 = None
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True);  view_172 = None
    view_173: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    view_174: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_12, [1, 128, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_55: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_54, view_174);  add_54 = view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_104: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_17, [0, 2, 1, 3]);  div_17 = None
    clone_9: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_175: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_9, [1, 128, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_176: "f32[128, 768]" = torch.ops.aten.view.default(view_175, [128, 768]);  view_175 = None
    permute_105: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_14: "f32[128, 768]" = torch.ops.aten.mm.default(view_176, permute_105);  permute_105 = None
    permute_106: "f32[768, 128]" = torch.ops.aten.permute.default(view_176, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_106, view_115);  permute_106 = view_115 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_176, [0], True);  view_176 = None
    view_177: "f32[768]" = torch.ops.aten.view.default(sum_31, [768]);  sum_31 = None
    permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    view_178: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_14, [1, 128, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_56: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_55, view_178);  add_55 = view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    sub_33: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_43);  add_35 = getitem_43 = None
    mul_92: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_10);  sub_33 = None
    mul_93: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_56, primals_83);  primals_83 = None
    mul_94: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_93, 768)
    sum_32: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_93, [2], True)
    mul_95: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_93, mul_92);  mul_93 = None
    sum_33: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True);  mul_95 = None
    mul_96: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_92, sum_33);  sum_33 = None
    sub_34: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_94, sum_32);  mul_94 = sum_32 = None
    sub_35: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_34, mul_96);  sub_34 = mul_96 = None
    div_18: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_97: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_35);  div_18 = sub_35 = None
    mul_98: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_56, mul_92);  mul_92 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_98, [0, 1]);  mul_98 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_56, [0, 1]);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_3: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_99: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_100: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_97, mul_99);  mul_99 = None
    clone_10: "f32[1, 128, 768]" = torch.ops.aten.clone.default(mul_100, memory_format = torch.contiguous_format);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_179: "f32[128, 768]" = torch.ops.aten.view.default(clone_10, [128, 768]);  clone_10 = None
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_16: "f32[128, 3072]" = torch.ops.aten.mm.default(view_179, permute_109);  permute_109 = None
    permute_110: "f32[768, 128]" = torch.ops.aten.permute.default(view_179, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_110, view_113);  permute_110 = view_113 = None
    permute_111: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True);  view_179 = None
    view_180: "f32[768]" = torch.ops.aten.view.default(sum_36, [768]);  sum_36 = None
    permute_112: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    view_181: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_16, [1, 128, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_101: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.7071067811865476)
    erf_9: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
    add_57: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_102: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_57, 0.5);  add_57 = None
    mul_103: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, view_112)
    mul_104: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_103, -0.5);  mul_103 = None
    exp_10: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_104);  mul_104 = None
    mul_105: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_106: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, mul_105);  view_112 = mul_105 = None
    add_58: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_102, mul_106);  mul_102 = mul_106 = None
    mul_107: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_181, add_58);  view_181 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_182: "f32[128, 3072]" = torch.ops.aten.view.default(mul_107, [128, 3072]);  mul_107 = None
    permute_113: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_18: "f32[128, 768]" = torch.ops.aten.mm.default(view_182, permute_113);  permute_113 = None
    permute_114: "f32[3072, 128]" = torch.ops.aten.permute.default(view_182, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_114, view_111);  permute_114 = view_111 = None
    permute_115: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_37: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
    view_183: "f32[3072]" = torch.ops.aten.view.default(sum_37, [3072]);  sum_37 = None
    permute_116: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    view_184: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_18, [1, 128, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_59: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_97, view_184);  mul_97 = view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    sub_36: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_39);  add_31 = getitem_39 = None
    mul_108: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_9);  sub_36 = None
    mul_109: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_59, primals_77);  primals_77 = None
    mul_110: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_109, 768)
    sum_38: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True)
    mul_111: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_109, mul_108);  mul_109 = None
    sum_39: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True);  mul_111 = None
    mul_112: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_108, sum_39);  sum_39 = None
    sub_37: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_110, sum_38);  mul_110 = sum_38 = None
    sub_38: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_37, mul_112);  sub_37 = mul_112 = None
    div_19: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_113: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_38);  div_19 = sub_38 = None
    mul_114: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_59, mul_108);  mul_108 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_114, [0, 1]);  mul_114 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_59, [0, 1]);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_185: "f32[128, 768]" = torch.ops.aten.view.default(mul_113, [128, 768])
    permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_20: "f32[128, 768]" = torch.ops.aten.mm.default(view_185, permute_117);  permute_117 = None
    permute_118: "f32[768, 128]" = torch.ops.aten.permute.default(view_185, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_118, view_109);  permute_118 = view_109 = None
    permute_119: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
    view_186: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_120: "f32[768, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    view_187: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_20, [1, 128, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_188: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_187, [1, 128, 12, 64]);  view_187 = None
    permute_121: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_189: "f32[12, 128, 64]" = torch.ops.aten.view.default(permute_121, [12, 128, 64]);  permute_121 = None
    permute_122: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_105, [0, 2, 1]);  view_105 = None
    bmm_16: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_122, view_189);  permute_122 = None
    permute_123: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
    bmm_17: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_189, permute_123);  view_189 = permute_123 = None
    view_190: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_16, [1, 12, 128, 64]);  bmm_16 = None
    view_191: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_17, [1, 12, 128, 128]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_4: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_115: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_116: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_191, mul_115);  view_191 = mul_115 = None
    clone_11: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(mul_116, memory_format = torch.contiguous_format);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_9: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_117: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(clone_11, alias_9);  clone_11 = None
    sum_43: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [-1], True)
    mul_118: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_9, sum_43);  alias_9 = sum_43 = None
    sub_39: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_22, scalar_tensor_5, sub_39);  expand_22 = scalar_tensor_5 = sub_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_192: "f32[12, 128, 128]" = torch.ops.aten.view.default(where_11, [12, 128, 128]);  where_11 = None
    permute_124: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    bmm_18: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_124, view_192);  permute_124 = None
    permute_125: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    bmm_19: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_192, permute_125);  view_192 = permute_125 = None
    view_193: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_18, [1, 12, 64, 128]);  bmm_18 = None
    view_194: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_19, [1, 12, 128, 64]);  bmm_19 = None
    permute_126: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_193, [0, 1, 3, 2]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_20: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_194, 8.0);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_127: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_12: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_195: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_12, [1, 128, 768]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_196: "f32[128, 768]" = torch.ops.aten.view.default(view_195, [128, 768]);  view_195 = None
    permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_22: "f32[128, 768]" = torch.ops.aten.mm.default(view_196, permute_128);  permute_128 = None
    permute_129: "f32[768, 128]" = torch.ops.aten.permute.default(view_196, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_129, view_98);  permute_129 = view_98 = None
    permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_196, [0], True);  view_196 = None
    view_197: "f32[768]" = torch.ops.aten.view.default(sum_44, [768]);  sum_44 = None
    permute_131: "f32[768, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    view_198: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_22, [1, 128, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_60: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_113, view_198);  mul_113 = view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_132: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_126, [0, 2, 1, 3]);  permute_126 = None
    view_199: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_132, [1, 128, 768]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_200: "f32[128, 768]" = torch.ops.aten.view.default(view_199, [128, 768]);  view_199 = None
    permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_24: "f32[128, 768]" = torch.ops.aten.mm.default(view_200, permute_133);  permute_133 = None
    permute_134: "f32[768, 128]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_134, view_95);  permute_134 = view_95 = None
    permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
    view_201: "f32[768]" = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
    permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    view_202: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_24, [1, 128, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_61: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_60, view_202);  add_60 = view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_137: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_20, [0, 2, 1, 3]);  div_20 = None
    clone_13: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    view_203: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_13, [1, 128, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_204: "f32[128, 768]" = torch.ops.aten.view.default(view_203, [128, 768]);  view_203 = None
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_26: "f32[128, 768]" = torch.ops.aten.mm.default(view_204, permute_138);  permute_138 = None
    permute_139: "f32[768, 128]" = torch.ops.aten.permute.default(view_204, [1, 0])
    mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, view_92);  permute_139 = view_92 = None
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
    view_205: "f32[768]" = torch.ops.aten.view.default(sum_46, [768]);  sum_46 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    view_206: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_26, [1, 128, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_62: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_61, view_206);  add_61 = view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    sub_40: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_35);  add_28 = getitem_35 = None
    mul_119: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_8);  sub_40 = None
    mul_120: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_62, primals_67);  primals_67 = None
    mul_121: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_120, 768)
    sum_47: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_120, [2], True)
    mul_122: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_120, mul_119);  mul_120 = None
    sum_48: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True);  mul_122 = None
    mul_123: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_119, sum_48);  sum_48 = None
    sub_41: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_121, sum_47);  mul_121 = sum_47 = None
    sub_42: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_123);  sub_41 = mul_123 = None
    div_21: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_124: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_42);  div_21 = sub_42 = None
    mul_125: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_62, mul_119);  mul_119 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 1]);  mul_125 = None
    sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_62, [0, 1]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_5: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_33, torch.float32);  getitem_33 = None
    mul_126: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_127: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_124, mul_126);  mul_126 = None
    clone_14: "f32[1, 128, 768]" = torch.ops.aten.clone.default(mul_127, memory_format = torch.contiguous_format);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_207: "f32[128, 768]" = torch.ops.aten.view.default(clone_14, [128, 768]);  clone_14 = None
    permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_28: "f32[128, 3072]" = torch.ops.aten.mm.default(view_207, permute_142);  permute_142 = None
    permute_143: "f32[768, 128]" = torch.ops.aten.permute.default(view_207, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_143, view_90);  permute_143 = view_90 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
    view_208: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
    permute_145: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_209: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_28, [1, 128, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_128: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.7071067811865476)
    erf_10: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_63: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_129: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
    mul_130: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, view_89)
    mul_131: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_130, -0.5);  mul_130 = None
    exp_11: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_131);  mul_131 = None
    mul_132: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_133: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, mul_132);  view_89 = mul_132 = None
    add_64: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_129, mul_133);  mul_129 = mul_133 = None
    mul_134: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_209, add_64);  view_209 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_210: "f32[128, 3072]" = torch.ops.aten.view.default(mul_134, [128, 3072]);  mul_134 = None
    permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_30: "f32[128, 768]" = torch.ops.aten.mm.default(view_210, permute_146);  permute_146 = None
    permute_147: "f32[3072, 128]" = torch.ops.aten.permute.default(view_210, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_147, view_88);  permute_147 = view_88 = None
    permute_148: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_52: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
    view_211: "f32[3072]" = torch.ops.aten.view.default(sum_52, [3072]);  sum_52 = None
    permute_149: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_212: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_30, [1, 128, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_65: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_124, view_212);  mul_124 = view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    sub_43: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_31);  add_24 = getitem_31 = None
    mul_135: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_7);  sub_43 = None
    mul_136: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_65, primals_61);  primals_61 = None
    mul_137: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_136, 768)
    sum_53: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_136, [2], True)
    mul_138: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_136, mul_135);  mul_136 = None
    sum_54: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True);  mul_138 = None
    mul_139: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_135, sum_54);  sum_54 = None
    sub_44: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_137, sum_53);  mul_137 = sum_53 = None
    sub_45: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_139);  sub_44 = mul_139 = None
    div_22: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_140: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_45);  div_22 = sub_45 = None
    mul_141: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_65, mul_135);  mul_135 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_141, [0, 1]);  mul_141 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_65, [0, 1]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_213: "f32[128, 768]" = torch.ops.aten.view.default(mul_140, [128, 768])
    permute_150: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_32: "f32[128, 768]" = torch.ops.aten.mm.default(view_213, permute_150);  permute_150 = None
    permute_151: "f32[768, 128]" = torch.ops.aten.permute.default(view_213, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_151, view_86);  permute_151 = view_86 = None
    permute_152: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
    view_214: "f32[768]" = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_215: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_32, [1, 128, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_216: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_215, [1, 128, 12, 64]);  view_215 = None
    permute_154: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_217: "f32[12, 128, 64]" = torch.ops.aten.view.default(permute_154, [12, 128, 64]);  permute_154 = None
    permute_155: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    bmm_20: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_155, view_217);  permute_155 = None
    permute_156: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    bmm_21: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_217, permute_156);  view_217 = permute_156 = None
    view_218: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_20, [1, 12, 128, 64]);  bmm_20 = None
    view_219: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_21, [1, 12, 128, 128]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_6: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_29, torch.float32);  getitem_29 = None
    mul_142: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_143: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_219, mul_142);  view_219 = mul_142 = None
    clone_15: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(mul_143, memory_format = torch.contiguous_format);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_10: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_144: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(clone_15, alias_10);  clone_15 = None
    sum_58: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [-1], True)
    mul_145: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_10, sum_58);  alias_10 = sum_58 = None
    sub_46: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_17, scalar_tensor_6, sub_46);  expand_17 = scalar_tensor_6 = sub_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_220: "f32[12, 128, 128]" = torch.ops.aten.view.default(where_12, [12, 128, 128]);  where_12 = None
    permute_157: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_22: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_157, view_220);  permute_157 = None
    permute_158: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_23: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_220, permute_158);  view_220 = permute_158 = None
    view_221: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_22, [1, 12, 64, 128]);  bmm_22 = None
    view_222: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_23, [1, 12, 128, 64]);  bmm_23 = None
    permute_159: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_221, [0, 1, 3, 2]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_23: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_222, 8.0);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_160: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
    clone_16: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_223: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_16, [1, 128, 768]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_224: "f32[128, 768]" = torch.ops.aten.view.default(view_223, [128, 768]);  view_223 = None
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_34: "f32[128, 768]" = torch.ops.aten.mm.default(view_224, permute_161);  permute_161 = None
    permute_162: "f32[768, 128]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_162, view_75);  permute_162 = view_75 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[768]" = torch.ops.aten.view.default(sum_59, [768]);  sum_59 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    view_226: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_34, [1, 128, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_66: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_140, view_226);  mul_140 = view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_165: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
    view_227: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_165, [1, 128, 768]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_228: "f32[128, 768]" = torch.ops.aten.view.default(view_227, [128, 768]);  view_227 = None
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_36: "f32[128, 768]" = torch.ops.aten.mm.default(view_228, permute_166);  permute_166 = None
    permute_167: "f32[768, 128]" = torch.ops.aten.permute.default(view_228, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_167, view_72);  permute_167 = view_72 = None
    permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True);  view_228 = None
    view_229: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_230: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_36, [1, 128, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_67: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_66, view_230);  add_66 = view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_170: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_23, [0, 2, 1, 3]);  div_23 = None
    clone_17: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    view_231: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_17, [1, 128, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_232: "f32[128, 768]" = torch.ops.aten.view.default(view_231, [128, 768]);  view_231 = None
    permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_38: "f32[128, 768]" = torch.ops.aten.mm.default(view_232, permute_171);  permute_171 = None
    permute_172: "f32[768, 128]" = torch.ops.aten.permute.default(view_232, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_69);  permute_172 = view_69 = None
    permute_173: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_232, [0], True);  view_232 = None
    view_233: "f32[768]" = torch.ops.aten.view.default(sum_61, [768]);  sum_61 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    view_234: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_38, [1, 128, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_68: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_67, view_234);  add_67 = view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    sub_47: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_27);  add_21 = getitem_27 = None
    mul_146: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_6);  sub_47 = None
    mul_147: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_68, primals_51);  primals_51 = None
    mul_148: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_147, 768)
    sum_62: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True)
    mul_149: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_147, mul_146);  mul_147 = None
    sum_63: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [2], True);  mul_149 = None
    mul_150: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_146, sum_63);  sum_63 = None
    sub_48: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_148, sum_62);  mul_148 = sum_62 = None
    sub_49: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_150);  sub_48 = mul_150 = None
    div_24: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_151: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_49);  div_24 = sub_49 = None
    mul_152: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_68, mul_146);  mul_146 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_152, [0, 1]);  mul_152 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_68, [0, 1]);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_7: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_153: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_154: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_151, mul_153);  mul_153 = None
    clone_18: "f32[1, 128, 768]" = torch.ops.aten.clone.default(mul_154, memory_format = torch.contiguous_format);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_235: "f32[128, 768]" = torch.ops.aten.view.default(clone_18, [128, 768]);  clone_18 = None
    permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_40: "f32[128, 3072]" = torch.ops.aten.mm.default(view_235, permute_175);  permute_175 = None
    permute_176: "f32[768, 128]" = torch.ops.aten.permute.default(view_235, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_67);  permute_176 = view_67 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_235, [0], True);  view_235 = None
    view_236: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    permute_178: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    view_237: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_40, [1, 128, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_155: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476)
    erf_11: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_69: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_156: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
    mul_157: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, view_66)
    mul_158: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_157, -0.5);  mul_157 = None
    exp_12: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_158);  mul_158 = None
    mul_159: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_160: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, mul_159);  view_66 = mul_159 = None
    add_70: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_156, mul_160);  mul_156 = mul_160 = None
    mul_161: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_237, add_70);  view_237 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_238: "f32[128, 3072]" = torch.ops.aten.view.default(mul_161, [128, 3072]);  mul_161 = None
    permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_42: "f32[128, 768]" = torch.ops.aten.mm.default(view_238, permute_179);  permute_179 = None
    permute_180: "f32[3072, 128]" = torch.ops.aten.permute.default(view_238, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_65);  permute_180 = view_65 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_238, [0], True);  view_238 = None
    view_239: "f32[3072]" = torch.ops.aten.view.default(sum_67, [3072]);  sum_67 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_240: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_42, [1, 128, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_71: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_151, view_240);  mul_151 = view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    sub_50: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  add_17 = getitem_23 = None
    mul_162: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_5);  sub_50 = None
    mul_163: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_71, primals_45);  primals_45 = None
    mul_164: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_163, 768)
    sum_68: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True)
    mul_165: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_163, mul_162);  mul_163 = None
    sum_69: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True);  mul_165 = None
    mul_166: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_162, sum_69);  sum_69 = None
    sub_51: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_164, sum_68);  mul_164 = sum_68 = None
    sub_52: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_166);  sub_51 = mul_166 = None
    div_25: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_167: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_52);  div_25 = sub_52 = None
    mul_168: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_71, mul_162);  mul_162 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 1]);  mul_168 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_71, [0, 1]);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_241: "f32[128, 768]" = torch.ops.aten.view.default(mul_167, [128, 768])
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_44: "f32[128, 768]" = torch.ops.aten.mm.default(view_241, permute_183);  permute_183 = None
    permute_184: "f32[768, 128]" = torch.ops.aten.permute.default(view_241, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_63);  permute_184 = view_63 = None
    permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_241, [0], True);  view_241 = None
    view_242: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_243: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_44, [1, 128, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_244: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_243, [1, 128, 12, 64]);  view_243 = None
    permute_187: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_245: "f32[12, 128, 64]" = torch.ops.aten.view.default(permute_187, [12, 128, 64]);  permute_187 = None
    permute_188: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_24: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_188, view_245);  permute_188 = None
    permute_189: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    bmm_25: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_245, permute_189);  view_245 = permute_189 = None
    view_246: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_24, [1, 12, 128, 64]);  bmm_24 = None
    view_247: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_25, [1, 12, 128, 128]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_8: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_169: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_170: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_247, mul_169);  view_247 = mul_169 = None
    clone_19: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(mul_170, memory_format = torch.contiguous_format);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_11: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_171: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(clone_19, alias_11);  clone_19 = None
    sum_73: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [-1], True)
    mul_172: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_11, sum_73);  alias_11 = sum_73 = None
    sub_53: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_12, scalar_tensor_7, sub_53);  expand_12 = scalar_tensor_7 = sub_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_248: "f32[12, 128, 128]" = torch.ops.aten.view.default(where_13, [12, 128, 128]);  where_13 = None
    permute_190: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    bmm_26: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_190, view_248);  permute_190 = None
    permute_191: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_27: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_248, permute_191);  view_248 = permute_191 = None
    view_249: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_26, [1, 12, 64, 128]);  bmm_26 = None
    view_250: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_27, [1, 12, 128, 64]);  bmm_27 = None
    permute_192: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_249, [0, 1, 3, 2]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_26: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_250, 8.0);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_193: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    clone_20: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_251: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_20, [1, 128, 768]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_252: "f32[128, 768]" = torch.ops.aten.view.default(view_251, [128, 768]);  view_251 = None
    permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_46: "f32[128, 768]" = torch.ops.aten.mm.default(view_252, permute_194);  permute_194 = None
    permute_195: "f32[768, 128]" = torch.ops.aten.permute.default(view_252, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_195, view_52);  permute_195 = view_52 = None
    permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
    view_253: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_254: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_46, [1, 128, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_72: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_167, view_254);  mul_167 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_198: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
    view_255: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_198, [1, 128, 768]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_256: "f32[128, 768]" = torch.ops.aten.view.default(view_255, [128, 768]);  view_255 = None
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_48: "f32[128, 768]" = torch.ops.aten.mm.default(view_256, permute_199);  permute_199 = None
    permute_200: "f32[768, 128]" = torch.ops.aten.permute.default(view_256, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_49);  permute_200 = view_49 = None
    permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
    view_257: "f32[768]" = torch.ops.aten.view.default(sum_75, [768]);  sum_75 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_258: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_48, [1, 128, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_73: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_72, view_258);  add_72 = view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_203: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_26, [0, 2, 1, 3]);  div_26 = None
    clone_21: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    view_259: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_21, [1, 128, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_260: "f32[128, 768]" = torch.ops.aten.view.default(view_259, [128, 768]);  view_259 = None
    permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_50: "f32[128, 768]" = torch.ops.aten.mm.default(view_260, permute_204);  permute_204 = None
    permute_205: "f32[768, 128]" = torch.ops.aten.permute.default(view_260, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_46);  permute_205 = view_46 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
    view_261: "f32[768]" = torch.ops.aten.view.default(sum_76, [768]);  sum_76 = None
    permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_262: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_50, [1, 128, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_74: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_73, view_262);  add_73 = view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    sub_54: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_19);  add_14 = getitem_19 = None
    mul_173: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_4);  sub_54 = None
    mul_174: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_74, primals_35);  primals_35 = None
    mul_175: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_174, 768)
    sum_77: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_174, [2], True)
    mul_176: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_174, mul_173);  mul_174 = None
    sum_78: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True);  mul_176 = None
    mul_177: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_173, sum_78);  sum_78 = None
    sub_55: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_175, sum_77);  mul_175 = sum_77 = None
    sub_56: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_177);  sub_55 = mul_177 = None
    div_27: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_178: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_56);  div_27 = sub_56 = None
    mul_179: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_74, mul_173);  mul_173 = None
    sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_179, [0, 1]);  mul_179 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_74, [0, 1]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_9: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_180: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_181: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_178, mul_180);  mul_180 = None
    clone_22: "f32[1, 128, 768]" = torch.ops.aten.clone.default(mul_181, memory_format = torch.contiguous_format);  mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_263: "f32[128, 768]" = torch.ops.aten.view.default(clone_22, [128, 768]);  clone_22 = None
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_52: "f32[128, 3072]" = torch.ops.aten.mm.default(view_263, permute_208);  permute_208 = None
    permute_209: "f32[768, 128]" = torch.ops.aten.permute.default(view_263, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_44);  permute_209 = view_44 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
    view_264: "f32[768]" = torch.ops.aten.view.default(sum_81, [768]);  sum_81 = None
    permute_211: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_265: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_52, [1, 128, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_182: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_12: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
    add_75: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_183: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_75, 0.5);  add_75 = None
    mul_184: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, view_43)
    mul_185: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_184, -0.5);  mul_184 = None
    exp_13: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_185);  mul_185 = None
    mul_186: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_187: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, mul_186);  view_43 = mul_186 = None
    add_76: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_183, mul_187);  mul_183 = mul_187 = None
    mul_188: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_265, add_76);  view_265 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_266: "f32[128, 3072]" = torch.ops.aten.view.default(mul_188, [128, 3072]);  mul_188 = None
    permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_54: "f32[128, 768]" = torch.ops.aten.mm.default(view_266, permute_212);  permute_212 = None
    permute_213: "f32[3072, 128]" = torch.ops.aten.permute.default(view_266, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_42);  permute_213 = view_42 = None
    permute_214: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_82: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
    view_267: "f32[3072]" = torch.ops.aten.view.default(sum_82, [3072]);  sum_82 = None
    permute_215: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_268: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_54, [1, 128, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_77: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_178, view_268);  mul_178 = view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    sub_57: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_15);  add_10 = getitem_15 = None
    mul_189: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_3);  sub_57 = None
    mul_190: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_77, primals_29);  primals_29 = None
    mul_191: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_190, 768)
    sum_83: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_190, [2], True)
    mul_192: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_190, mul_189);  mul_190 = None
    sum_84: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True);  mul_192 = None
    mul_193: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_189, sum_84);  sum_84 = None
    sub_58: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_191, sum_83);  mul_191 = sum_83 = None
    sub_59: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_58, mul_193);  sub_58 = mul_193 = None
    div_28: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_194: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_59);  div_28 = sub_59 = None
    mul_195: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_77, mul_189);  mul_189 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 1]);  mul_195 = None
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_77, [0, 1]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_269: "f32[128, 768]" = torch.ops.aten.view.default(mul_194, [128, 768])
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_56: "f32[128, 768]" = torch.ops.aten.mm.default(view_269, permute_216);  permute_216 = None
    permute_217: "f32[768, 128]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_40);  permute_217 = view_40 = None
    permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[768]" = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_271: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_56, [1, 128, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_272: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_271, [1, 128, 12, 64]);  view_271 = None
    permute_220: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_273: "f32[12, 128, 64]" = torch.ops.aten.view.default(permute_220, [12, 128, 64]);  permute_220 = None
    permute_221: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    bmm_28: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_221, view_273);  permute_221 = None
    permute_222: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    bmm_29: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_273, permute_222);  view_273 = permute_222 = None
    view_274: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_28, [1, 12, 128, 64]);  bmm_28 = None
    view_275: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_29, [1, 12, 128, 128]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_10: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
    mul_196: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_197: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_275, mul_196);  view_275 = mul_196 = None
    clone_23: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(mul_197, memory_format = torch.contiguous_format);  mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_12: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_198: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(clone_23, alias_12);  clone_23 = None
    sum_88: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [-1], True)
    mul_199: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_12, sum_88);  alias_12 = sum_88 = None
    sub_60: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_7, scalar_tensor_8, sub_60);  expand_7 = scalar_tensor_8 = sub_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_276: "f32[12, 128, 128]" = torch.ops.aten.view.default(where_14, [12, 128, 128]);  where_14 = None
    permute_223: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    bmm_30: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_223, view_276);  permute_223 = None
    permute_224: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    bmm_31: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_276, permute_224);  view_276 = permute_224 = None
    view_277: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_30, [1, 12, 64, 128]);  bmm_30 = None
    view_278: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_31, [1, 12, 128, 64]);  bmm_31 = None
    permute_225: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_277, [0, 1, 3, 2]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_29: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_278, 8.0);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_226: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
    clone_24: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    view_279: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_24, [1, 128, 768]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_280: "f32[128, 768]" = torch.ops.aten.view.default(view_279, [128, 768]);  view_279 = None
    permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_58: "f32[128, 768]" = torch.ops.aten.mm.default(view_280, permute_227);  permute_227 = None
    permute_228: "f32[768, 128]" = torch.ops.aten.permute.default(view_280, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_228, view_29);  permute_228 = view_29 = None
    permute_229: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
    view_281: "f32[768]" = torch.ops.aten.view.default(sum_89, [768]);  sum_89 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    view_282: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_58, [1, 128, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_78: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_194, view_282);  mul_194 = view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_231: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
    view_283: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_231, [1, 128, 768]);  permute_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_284: "f32[128, 768]" = torch.ops.aten.view.default(view_283, [128, 768]);  view_283 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_60: "f32[128, 768]" = torch.ops.aten.mm.default(view_284, permute_232);  permute_232 = None
    permute_233: "f32[768, 128]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_233, view_26);  permute_233 = view_26 = None
    permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
    view_285: "f32[768]" = torch.ops.aten.view.default(sum_90, [768]);  sum_90 = None
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    view_286: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_60, [1, 128, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_79: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_78, view_286);  add_78 = view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_236: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_29, [0, 2, 1, 3]);  div_29 = None
    clone_25: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    view_287: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_25, [1, 128, 768]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_288: "f32[128, 768]" = torch.ops.aten.view.default(view_287, [128, 768]);  view_287 = None
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_62: "f32[128, 768]" = torch.ops.aten.mm.default(view_288, permute_237);  permute_237 = None
    permute_238: "f32[768, 128]" = torch.ops.aten.permute.default(view_288, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_238, view_23);  permute_238 = view_23 = None
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
    view_289: "f32[768]" = torch.ops.aten.view.default(sum_91, [768]);  sum_91 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_290: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_62, [1, 128, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_80: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_79, view_290);  add_79 = view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    sub_61: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_11);  add_7 = getitem_11 = None
    mul_200: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_2);  sub_61 = None
    mul_201: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_80, primals_19);  primals_19 = None
    mul_202: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_201, 768)
    sum_92: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True)
    mul_203: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_201, mul_200);  mul_201 = None
    sum_93: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True);  mul_203 = None
    mul_204: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_200, sum_93);  sum_93 = None
    sub_62: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_202, sum_92);  mul_202 = sum_92 = None
    sub_63: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_204);  sub_62 = mul_204 = None
    div_30: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_205: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_63);  div_30 = sub_63 = None
    mul_206: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_80, mul_200);  mul_200 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1]);  mul_206 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_80, [0, 1]);  add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_11: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_9, torch.float32);  getitem_9 = None
    mul_207: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_208: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_205, mul_207);  mul_207 = None
    clone_26: "f32[1, 128, 768]" = torch.ops.aten.clone.default(mul_208, memory_format = torch.contiguous_format);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_291: "f32[128, 768]" = torch.ops.aten.view.default(clone_26, [128, 768]);  clone_26 = None
    permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_64: "f32[128, 3072]" = torch.ops.aten.mm.default(view_291, permute_241);  permute_241 = None
    permute_242: "f32[768, 128]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_21);  permute_242 = view_21 = None
    permute_243: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_244: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_293: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm_64, [1, 128, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_209: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476)
    erf_13: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_81: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_210: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_81, 0.5);  add_81 = None
    mul_211: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, view_20)
    mul_212: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_211, -0.5);  mul_211 = None
    exp_14: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_212);  mul_212 = None
    mul_213: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_214: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, mul_213);  view_20 = mul_213 = None
    add_82: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_210, mul_214);  mul_210 = mul_214 = None
    mul_215: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_293, add_82);  view_293 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_294: "f32[128, 3072]" = torch.ops.aten.view.default(mul_215, [128, 3072]);  mul_215 = None
    permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_66: "f32[128, 768]" = torch.ops.aten.mm.default(view_294, permute_245);  permute_245 = None
    permute_246: "f32[3072, 128]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_19);  permute_246 = view_19 = None
    permute_247: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_97: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[3072]" = torch.ops.aten.view.default(sum_97, [3072]);  sum_97 = None
    permute_248: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_296: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_66, [1, 128, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_83: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_205, view_296);  mul_205 = view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    sub_64: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_7);  add_3 = getitem_7 = None
    mul_216: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_1);  sub_64 = None
    mul_217: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_83, primals_13);  primals_13 = None
    mul_218: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_217, 768)
    sum_98: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True)
    mul_219: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_217, mul_216);  mul_217 = None
    sum_99: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [2], True);  mul_219 = None
    mul_220: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_216, sum_99);  sum_99 = None
    sub_65: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_218, sum_98);  mul_218 = sum_98 = None
    sub_66: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_220);  sub_65 = mul_220 = None
    div_31: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_221: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_66);  div_31 = sub_66 = None
    mul_222: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_83, mul_216);  mul_216 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 1]);  mul_222 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_83, [0, 1]);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_297: "f32[128, 768]" = torch.ops.aten.view.default(mul_221, [128, 768])
    permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_68: "f32[128, 768]" = torch.ops.aten.mm.default(view_297, permute_249);  permute_249 = None
    permute_250: "f32[768, 128]" = torch.ops.aten.permute.default(view_297, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_17);  permute_250 = view_17 = None
    permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
    view_298: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_299: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_68, [1, 128, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_300: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_299, [1, 128, 12, 64]);  view_299 = None
    permute_253: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_301: "f32[12, 128, 64]" = torch.ops.aten.view.default(permute_253, [12, 128, 64]);  permute_253 = None
    permute_254: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    bmm_32: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_254, view_301);  permute_254 = None
    permute_255: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    bmm_33: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_301, permute_255);  view_301 = permute_255 = None
    view_302: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_32, [1, 12, 128, 64]);  bmm_32 = None
    view_303: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_33, [1, 12, 128, 128]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_12: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_223: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_224: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_303, mul_223);  view_303 = mul_223 = None
    clone_27: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(mul_224, memory_format = torch.contiguous_format);  mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_13: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_225: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(clone_27, alias_13);  clone_27 = None
    sum_103: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [-1], True)
    mul_226: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_13, sum_103);  alias_13 = sum_103 = None
    sub_67: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, scalar_tensor_9, sub_67);  expand_2 = scalar_tensor_9 = sub_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_304: "f32[12, 128, 128]" = torch.ops.aten.view.default(where_15, [12, 128, 128]);  where_15 = None
    permute_256: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_34: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_256, view_304);  permute_256 = None
    permute_257: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_35: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_304, permute_257);  view_304 = permute_257 = None
    view_305: "f32[1, 12, 64, 128]" = torch.ops.aten.view.default(bmm_34, [1, 12, 64, 128]);  bmm_34 = None
    view_306: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_35, [1, 12, 128, 64]);  bmm_35 = None
    permute_258: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_305, [0, 1, 3, 2]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_32: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_306, 8.0);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_259: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
    clone_28: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_307: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_28, [1, 128, 768]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_308: "f32[128, 768]" = torch.ops.aten.view.default(view_307, [128, 768]);  view_307 = None
    permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_70: "f32[128, 768]" = torch.ops.aten.mm.default(view_308, permute_260);  permute_260 = None
    permute_261: "f32[768, 128]" = torch.ops.aten.permute.default(view_308, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_261, view_6);  permute_261 = view_6 = None
    permute_262: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
    view_309: "f32[768]" = torch.ops.aten.view.default(sum_104, [768]);  sum_104 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_310: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_70, [1, 128, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_84: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_221, view_310);  mul_221 = view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_264: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
    view_311: "f32[1, 128, 768]" = torch.ops.aten.view.default(permute_264, [1, 128, 768]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_312: "f32[128, 768]" = torch.ops.aten.view.default(view_311, [128, 768]);  view_311 = None
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_72: "f32[128, 768]" = torch.ops.aten.mm.default(view_312, permute_265);  permute_265 = None
    permute_266: "f32[768, 128]" = torch.ops.aten.permute.default(view_312, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_266, view_3);  permute_266 = view_3 = None
    permute_267: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
    view_313: "f32[768]" = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
    view_314: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_72, [1, 128, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_85: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_84, view_314);  add_84 = view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_269: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_32, [0, 2, 1, 3]);  div_32 = None
    clone_29: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
    view_315: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_29, [1, 128, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_316: "f32[128, 768]" = torch.ops.aten.view.default(view_315, [128, 768]);  view_315 = None
    permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_74: "f32[128, 768]" = torch.ops.aten.mm.default(view_316, permute_270);  permute_270 = None
    permute_271: "f32[768, 128]" = torch.ops.aten.permute.default(view_316, [1, 0])
    mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_271, view);  permute_271 = view = None
    permute_272: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
    view_317: "f32[768]" = torch.ops.aten.view.default(sum_106, [768]);  sum_106 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_318: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_74, [1, 128, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_86: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_85, view_318);  add_85 = view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:137, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
    convert_element_type_13: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_227: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_228: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_86, mul_227);  add_86 = mul_227 = None
    clone_30: "f32[1, 128, 768]" = torch.ops.aten.clone.default(mul_228, memory_format = torch.contiguous_format);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:136, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
    sub_68: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
    mul_229: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt);  sub_68 = None
    mul_230: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(clone_30, primals_3);  primals_3 = None
    mul_231: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_230, 768)
    sum_107: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [2], True)
    mul_232: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_230, mul_229);  mul_230 = None
    sum_108: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_232, [2], True);  mul_232 = None
    mul_233: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_229, sum_108);  sum_108 = None
    sub_69: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_231, sum_107);  mul_231 = sum_107 = None
    sub_70: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_233);  sub_69 = mul_233 = None
    div_33: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_234: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_70);  div_33 = sub_70 = None
    mul_235: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(clone_30, mul_229);  mul_229 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_235, [0, 1]);  mul_235 = None
    sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_30, [0, 1]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:133, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
    eq_6: "b8[1, 128]" = torch.ops.aten.eq.Scalar(slice_2, -1)
    unsqueeze_2: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_6, -1);  eq_6 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[1, 128, 768]" = torch.ops.aten.where.self(unsqueeze_2, scalar_tensor_10, mul_234);  unsqueeze_2 = scalar_tensor_10 = None
    full_2: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_2, [slice_2], where_16, True);  full_2 = slice_2 = where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:120, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
    eq_7: "b8[1, 128]" = torch.ops.aten.eq.Scalar(primals_108, 0)
    unsqueeze_3: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_7, -1);  eq_7 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[1, 128, 768]" = torch.ops.aten.where.self(unsqueeze_3, scalar_tensor_11, mul_234);  unsqueeze_3 = scalar_tensor_11 = mul_234 = None
    full_3: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[30522, 768]" = torch.ops.aten._unsafe_index_put.default(full_3, [primals_108], where_17, True);  full_3 = primals_108 = where_17 = None
    return pytree.tree_unflatten([div_12, view_141, _unsafe_index_put_1, _unsafe_index_put, sum_109, sum_110, permute_273, view_317, permute_268, view_313, permute_263, view_309, permute_252, view_298, sum_100, sum_101, permute_248, view_295, permute_244, view_292, sum_94, sum_95, permute_240, view_289, permute_235, view_285, permute_230, view_281, permute_219, view_270, sum_85, sum_86, permute_215, view_267, permute_211, view_264, sum_79, sum_80, permute_207, view_261, permute_202, view_257, permute_197, view_253, permute_186, view_242, sum_70, sum_71, permute_182, view_239, permute_178, view_236, sum_64, sum_65, permute_174, view_233, permute_169, view_229, permute_164, view_225, permute_153, view_214, sum_55, sum_56, permute_149, view_211, permute_145, view_208, sum_49, sum_50, permute_141, view_205, permute_136, view_201, permute_131, view_197, permute_120, view_186, sum_40, sum_41, permute_116, view_183, permute_112, view_180, sum_34, sum_35, permute_108, view_177, permute_103, view_173, permute_98, view_169, permute_87, view_158, sum_25, sum_26, permute_83, view_155, permute_79, view_152, sum_19, sum_20, permute_75, view_149, sum_14, sum_15, permute_71, view_146, None, None, None], self._out_spec)
    