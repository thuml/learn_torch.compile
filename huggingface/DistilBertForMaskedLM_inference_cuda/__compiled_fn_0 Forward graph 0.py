from __future__ import annotations



def forward(self, arg0_1: "f32[30522, 768]", arg1_1: "f32[512, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768, 768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[3072]", arg16_1: "f32[768, 3072]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768, 768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768]", arg25_1: "f32[768]", arg26_1: "f32[768, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[3072, 768]", arg31_1: "f32[3072]", arg32_1: "f32[768, 3072]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768, 768]", arg37_1: "f32[768]", arg38_1: "f32[768, 768]", arg39_1: "f32[768]", arg40_1: "f32[768, 768]", arg41_1: "f32[768]", arg42_1: "f32[768, 768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[3072, 768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768, 768]", arg53_1: "f32[768]", arg54_1: "f32[768, 768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[3072]", arg64_1: "f32[768, 3072]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768, 768]", arg71_1: "f32[768]", arg72_1: "f32[768, 768]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[3072, 768]", arg79_1: "f32[3072]", arg80_1: "f32[768, 3072]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768, 768]", arg85_1: "f32[768]", arg86_1: "f32[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[3072, 768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768, 768]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[30522, 768]", arg105_1: "f32[30522]", arg106_1: "i64[1, 512]", arg107_1: "i64[1, 128]", arg108_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:602, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
    full: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:120, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
    embedding: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg0_1, arg107_1, 0);  arg0_1 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:128, code: position_ids = self.position_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(arg106_1, 0, 0, 9223372036854775807);  arg106_1 = None
    slice_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 128);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:133, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
    embedding_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, slice_2);  arg1_1 = slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:135, code: embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
    add: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:136, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
    mul: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_2: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:137, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
    clone: "f32[1, 128, 768]" = torch.ops.aten.clone.default(add_2);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view: "f32[128, 768]" = torch.ops.aten.view.default(clone, [128, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm: "f32[128, 768]" = torch.ops.aten.addmm.default(arg5_1, view, permute);  arg5_1 = view = permute = None
    view_1: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm, [1, 128, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_2: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_1, [1, -1, 12, 64]);  view_1 = None
    permute_1: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_3: "f32[128, 768]" = torch.ops.aten.view.default(clone, [128, 768])
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm_1: "f32[128, 768]" = torch.ops.aten.addmm.default(arg7_1, view_3, permute_2);  arg7_1 = view_3 = permute_2 = None
    view_4: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_1, [1, 128, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_5: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_4, [1, -1, 12, 64]);  view_4 = None
    permute_3: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_6: "f32[128, 768]" = torch.ops.aten.view.default(clone, [128, 768])
    permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    addmm_2: "f32[128, 768]" = torch.ops.aten.addmm.default(arg9_1, view_6, permute_4);  arg9_1 = view_6 = permute_4 = None
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
    bmm: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
    view_11: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm, [1, 12, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_12: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq, [1, 1, 1, 128]);  eq = None
    expand_2: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_12, [1, 12, 128, 128]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, lift_fresh_copy, view_11);  expand_2 = lift_fresh_copy = view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    clone_1: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_3: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(clone_1, [1, 12, 128, 128]);  clone_1 = None
    view_13: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_3, [12, 128, 128]);  expand_3 = None
    expand_4: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 128, 64]);  permute_5 = None
    view_14: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_4, [12, 128, 64]);  expand_4 = None
    bmm_1: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
    view_15: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_7: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
    clone_2: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_16: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_2, [1, -1, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_17: "f32[128, 768]" = torch.ops.aten.view.default(view_16, [128, 768]);  view_16 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm_3: "f32[128, 768]" = torch.ops.aten.addmm.default(arg11_1, view_17, permute_8);  arg11_1 = view_17 = permute_8 = None
    view_18: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_3, [1, 128, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_3: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_18, clone);  view_18 = clone = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
    mul_2: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_3: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg12_1);  mul_2 = arg12_1 = None
    add_5: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, arg13_1);  mul_3 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_19: "f32[128, 768]" = torch.ops.aten.view.default(add_5, [128, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    addmm_4: "f32[128, 3072]" = torch.ops.aten.addmm.default(arg15_1, view_19, permute_9);  arg15_1 = view_19 = permute_9 = None
    view_20: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 128, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_4: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, 0.5)
    mul_5: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476);  view_20 = None
    erf: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_21: "f32[128, 3072]" = torch.ops.aten.view.default(mul_6, [128, 3072]);  mul_6 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_5: "f32[128, 768]" = torch.ops.aten.addmm.default(arg17_1, view_21, permute_10);  arg17_1 = view_21 = permute_10 = None
    view_22: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_5, [1, 128, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    clone_3: "f32[1, 128, 768]" = torch.ops.aten.clone.default(view_22);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_7: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(clone_3, add_5);  clone_3 = add_5 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  add_7 = getitem_5 = None
    mul_7: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_8: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_7, arg18_1);  mul_7 = arg18_1 = None
    add_9: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_8, arg19_1);  mul_8 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_23: "f32[128, 768]" = torch.ops.aten.view.default(add_9, [128, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_6: "f32[128, 768]" = torch.ops.aten.addmm.default(arg21_1, view_23, permute_11);  arg21_1 = view_23 = permute_11 = None
    view_24: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_6, [1, 128, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_25: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_24, [1, -1, 12, 64]);  view_24 = None
    permute_12: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_26: "f32[128, 768]" = torch.ops.aten.view.default(add_9, [128, 768])
    permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    addmm_7: "f32[128, 768]" = torch.ops.aten.addmm.default(arg23_1, view_26, permute_13);  arg23_1 = view_26 = permute_13 = None
    view_27: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_7, [1, 128, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_28: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_27, [1, -1, 12, 64]);  view_27 = None
    permute_14: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_29: "f32[128, 768]" = torch.ops.aten.view.default(add_9, [128, 768])
    permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_8: "f32[128, 768]" = torch.ops.aten.addmm.default(arg25_1, view_29, permute_15);  arg25_1 = view_29 = permute_15 = None
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
    bmm_2: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_32, view_33);  view_32 = view_33 = None
    view_34: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_2, [1, 12, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_1: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_35: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_1, [1, 1, 1, 128]);  eq_1 = None
    expand_7: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_35, [1, 12, 128, 128]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant1 = self._tensor_constant1
    lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_1: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_7, lift_fresh_copy_1, view_34);  expand_7 = lift_fresh_copy_1 = view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_1: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    clone_4: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_8: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(clone_4, [1, 12, 128, 128]);  clone_4 = None
    view_36: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_8, [12, 128, 128]);  expand_8 = None
    expand_9: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_16, [1, 12, 128, 64]);  permute_16 = None
    view_37: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_9, [12, 128, 64]);  expand_9 = None
    bmm_3: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
    view_38: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_18: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_5: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_39: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_5, [1, -1, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_40: "f32[128, 768]" = torch.ops.aten.view.default(view_39, [128, 768]);  view_39 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_9: "f32[128, 768]" = torch.ops.aten.addmm.default(arg27_1, view_40, permute_19);  arg27_1 = view_40 = permute_19 = None
    view_41: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_9, [1, 128, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_10: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_41, add_9);  view_41 = add_9 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_5: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_7);  add_10 = getitem_7 = None
    mul_9: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_10: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg28_1);  mul_9 = arg28_1 = None
    add_12: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_10, arg29_1);  mul_10 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_42: "f32[128, 768]" = torch.ops.aten.view.default(add_12, [128, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_10: "f32[128, 3072]" = torch.ops.aten.addmm.default(arg31_1, view_42, permute_20);  arg31_1 = view_42 = permute_20 = None
    view_43: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 128, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_11: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_12: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_1: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_44: "f32[128, 3072]" = torch.ops.aten.view.default(mul_13, [128, 3072]);  mul_13 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    addmm_11: "f32[128, 768]" = torch.ops.aten.addmm.default(arg33_1, view_44, permute_21);  arg33_1 = view_44 = permute_21 = None
    view_45: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_11, [1, 128, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    clone_6: "f32[1, 128, 768]" = torch.ops.aten.clone.default(view_45);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_14: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(clone_6, add_12);  clone_6 = add_12 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  add_14 = getitem_9 = None
    mul_14: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_15: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg34_1);  mul_14 = arg34_1 = None
    add_16: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_15, arg35_1);  mul_15 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_46: "f32[128, 768]" = torch.ops.aten.view.default(add_16, [128, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_12: "f32[128, 768]" = torch.ops.aten.addmm.default(arg37_1, view_46, permute_22);  arg37_1 = view_46 = permute_22 = None
    view_47: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_12, [1, 128, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_48: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_47, [1, -1, 12, 64]);  view_47 = None
    permute_23: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_49: "f32[128, 768]" = torch.ops.aten.view.default(add_16, [128, 768])
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    addmm_13: "f32[128, 768]" = torch.ops.aten.addmm.default(arg39_1, view_49, permute_24);  arg39_1 = view_49 = permute_24 = None
    view_50: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_13, [1, 128, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_51: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_50, [1, -1, 12, 64]);  view_50 = None
    permute_25: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_52: "f32[128, 768]" = torch.ops.aten.view.default(add_16, [128, 768])
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    addmm_14: "f32[128, 768]" = torch.ops.aten.addmm.default(arg41_1, view_52, permute_26);  arg41_1 = view_52 = permute_26 = None
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
    bmm_4: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
    view_57: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_4, [1, 12, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_2: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_58: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_2, [1, 1, 1, 128]);  eq_2 = None
    expand_12: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_58, [1, 12, 128, 128]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant2 = self._tensor_constant2
    lift_fresh_copy_2: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_2: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_12, lift_fresh_copy_2, view_57);  expand_12 = lift_fresh_copy_2 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_2: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    clone_7: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_13: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(clone_7, [1, 12, 128, 128]);  clone_7 = None
    view_59: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_13, [12, 128, 128]);  expand_13 = None
    expand_14: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 128, 64]);  permute_27 = None
    view_60: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_14, [12, 128, 64]);  expand_14 = None
    bmm_5: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_59, view_60);  view_59 = view_60 = None
    view_61: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_29: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
    clone_8: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_62: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_8, [1, -1, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_63: "f32[128, 768]" = torch.ops.aten.view.default(view_62, [128, 768]);  view_62 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_15: "f32[128, 768]" = torch.ops.aten.addmm.default(arg43_1, view_63, permute_30);  arg43_1 = view_63 = permute_30 = None
    view_64: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_15, [1, 128, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_17: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_64, add_16);  view_64 = add_16 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_8: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  add_17 = getitem_11 = None
    mul_16: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_17: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg44_1);  mul_16 = arg44_1 = None
    add_19: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, arg45_1);  mul_17 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_65: "f32[128, 768]" = torch.ops.aten.view.default(add_19, [128, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_16: "f32[128, 3072]" = torch.ops.aten.addmm.default(arg47_1, view_65, permute_31);  arg47_1 = view_65 = permute_31 = None
    view_66: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 128, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_18: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
    mul_19: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476);  view_66 = None
    erf_2: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_67: "f32[128, 3072]" = torch.ops.aten.view.default(mul_20, [128, 3072]);  mul_20 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_17: "f32[128, 768]" = torch.ops.aten.addmm.default(arg49_1, view_67, permute_32);  arg49_1 = view_67 = permute_32 = None
    view_68: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_17, [1, 128, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    clone_9: "f32[1, 128, 768]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_21: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(clone_9, add_19);  clone_9 = add_19 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_22: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_9: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_13);  add_21 = getitem_13 = None
    mul_21: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_22: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg50_1);  mul_21 = arg50_1 = None
    add_23: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_22, arg51_1);  mul_22 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_69: "f32[128, 768]" = torch.ops.aten.view.default(add_23, [128, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_18: "f32[128, 768]" = torch.ops.aten.addmm.default(arg53_1, view_69, permute_33);  arg53_1 = view_69 = permute_33 = None
    view_70: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_18, [1, 128, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_71: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_70, [1, -1, 12, 64]);  view_70 = None
    permute_34: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_72: "f32[128, 768]" = torch.ops.aten.view.default(add_23, [128, 768])
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_19: "f32[128, 768]" = torch.ops.aten.addmm.default(arg55_1, view_72, permute_35);  arg55_1 = view_72 = permute_35 = None
    view_73: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_19, [1, 128, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_74: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_73, [1, -1, 12, 64]);  view_73 = None
    permute_36: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_75: "f32[128, 768]" = torch.ops.aten.view.default(add_23, [128, 768])
    permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_20: "f32[128, 768]" = torch.ops.aten.addmm.default(arg57_1, view_75, permute_37);  arg57_1 = view_75 = permute_37 = None
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
    bmm_6: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
    view_80: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_6, [1, 12, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_3: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_81: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_3, [1, 1, 1, 128]);  eq_3 = None
    expand_17: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_81, [1, 12, 128, 128]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant3 = self._tensor_constant3
    lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_3: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_17, lift_fresh_copy_3, view_80);  expand_17 = lift_fresh_copy_3 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_3: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    clone_10: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_18: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(clone_10, [1, 12, 128, 128]);  clone_10 = None
    view_82: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_18, [12, 128, 128]);  expand_18 = None
    expand_19: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_38, [1, 12, 128, 64]);  permute_38 = None
    view_83: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_19, [12, 128, 64]);  expand_19 = None
    bmm_7: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
    view_84: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_40: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    clone_11: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_85: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_11, [1, -1, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_86: "f32[128, 768]" = torch.ops.aten.view.default(view_85, [128, 768]);  view_85 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_21: "f32[128, 768]" = torch.ops.aten.addmm.default(arg59_1, view_86, permute_41);  arg59_1 = view_86 = permute_41 = None
    view_87: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_21, [1, 128, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_24: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_87, add_23);  view_87 = add_23 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_11: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_15);  add_24 = getitem_15 = None
    mul_23: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_24: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_23, arg60_1);  mul_23 = arg60_1 = None
    add_26: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_24, arg61_1);  mul_24 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_88: "f32[128, 768]" = torch.ops.aten.view.default(add_26, [128, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    addmm_22: "f32[128, 3072]" = torch.ops.aten.addmm.default(arg63_1, view_88, permute_42);  arg63_1 = view_88 = permute_42 = None
    view_89: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 128, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_25: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    mul_26: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.7071067811865476);  view_89 = None
    erf_3: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_27: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_27: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_90: "f32[128, 3072]" = torch.ops.aten.view.default(mul_27, [128, 3072]);  mul_27 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_23: "f32[128, 768]" = torch.ops.aten.addmm.default(arg65_1, view_90, permute_43);  arg65_1 = view_90 = permute_43 = None
    view_91: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_23, [1, 128, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    clone_12: "f32[1, 128, 768]" = torch.ops.aten.clone.default(view_91);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_28: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(clone_12, add_26);  clone_12 = add_26 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    add_29: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_12: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_17);  add_28 = getitem_17 = None
    mul_28: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_29: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg66_1);  mul_28 = arg66_1 = None
    add_30: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_29, arg67_1);  mul_29 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_92: "f32[128, 768]" = torch.ops.aten.view.default(add_30, [128, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm_24: "f32[128, 768]" = torch.ops.aten.addmm.default(arg69_1, view_92, permute_44);  arg69_1 = view_92 = permute_44 = None
    view_93: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_24, [1, 128, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_94: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_93, [1, -1, 12, 64]);  view_93 = None
    permute_45: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_95: "f32[128, 768]" = torch.ops.aten.view.default(add_30, [128, 768])
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_25: "f32[128, 768]" = torch.ops.aten.addmm.default(arg71_1, view_95, permute_46);  arg71_1 = view_95 = permute_46 = None
    view_96: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_25, [1, 128, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_97: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_96, [1, -1, 12, 64]);  view_96 = None
    permute_47: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_98: "f32[128, 768]" = torch.ops.aten.view.default(add_30, [128, 768])
    permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_26: "f32[128, 768]" = torch.ops.aten.addmm.default(arg73_1, view_98, permute_48);  arg73_1 = view_98 = permute_48 = None
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
    bmm_8: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_101, view_102);  view_101 = view_102 = None
    view_103: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_8, [1, 12, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_4: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0)
    view_104: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_4, [1, 1, 1, 128]);  eq_4 = None
    expand_22: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_104, [1, 12, 128, 128]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant4 = self._tensor_constant4
    lift_fresh_copy_4: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_4: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_22, lift_fresh_copy_4, view_103);  expand_22 = lift_fresh_copy_4 = view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_4: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    clone_13: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_23: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(clone_13, [1, 12, 128, 128]);  clone_13 = None
    view_105: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_23, [12, 128, 128]);  expand_23 = None
    expand_24: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_49, [1, 12, 128, 64]);  permute_49 = None
    view_106: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_24, [12, 128, 64]);  expand_24 = None
    bmm_9: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_105, view_106);  view_105 = view_106 = None
    view_107: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_51: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    clone_14: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_108: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_14, [1, -1, 768]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_109: "f32[128, 768]" = torch.ops.aten.view.default(view_108, [128, 768]);  view_108 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_27: "f32[128, 768]" = torch.ops.aten.addmm.default(arg75_1, view_109, permute_52);  arg75_1 = view_109 = permute_52 = None
    view_110: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_27, [1, 128, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_31: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_110, add_30);  view_110 = add_30 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_14: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_19);  add_31 = getitem_19 = None
    mul_30: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_31: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg76_1);  mul_30 = arg76_1 = None
    add_33: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_31, arg77_1);  mul_31 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_111: "f32[128, 768]" = torch.ops.aten.view.default(add_33, [128, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_28: "f32[128, 3072]" = torch.ops.aten.addmm.default(arg79_1, view_111, permute_53);  arg79_1 = view_111 = permute_53 = None
    view_112: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 128, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_32: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.5)
    mul_33: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.7071067811865476);  view_112 = None
    erf_4: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_34: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_34: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_113: "f32[128, 3072]" = torch.ops.aten.view.default(mul_34, [128, 3072]);  mul_34 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_29: "f32[128, 768]" = torch.ops.aten.addmm.default(arg81_1, view_113, permute_54);  arg81_1 = view_113 = permute_54 = None
    view_114: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_29, [1, 128, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    clone_15: "f32[1, 128, 768]" = torch.ops.aten.clone.default(view_114);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_35: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(clone_15, add_33);  clone_15 = add_33 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_15: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_21);  add_35 = getitem_21 = None
    mul_35: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_36: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg82_1);  mul_35 = arg82_1 = None
    add_37: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_36, arg83_1);  mul_36 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_115: "f32[128, 768]" = torch.ops.aten.view.default(add_37, [128, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_30: "f32[128, 768]" = torch.ops.aten.addmm.default(arg85_1, view_115, permute_55);  arg85_1 = view_115 = permute_55 = None
    view_116: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_30, [1, 128, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_117: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_116, [1, -1, 12, 64]);  view_116 = None
    permute_56: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_118: "f32[128, 768]" = torch.ops.aten.view.default(add_37, [128, 768])
    permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_31: "f32[128, 768]" = torch.ops.aten.addmm.default(arg87_1, view_118, permute_57);  arg87_1 = view_118 = permute_57 = None
    view_119: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_31, [1, 128, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_120: "f32[1, 128, 12, 64]" = torch.ops.aten.view.default(view_119, [1, -1, 12, 64]);  view_119 = None
    permute_58: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_121: "f32[128, 768]" = torch.ops.aten.view.default(add_37, [128, 768])
    permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_32: "f32[128, 768]" = torch.ops.aten.addmm.default(arg89_1, view_121, permute_59);  arg89_1 = view_121 = permute_59 = None
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
    bmm_10: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
    view_126: "f32[1, 12, 128, 128]" = torch.ops.aten.view.default(bmm_10, [1, 12, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq_5: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0);  full = None
    view_127: "b8[1, 1, 1, 128]" = torch.ops.aten.view.default(eq_5, [1, 1, 1, 128]);  eq_5 = None
    expand_27: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_127, [1, 12, 128, 128]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    _tensor_constant5 = self._tensor_constant5
    lift_fresh_copy_5: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_5: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_27, lift_fresh_copy_5, view_126);  expand_27 = lift_fresh_copy_5 = view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    amax_5: "f32[1, 12, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 12, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    clone_16: "f32[1, 12, 128, 128]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    expand_28: "f32[1, 12, 128, 128]" = torch.ops.aten.expand.default(clone_16, [1, 12, 128, 128]);  clone_16 = None
    view_128: "f32[12, 128, 128]" = torch.ops.aten.view.default(expand_28, [12, 128, 128]);  expand_28 = None
    expand_29: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_60, [1, 12, 128, 64]);  permute_60 = None
    view_129: "f32[12, 128, 64]" = torch.ops.aten.view.default(expand_29, [12, 128, 64]);  expand_29 = None
    bmm_11: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_128, view_129);  view_128 = view_129 = None
    view_130: "f32[1, 12, 128, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_62: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    clone_17: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_131: "f32[1, 128, 768]" = torch.ops.aten.view.default(clone_17, [1, -1, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_132: "f32[128, 768]" = torch.ops.aten.view.default(view_131, [128, 768]);  view_131 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_33: "f32[128, 768]" = torch.ops.aten.addmm.default(arg91_1, view_132, permute_63);  arg91_1 = view_132 = permute_63 = None
    view_133: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_33, [1, 128, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_38: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_133, add_37);  view_133 = add_37 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_17: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_23);  add_38 = getitem_23 = None
    mul_37: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_38: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_37, arg92_1);  mul_37 = arg92_1 = None
    add_40: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_38, arg93_1);  mul_38 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_134: "f32[128, 768]" = torch.ops.aten.view.default(add_40, [128, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_34: "f32[128, 3072]" = torch.ops.aten.addmm.default(arg95_1, view_134, permute_64);  arg95_1 = view_134 = permute_64 = None
    view_135: "f32[1, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 128, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_39: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, 0.5)
    mul_40: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476);  view_135 = None
    erf_5: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_41: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_41: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_136: "f32[128, 3072]" = torch.ops.aten.view.default(mul_41, [128, 3072]);  mul_41 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_35: "f32[128, 768]" = torch.ops.aten.addmm.default(arg97_1, view_136, permute_65);  arg97_1 = view_136 = permute_65 = None
    view_137: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_35, [1, 128, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    clone_18: "f32[1, 128, 768]" = torch.ops.aten.clone.default(view_137);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    add_42: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(clone_18, add_40);  clone_18 = add_40 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_43: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_18: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_25);  add_42 = getitem_25 = None
    mul_42: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_43: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg98_1);  mul_42 = arg98_1 = None
    add_44: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, arg99_1);  mul_43 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:702, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
    view_138: "f32[128, 768]" = torch.ops.aten.view.default(add_44, [128, 768]);  add_44 = None
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_36: "f32[128, 768]" = torch.ops.aten.addmm.default(arg101_1, view_138, permute_66);  arg101_1 = view_138 = permute_66 = None
    view_139: "f32[1, 128, 768]" = torch.ops.aten.view.default(addmm_36, [1, 128, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_44: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, 0.5)
    mul_45: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_139, 0.7071067811865476);  view_139 = None
    erf_6: "f32[1, 128, 768]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
    add_45: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_46: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_44, add_45);  mul_44 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:704, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
    var_mean_13 = torch.ops.aten.var_mean.correction(mul_46, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_19: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_46, getitem_27);  mul_46 = getitem_27 = None
    mul_47: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = rsqrt_13 = None
    mul_48: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_47, arg102_1);  mul_47 = arg102_1 = None
    add_47: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_48, arg103_1);  mul_48 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:705, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
    view_140: "f32[128, 768]" = torch.ops.aten.view.default(add_47, [128, 768]);  add_47 = None
    permute_67: "f32[768, 30522]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_37: "f32[128, 30522]" = torch.ops.aten.addmm.default(arg105_1, view_140, permute_67);  arg105_1 = view_140 = permute_67 = None
    view_141: "f32[1, 128, 30522]" = torch.ops.aten.view.default(addmm_37, [1, 128, 30522]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:709, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
    view_142: "f32[128, 30522]" = torch.ops.aten.view.default(view_141, [-1, 30522])
    view_143: "i64[128]" = torch.ops.aten.view.default(arg108_1, [-1]);  arg108_1 = None
    amax_6: "f32[128, 1]" = torch.ops.aten.amax.default(view_142, [1], True)
    sub_20: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(view_142, amax_6);  view_142 = amax_6 = None
    exp_6: "f32[128, 30522]" = torch.ops.aten.exp.default(sub_20)
    sum_7: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
    sub_21: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(sub_20, log);  sub_20 = log = None
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
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_143, -100);  view_143 = None
    sum_8: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
    sum_9: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
    div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type);  sum_9 = convert_element_type = None
    return (div_12, view_141)
    