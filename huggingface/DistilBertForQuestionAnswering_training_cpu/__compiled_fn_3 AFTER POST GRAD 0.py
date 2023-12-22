from __future__ import annotations



def forward(self, primals_1: "f32[30522, 768]", primals_2: "f32[512, 768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768, 768]", primals_6: "f32[768]", primals_7: "f32[768, 768]", primals_8: "f32[768]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768, 768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[3072, 768]", primals_16: "f32[3072]", primals_17: "f32[768, 3072]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768, 768]", primals_22: "f32[768]", primals_23: "f32[768, 768]", primals_24: "f32[768]", primals_25: "f32[768, 768]", primals_26: "f32[768]", primals_27: "f32[768, 768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[3072, 768]", primals_32: "f32[3072]", primals_33: "f32[768, 3072]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768, 768]", primals_38: "f32[768]", primals_39: "f32[768, 768]", primals_40: "f32[768]", primals_41: "f32[768, 768]", primals_42: "f32[768]", primals_43: "f32[768, 768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[3072, 768]", primals_48: "f32[3072]", primals_49: "f32[768, 3072]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768, 768]", primals_54: "f32[768]", primals_55: "f32[768, 768]", primals_56: "f32[768]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_59: "f32[768, 768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[3072, 768]", primals_64: "f32[3072]", primals_65: "f32[768, 3072]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768, 768]", primals_70: "f32[768]", primals_71: "f32[768, 768]", primals_72: "f32[768]", primals_73: "f32[768, 768]", primals_74: "f32[768]", primals_75: "f32[768, 768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[3072, 768]", primals_80: "f32[3072]", primals_81: "f32[768, 3072]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768, 768]", primals_86: "f32[768]", primals_87: "f32[768, 768]", primals_88: "f32[768]", primals_89: "f32[768, 768]", primals_90: "f32[768]", primals_91: "f32[768, 768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_94: "f32[768]", primals_95: "f32[3072, 768]", primals_96: "f32[3072]", primals_97: "f32[768, 3072]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[2, 768]", primals_102: "f32[2]", primals_103: "i64[1, 512]", primals_104: "i64[1, 128]", primals_105: "i64[1]", primals_106: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:602, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
    full: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:120, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
    embedding: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(primals_1, primals_104, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:128, code: position_ids = self.position_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_103, 0, 0, 9223372036854775807);  primals_103 = None
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
    sub: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
    mul: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul, primals_3)
    add_2: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:137, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
    native_dropout = torch.ops.aten.native_dropout.default(add_2, 0.1, True);  add_2 = None
    getitem_2: "f32[1, 128, 768]" = native_dropout[0]
    getitem_3: "b8[1, 128, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view: "f32[128, 768]" = torch.ops.aten.reshape.default(getitem_2, [128, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_6, view, permute);  primals_6 = None
    view_1: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm, [1, 128, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_2: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_1, [1, -1, 12, 64]);  view_1 = None
    permute_1: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_1: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_8, view, permute_2);  primals_8 = None
    view_4: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_5: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_4, [1, -1, 12, 64]);  view_4 = None
    permute_3: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_2: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_10, view, permute_4);  primals_10 = None
    view_7: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_8: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_7, [1, -1, 12, 64]);  view_7 = None
    permute_5: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_1, 8.0);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_6: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    expand: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div, [1, 12, 128, 64]);  div = None
    view_9: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand, [12, 128, 64]);  expand = None
    expand_1: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 128]);  permute_6 = None
    view_10: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_1, [12, 64, 128]);  expand_1 = None
    bmm: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 12, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(full, 0);  full = None
    view_12: "b8[1, 1, 1, 128]" = torch.ops.aten.reshape.default(eq, [1, 1, 1, 128]);  eq = None
    expand_2: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_12, [1, 12, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:223, code: mask, torch.tensor(torch.finfo(scores.dtype).min)
    full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_11);  view_11 = None
    
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
    view_13: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [12, 128, 128]);  expand_3 = None
    expand_4: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 128, 64]);  permute_5 = None
    view_14: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_4, [12, 128, 64]);  expand_4 = None
    bmm_1: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_13, view_14)
    view_15: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_7: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
    clone: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_16: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone, [1, -1, 768]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_17: "f32[128, 768]" = torch.ops.aten.reshape.default(view_16, [128, 768]);  view_16 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_3: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_12, view_17, permute_8);  primals_12 = None
    view_18: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_3: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_18, getitem_2);  view_18 = getitem_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_7);  add_3 = getitem_7 = None
    mul_2: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_3: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_13)
    add_5: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_14);  mul_3 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_19: "f32[128, 768]" = torch.ops.aten.reshape.default(add_5, [128, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_4: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_16, view_19, permute_9);  primals_16 = None
    view_20: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_4: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, 0.5)
    mul_5: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476);  view_20 = None
    erf: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_21: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_6, [128, 3072]);  mul_6 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_5: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_18, view_21, permute_10);  primals_18 = None
    view_22: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 768]);  addmm_5 = None
    
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
    sub_3: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_11);  add_7 = getitem_11 = None
    mul_7: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_8: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_19)
    add_9: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_8, primals_20);  mul_8 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_23: "f32[128, 768]" = torch.ops.aten.reshape.default(add_9, [128, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_6: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_22, view_23, permute_11);  primals_22 = None
    view_24: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_25: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_24, [1, -1, 12, 64]);  view_24 = None
    permute_12: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_7: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_24, view_23, permute_13);  primals_24 = None
    view_27: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_28: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_27, [1, -1, 12, 64]);  view_27 = None
    permute_14: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_8: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_26, view_23, permute_15);  primals_26 = None
    view_30: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_31: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_30, [1, -1, 12, 64]);  view_30 = None
    permute_16: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_2: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_12, 8.0);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_17: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
    expand_5: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_2, [1, 12, 128, 64]);  div_2 = None
    view_32: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_5, [12, 128, 64]);  expand_5 = None
    expand_6: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 128]);  permute_17 = None
    view_33: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_6, [12, 64, 128]);  expand_6 = None
    bmm_2: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_32, view_33)
    view_34: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_1: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_34);  view_34 = None
    
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
    view_36: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_8, [12, 128, 128]);  expand_8 = None
    expand_9: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_16, [1, 12, 128, 64]);  permute_16 = None
    view_37: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_9, [12, 128, 64]);  expand_9 = None
    bmm_3: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_18: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_1: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_39: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_1, [1, -1, 768]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_40: "f32[128, 768]" = torch.ops.aten.reshape.default(view_39, [128, 768]);  view_39 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_9: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_28, view_40, permute_19);  primals_28 = None
    view_41: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_10: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_41, add_9);  view_41 = add_9 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_5: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_15);  add_10 = getitem_15 = None
    mul_9: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_10: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_29)
    add_12: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_30);  mul_10 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_42: "f32[128, 768]" = torch.ops.aten.reshape.default(add_12, [128, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_10: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_32, view_42, permute_20);  primals_32 = None
    view_43: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_11: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_12: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_1: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_44: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_13, [128, 3072]);  mul_13 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    addmm_11: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_34, view_44, permute_21);  primals_34 = None
    view_45: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 768]);  addmm_11 = None
    
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
    sub_6: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_19);  add_14 = getitem_19 = None
    mul_14: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_15: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_35)
    add_16: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_36);  mul_15 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_46: "f32[128, 768]" = torch.ops.aten.reshape.default(add_16, [128, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_12: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_38, view_46, permute_22);  primals_38 = None
    view_47: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_48: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_47, [1, -1, 12, 64]);  view_47 = None
    permute_23: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    addmm_13: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_40, view_46, permute_24);  primals_40 = None
    view_50: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_51: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_50, [1, -1, 12, 64]);  view_50 = None
    permute_25: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    addmm_14: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_42, view_46, permute_26);  primals_42 = None
    view_53: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_54: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_53, [1, -1, 12, 64]);  view_53 = None
    permute_27: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_4: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_23, 8.0);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_28: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
    expand_10: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_4, [1, 12, 128, 64]);  div_4 = None
    view_55: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_10, [12, 128, 64]);  expand_10 = None
    expand_11: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 128]);  permute_28 = None
    view_56: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_11, [12, 64, 128]);  expand_11 = None
    bmm_4: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56)
    view_57: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_2: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_57);  view_57 = None
    
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
    view_59: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_13, [12, 128, 128]);  expand_13 = None
    expand_14: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 128, 64]);  permute_27 = None
    view_60: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_14, [12, 128, 64]);  expand_14 = None
    bmm_5: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_59, view_60)
    view_61: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_29: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
    clone_2: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_62: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_2, [1, -1, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_63: "f32[128, 768]" = torch.ops.aten.reshape.default(view_62, [128, 768]);  view_62 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_15: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_44, view_63, permute_30);  primals_44 = None
    view_64: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_17: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_64, add_16);  view_64 = add_16 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_8: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  add_17 = getitem_23 = None
    mul_16: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_17: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_45)
    add_19: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_46);  mul_17 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_65: "f32[128, 768]" = torch.ops.aten.reshape.default(add_19, [128, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_16: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_48, view_65, permute_31);  primals_48 = None
    view_66: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_18: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
    mul_19: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476);  view_66 = None
    erf_2: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_67: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_20, [128, 3072]);  mul_20 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_17: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_50, view_67, permute_32);  primals_50 = None
    view_68: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 768]);  addmm_17 = None
    
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
    sub_9: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_27);  add_21 = getitem_27 = None
    mul_21: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_22: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_51)
    add_23: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_52);  mul_22 = primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_69: "f32[128, 768]" = torch.ops.aten.reshape.default(add_23, [128, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_18: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_54, view_69, permute_33);  primals_54 = None
    view_70: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_71: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_70, [1, -1, 12, 64]);  view_70 = None
    permute_34: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_19: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_56, view_69, permute_35);  primals_56 = None
    view_73: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_74: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_73, [1, -1, 12, 64]);  view_73 = None
    permute_36: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_20: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_58, view_69, permute_37);  primals_58 = None
    view_76: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_77: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_76, [1, -1, 12, 64]);  view_76 = None
    permute_38: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_6: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_34, 8.0);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_39: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
    expand_15: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_6, [1, 12, 128, 64]);  div_6 = None
    view_78: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_15, [12, 128, 64]);  expand_15 = None
    expand_16: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_39, [1, 12, 64, 128]);  permute_39 = None
    view_79: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_16, [12, 64, 128]);  expand_16 = None
    bmm_6: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_3: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_80);  view_80 = None
    
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
    view_82: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_18, [12, 128, 128]);  expand_18 = None
    expand_19: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_38, [1, 12, 128, 64]);  permute_38 = None
    view_83: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_19, [12, 128, 64]);  expand_19 = None
    bmm_7: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_40: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    clone_3: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_85: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_3, [1, -1, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_86: "f32[128, 768]" = torch.ops.aten.reshape.default(view_85, [128, 768]);  view_85 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_21: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_60, view_86, permute_41);  primals_60 = None
    view_87: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_24: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_87, add_23);  view_87 = add_23 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_11: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_31);  add_24 = getitem_31 = None
    mul_23: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_24: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_61)
    add_26: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_62);  mul_24 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_88: "f32[128, 768]" = torch.ops.aten.reshape.default(add_26, [128, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    addmm_22: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_64, view_88, permute_42);  primals_64 = None
    view_89: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_25: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    mul_26: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.7071067811865476);  view_89 = None
    erf_3: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_27: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_27: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_90: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_27, [128, 3072]);  mul_27 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    addmm_23: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_66, view_90, permute_43);  primals_66 = None
    view_91: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 768]);  addmm_23 = None
    
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
    sub_12: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_35);  add_28 = getitem_35 = None
    mul_28: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_29: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_67)
    add_30: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_68);  mul_29 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_92: "f32[128, 768]" = torch.ops.aten.reshape.default(add_30, [128, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    addmm_24: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_70, view_92, permute_44);  primals_70 = None
    view_93: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_94: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_93, [1, -1, 12, 64]);  view_93 = None
    permute_45: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_25: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_72, view_92, permute_46);  primals_72 = None
    view_96: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_97: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_96, [1, -1, 12, 64]);  view_96 = None
    permute_47: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_26: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_74, view_92, permute_48);  primals_74 = None
    view_99: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_100: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_99, [1, -1, 12, 64]);  view_99 = None
    permute_49: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_8: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_45, 8.0);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_50: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
    expand_20: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_8, [1, 12, 128, 64]);  div_8 = None
    view_101: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_20, [12, 128, 64]);  expand_20 = None
    expand_21: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_50, [1, 12, 64, 128]);  permute_50 = None
    view_102: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_21, [12, 64, 128]);  expand_21 = None
    bmm_8: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_101, view_102)
    view_103: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_4: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_103);  view_103 = None
    
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
    view_105: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_23, [12, 128, 128]);  expand_23 = None
    expand_24: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_49, [1, 12, 128, 64]);  permute_49 = None
    view_106: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_24, [12, 128, 64]);  expand_24 = None
    bmm_9: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_105, view_106)
    view_107: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_51: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    clone_4: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_108: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_4, [1, -1, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_109: "f32[128, 768]" = torch.ops.aten.reshape.default(view_108, [128, 768]);  view_108 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_27: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_76, view_109, permute_52);  primals_76 = None
    view_110: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_31: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_110, add_30);  view_110 = add_30 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_14: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_39);  add_31 = getitem_39 = None
    mul_30: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_31: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_77)
    add_33: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_78);  mul_31 = primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_111: "f32[128, 768]" = torch.ops.aten.reshape.default(add_33, [128, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_28: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_80, view_111, permute_53);  primals_80 = None
    view_112: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_32: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.5)
    mul_33: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.7071067811865476);  view_112 = None
    erf_4: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_34: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_34: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_113: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_34, [128, 3072]);  mul_34 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_29: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_82, view_113, permute_54);  primals_82 = None
    view_114: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 768]);  addmm_29 = None
    
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
    sub_15: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_43);  add_35 = getitem_43 = None
    mul_35: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_36: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_83)
    add_37: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_84);  mul_36 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_115: "f32[128, 768]" = torch.ops.aten.reshape.default(add_37, [128, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_30: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_86, view_115, permute_55);  primals_86 = None
    view_116: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_117: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_116, [1, -1, 12, 64]);  view_116 = None
    permute_56: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_31: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_88, view_115, permute_57);  primals_88 = None
    view_119: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_120: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_119, [1, -1, 12, 64]);  view_119 = None
    permute_58: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_32: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_90, view_115, permute_59);  primals_90 = None
    view_122: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    view_123: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_122, [1, -1, 12, 64]);  view_122 = None
    permute_60: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_10: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(permute_56, 8.0);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_61: "f32[1, 12, 64, 128]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
    expand_25: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(div_10, [1, 12, 128, 64]);  div_10 = None
    view_124: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_25, [12, 128, 64]);  expand_25 = None
    expand_26: "f32[1, 12, 64, 128]" = torch.ops.aten.expand.default(permute_61, [1, 12, 64, 128]);  permute_61 = None
    view_125: "f32[12, 64, 128]" = torch.ops.aten.reshape.default(expand_26, [12, 64, 128]);  expand_26 = None
    bmm_10: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_124, view_125)
    view_126: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_5: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default, view_126);  expand_2 = full_default = view_126 = None
    
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
    view_128: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(expand_28, [12, 128, 128]);  expand_28 = None
    expand_29: "f32[1, 12, 128, 64]" = torch.ops.aten.expand.default(permute_60, [1, 12, 128, 64]);  permute_60 = None
    view_129: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(expand_29, [12, 128, 64]);  expand_29 = None
    bmm_11: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_128, view_129)
    view_130: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    permute_62: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    clone_5: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_131: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_5, [1, -1, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_132: "f32[128, 768]" = torch.ops.aten.reshape.default(view_131, [128, 768]);  view_131 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_33: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_92, view_132, permute_63);  primals_92 = None
    view_133: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    add_38: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(view_133, add_37);  view_133 = add_37 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_17: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_47);  add_38 = getitem_47 = None
    mul_37: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_38: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_93)
    add_40: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_94);  mul_38 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_134: "f32[128, 768]" = torch.ops.aten.reshape.default(add_40, [128, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_34: "f32[128, 3072]" = torch.ops.aten.addmm.default(primals_96, view_134, permute_64);  primals_96 = None
    view_135: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_39: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, 0.5)
    mul_40: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476);  view_135 = None
    erf_5: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_41: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_41: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_136: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_41, [128, 3072]);  mul_41 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_35: "f32[128, 768]" = torch.ops.aten.addmm.default(primals_98, view_136, permute_65);  primals_98 = None
    view_137: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 768]);  addmm_35 = None
    
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
    sub_18: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_51);  add_42 = getitem_51 = None
    mul_42: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_43: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_99)
    add_44: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_100);  mul_43 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:922, code: hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
    native_dropout_13 = torch.ops.aten.native_dropout.default(add_44, 0.1, True);  add_44 = None
    getitem_52: "f32[1, 128, 768]" = native_dropout_13[0]
    getitem_53: "b8[1, 128, 768]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:923, code: logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
    view_138: "f32[128, 768]" = torch.ops.aten.reshape.default(getitem_52, [128, 768]);  getitem_52 = None
    permute_66: "f32[768, 2]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_36: "f32[128, 2]" = torch.ops.aten.addmm.default(primals_102, view_138, permute_66);  primals_102 = None
    view_139: "f32[1, 128, 2]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 2]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:924, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_139, [1, 1], 2);  view_139 = None
    getitem_54: "f32[1, 128, 1]" = split_with_sizes[0]
    getitem_55: "f32[1, 128, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_54, -1);  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:925, code: start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
    clone_6: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_55, -1);  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:926, code: end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
    clone_7: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:937, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(primals_105, 0);  primals_105 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:938, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_106, 0);  primals_106 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:941, code: start_loss = loss_fct(start_logits, start_positions)
    amax_6: "f32[1, 1]" = torch.ops.aten.amax.default(clone_6, [1], True)
    sub_19: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_6, amax_6);  amax_6 = None
    exp_6: "f32[1, 128]" = torch.ops.aten.exp.default(sub_19)
    sum_7: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
    sub_20: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_19, log);  sub_19 = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128)
    full_default_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_6: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, full_default_6)
    unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_20, 1, unsqueeze);  unsqueeze = None
    squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
    full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_7: "f32[1]" = torch.ops.aten.where.self(ne, neg, full_default_7);  neg = None
    sum_8: "i64[]" = torch.ops.aten.sum.default(ne)
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
    sum_9: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
    div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type);  sum_9 = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:942, code: end_loss = loss_fct(end_logits, end_positions)
    amax_7: "f32[1, 1]" = torch.ops.aten.amax.default(clone_7, [1], True)
    sub_21: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_7, amax_7);  amax_7 = None
    exp_7: "f32[1, 128]" = torch.ops.aten.exp.default(sub_21)
    sum_10: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [1], True);  exp_7 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_10);  sum_10 = None
    sub_22: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_21, log_1);  sub_21 = log_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
    where_8: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_6)
    unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_8, 1);  where_8 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_22, 1, unsqueeze_1);  unsqueeze_1 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    where_9: "f32[1]" = torch.ops.aten.where.self(ne_3, neg_1, full_default_7);  neg_1 = full_default_7 = None
    sum_11: "i64[]" = torch.ops.aten.sum.default(ne_3)
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_11, torch.float32);  sum_11 = None
    sum_12: "f32[]" = torch.ops.aten.sum.default(where_9);  where_9 = None
    div_13: "f32[]" = torch.ops.aten.div.Tensor(sum_12, convert_element_type_1);  sum_12 = convert_element_type_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:943, code: total_loss = (start_loss + end_loss) / 2
    add_45: "f32[]" = torch.ops.aten.add.Tensor(div_12, div_13);  div_12 = div_13 = None
    div_14: "f32[]" = torch.ops.aten.div.Tensor(add_45, 2);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:942, code: end_loss = loss_fct(end_logits, end_positions)
    unsqueeze_2: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
    ne_6: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_2, 128)
    where_10: "i64[1, 1]" = torch.ops.aten.where.self(ne_6, unsqueeze_2, full_default_6);  unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:941, code: start_loss = loss_fct(start_logits, start_positions)
    unsqueeze_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
    ne_8: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, 128)
    where_12: "i64[1, 1]" = torch.ops.aten.where.self(ne_8, unsqueeze_3, full_default_6);  unsqueeze_3 = full_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:923, code: logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
    permute_67: "f32[2, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    div_18: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    permute_75: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    div_19: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    permute_84: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
    permute_85: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_10: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_86: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    permute_87: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_95: "f32[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    div_21: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    permute_104: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    permute_108: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    div_22: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    permute_117: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_105, [0, 2, 1]);  view_105 = None
    permute_118: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_11: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_119: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    permute_120: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    div_24: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    permute_137: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    permute_141: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    div_25: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    permute_145: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    permute_150: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    permute_151: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_12: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_152: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    permute_153: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_156: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    div_27: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    permute_170: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    permute_174: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    div_28: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    permute_183: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    permute_184: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_13: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_185: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    permute_186: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_189: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    div_30: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    permute_203: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    permute_207: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    div_31: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    permute_211: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    permute_216: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    permute_217: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_14: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_218: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    permute_219: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_222: "f32[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    div_33: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    permute_236: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    permute_240: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    div_34: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    permute_244: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    permute_249: "f32[12, 128, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    permute_250: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    alias_15: "f32[1, 12, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    permute_251: "f32[12, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_252: "f32[12, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    permute_255: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:136, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
    div_36: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [div_14, clone_6, clone_7, primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_104, slice_2, mul, getitem_3, view, view_12, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_7, view_23, getitem_13, view_40, mul_9, view_42, addmm_10, view_44, getitem_17, mul_14, view_46, getitem_21, view_63, mul_16, view_65, addmm_16, view_67, getitem_25, mul_21, view_69, getitem_29, view_86, mul_23, view_88, addmm_22, view_90, getitem_33, mul_28, view_92, getitem_37, view_109, mul_30, view_111, addmm_28, view_113, getitem_41, mul_35, view_115, getitem_45, view_132, mul_37, view_134, addmm_34, view_136, getitem_49, mul_42, getitem_53, view_138, sub_20, ne, sub_22, ne_3, ne_6, where_10, ne_8, where_12, permute_67, div_18, permute_71, permute_75, div_19, permute_79, permute_84, permute_85, alias_10, permute_86, permute_87, permute_90, permute_95, permute_100, div_21, permute_104, permute_108, div_22, permute_112, permute_117, permute_118, alias_11, permute_119, permute_120, permute_123, permute_128, permute_133, div_24, permute_137, permute_141, div_25, permute_145, permute_150, permute_151, alias_12, permute_152, permute_153, permute_156, permute_161, permute_166, div_27, permute_170, permute_174, div_28, permute_178, permute_183, permute_184, alias_13, permute_185, permute_186, permute_189, permute_194, permute_199, div_30, permute_203, permute_207, div_31, permute_211, permute_216, permute_217, alias_14, permute_218, permute_219, permute_222, permute_227, permute_232, div_33, permute_236, permute_240, div_34, permute_244, permute_249, permute_250, alias_15, permute_251, permute_252, permute_255, permute_260, permute_265, div_36]
    