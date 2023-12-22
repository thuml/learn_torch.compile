from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[768]"; primals_2: "f32[768]"; primals_3: "f32[768, 768]"; primals_4: "f32[768]"; primals_5: "f32[768, 768]"; primals_6: "f32[768]"; primals_7: "f32[768, 768]"; primals_8: "f32[768]"; primals_9: "f32[768, 768]"; primals_10: "f32[768]"; primals_11: "f32[768]"; primals_12: "f32[768]"; primals_13: "f32[3072, 768]"; primals_14: "f32[3072]"; primals_15: "f32[768, 3072]"; primals_16: "f32[768]"; primals_17: "f32[1, 2048, 768]"; primals_18: "f32[1, 1, 2048, 2048]"; tangents_1: "f32[1, 2048, 768]"; tangents_2: "f32[1, 12, 2048, 64]"; tangents_3: "f32[1, 12, 2048, 64]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, tangents_1, tangents_2, tangents_3, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(primals_17, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 2048, 1]" = var_mean[0]
    getitem_1: "f32[1, 2048, 1]" = var_mean[1];  var_mean = None
    add: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(primals_17, getitem_1)
    mul: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
    add_1: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[2048, 768]" = torch.ops.aten.view.default(add_1, [2048, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_4, view, permute);  primals_4 = None
    view_1: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm, [1, 2048, 768]);  addmm = None
    mul_2: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_2: "f32[2048, 768]" = torch.ops.aten.view.default(add_1, [2048, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_1: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_6, view_2, permute_1);  primals_6 = None
    view_3: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_1, [1, 2048, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_3, [1, -1, 12, 64]);  view_3 = None
    permute_2: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_5: "f32[2048, 768]" = torch.ops.aten.view.default(add_1, [2048, 768]);  add_1 = None
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_2: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_8, view_5, permute_3);  primals_8 = None
    view_6: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_2, [1, 2048, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_6, [1, -1, 12, 64]);  view_6 = None
    permute_4: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_2, [1, 2048, 12, 64]);  mul_2 = None
    permute_5: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_9: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_2, [12, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_10: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_11: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_1, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_9, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_12: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm, [1, 12, 2048, 2048]);  bmm = None
    add_2: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_12, primals_18);  view_12 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_2, lift_fresh_copy)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_13: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum, [12, 2048, 2048]);  maximum = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_13, [-1], True)
    sub_1: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_13, amax);  view_13 = amax = None
    exp: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[12, 2048, 2048]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_3: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_3, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_14: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 2048, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_4: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_15: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_4, [1, 2048, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_16: "f32[2048, 768]" = torch.ops.aten.view.default(view_15, [2048, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_3: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_10, view_16, permute_8);  primals_10 = None
    view_17: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_3, [1, 2048, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_2: "f32[1, 2048, 768]" = native_dropout[0]
    getitem_3: "b8[1, 2048, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_3: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(primals_17, getitem_2);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_18: "f32[2048, 768]" = torch.ops.aten.view.default(add_3, [-1, 768]);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(view_18, [1], correction = 0, keepdim = True)
    getitem_4: "f32[2048, 1]" = var_mean_1[0]
    getitem_5: "f32[2048, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_1: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_18, getitem_5)
    mul_3: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_4: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_11);  mul_3 = None
    add_5: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_12);  mul_4 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_4: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_14, add_5, permute_9);  primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None
    alias_1: "f32[2048, 3072]" = torch.ops.aten.alias.default(relu)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_5: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_16, relu, permute_10);  primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_1 = torch.ops.aten.native_dropout.default(addmm_5, 0.1, True);  addmm_5 = None
    getitem_6: "f32[2048, 768]" = native_dropout_1[0]
    getitem_7: "b8[2048, 768]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_6: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_18, getitem_6);  getitem_6 = None
    view_19: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_6, [1, 2048, 768]);  add_6 = None
    view_20: "f32[2048, 768]" = torch.ops.aten.view.default(tangents_1, [2048, 768]);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[2048, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_5: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_6: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(view_20, mul_5);  mul_5 = None
    clone_5: "f32[2048, 768]" = torch.ops.aten.clone.default(mul_6, memory_format = torch.contiguous_format);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm: "f32[2048, 3072]" = torch.ops.aten.mm.default(clone_5, permute_11);  permute_11 = None
    permute_12: "f32[768, 2048]" = torch.ops.aten.permute.default(clone_5, [1, 0])
    mm_1: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_12, relu);  permute_12 = relu = None
    permute_13: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_2: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(clone_5, [0], True);  clone_5 = None
    view_21: "f32[768]" = torch.ops.aten.view.default(sum_2, [768]);  sum_2 = None
    permute_14: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    alias_2: "f32[2048, 3072]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    le: "b8[2048, 3072]" = torch.ops.aten.le.Scalar(alias_2, 0);  alias_2 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[2048, 3072]" = torch.ops.aten.where.self(le, scalar_tensor, mm);  le = scalar_tensor = mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_15: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_2: "f32[2048, 768]" = torch.ops.aten.mm.default(where, permute_15);  permute_15 = None
    permute_16: "f32[3072, 2048]" = torch.ops.aten.permute.default(where, [1, 0])
    mm_3: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_16, add_5);  permute_16 = add_5 = None
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_3: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
    view_22: "f32[3072]" = torch.ops.aten.view.default(sum_3, [3072]);  sum_3 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_3: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_18, getitem_5);  view_18 = getitem_5 = None
    mul_7: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_8: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mm_2, primals_11);  primals_11 = None
    mul_9: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_8, 768)
    sum_4: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [1], True)
    mul_10: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_8, mul_7);  mul_8 = None
    sum_5: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_10, [1], True);  mul_10 = None
    mul_11: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_5);  sum_5 = None
    sub_4: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(mul_9, sum_4);  mul_9 = sum_4 = None
    sub_5: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(sub_4, mul_11);  sub_4 = mul_11 = None
    div_1: "f32[2048, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_12: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
    mul_13: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mm_2, mul_7);  mul_7 = None
    sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_13, [0]);  mul_13 = None
    sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(mm_2, [0]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    add_7: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_20, mul_12);  view_20 = mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_23: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_7, [1, 2048, 768]);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_14: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_15: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_23, mul_14);  mul_14 = None
    clone_6: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(mul_15, memory_format = torch.contiguous_format);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_24: "f32[2048, 768]" = torch.ops.aten.view.default(clone_6, [2048, 768]);  clone_6 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_4: "f32[2048, 768]" = torch.ops.aten.mm.default(view_24, permute_19);  permute_19 = None
    permute_20: "f32[768, 2048]" = torch.ops.aten.permute.default(view_24, [1, 0])
    mm_5: "f32[768, 768]" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
    permute_21: "f32[768, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_24, [0], True);  view_24 = None
    view_25: "f32[768]" = torch.ops.aten.view.default(sum_8, [768]);  sum_8 = None
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    view_26: "f32[1, 2048, 768]" = torch.ops.aten.view.default(mm_4, [1, 2048, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_27: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_26, [1, 2048, 12, 64]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_23: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_28: "f32[12, 2048, 64]" = torch.ops.aten.view.default(permute_23, [12, 2048, 64]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_24: "f32[12, 2048, 2048]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    bmm_2: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(permute_24, view_28);  permute_24 = None
    permute_25: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_3: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_28, permute_25);  view_28 = permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_3: "f32[12, 2048, 2048]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_16: "f32[12, 2048, 2048]" = torch.ops.aten.mul.Tensor(bmm_3, alias_3);  bmm_3 = None
    sum_9: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_16, [-1], True)
    mul_17: "f32[12, 2048, 2048]" = torch.ops.aten.mul.Tensor(alias_3, sum_9);  alias_3 = sum_9 = None
    sub_6: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_29: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(sub_6, [1, 12, 2048, 2048]);  sub_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    div_2: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Scalar(view_29, 2)
    eq: "b8[1, 12, 2048, 2048]" = torch.ops.aten.eq.Tensor(add_2, lift_fresh_copy)
    where_1: "f32[1, 12, 2048, 2048]" = torch.ops.aten.where.self(eq, div_2, view_29);  eq = div_2 = view_29 = None
    lt: "b8[1, 12, 2048, 2048]" = torch.ops.aten.lt.Tensor(add_2, lift_fresh_copy);  add_2 = lift_fresh_copy = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[1, 12, 2048, 2048]" = torch.ops.aten.where.self(lt, scalar_tensor_1, where_1);  lt = scalar_tensor_1 = where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_26: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    view_31: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(where_2, [12, 2048, 2048]);  where_2 = None
    bmm_4: "f32[12, 64, 2048]" = torch.ops.aten.bmm.default(permute_26, view_31);  permute_26 = None
    permute_27: "f32[12, 2048, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    bmm_5: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(view_31, permute_27);  view_31 = permute_27 = None
    permute_28: "f32[12, 2048, 64]" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_32: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_2, [1, 12, 2048, 64]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    add_8: "f32[1, 12, 2048, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_32);  tangents_3 = view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_33: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(permute_28, [1, 12, 2048, 64]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    add_9: "f32[1, 12, 2048, 64]" = torch.ops.aten.add.Tensor(tangents_2, view_33);  tangents_2 = view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_34: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 2048, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_29: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
    clone_7: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_35: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_7, [1, 2048, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_30: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(add_8, [0, 2, 1, 3]);  add_8 = None
    clone_8: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_36: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_8, [1, 2048, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_37: "f32[2048, 768]" = torch.ops.aten.view.default(view_36, [2048, 768]);  view_36 = None
    permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_37, permute_31);  permute_31 = None
    permute_32: "f32[768, 2048]" = torch.ops.aten.permute.default(view_37, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_32, view_5);  permute_32 = view_5 = None
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True);  view_37 = None
    view_38: "f32[768]" = torch.ops.aten.view.default(sum_10, [768]);  sum_10 = None
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    view_39: "f32[1, 2048, 768]" = torch.ops.aten.view.default(mm_6, [1, 2048, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_35: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(add_9, [0, 2, 1, 3]);  add_9 = None
    clone_9: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    view_40: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_9, [1, 2048, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_41: "f32[2048, 768]" = torch.ops.aten.view.default(view_40, [2048, 768]);  view_40 = None
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_8: "f32[2048, 768]" = torch.ops.aten.mm.default(view_41, permute_36);  permute_36 = None
    permute_37: "f32[768, 2048]" = torch.ops.aten.permute.default(view_41, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_37, view_2);  permute_37 = view_2 = None
    permute_38: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
    view_42: "f32[768]" = torch.ops.aten.view.default(sum_11, [768]);  sum_11 = None
    permute_39: "f32[768, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_43: "f32[1, 2048, 768]" = torch.ops.aten.view.default(mm_8, [1, 2048, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_10: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_39, view_43);  view_39 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_18: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_35, 0.125);  view_35 = None
    view_44: "f32[2048, 768]" = torch.ops.aten.view.default(mul_18, [2048, 768]);  mul_18 = None
    permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_44, permute_40);  permute_40 = None
    permute_41: "f32[768, 2048]" = torch.ops.aten.permute.default(view_44, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True);  view_44 = None
    view_45: "f32[768]" = torch.ops.aten.view.default(sum_12, [768]);  sum_12 = None
    permute_43: "f32[768, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_46: "f32[1, 2048, 768]" = torch.ops.aten.view.default(mm_10, [1, 2048, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_11: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(add_10, view_46);  add_10 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_7: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(primals_17, getitem_1);  primals_17 = getitem_1 = None
    mul_19: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt);  sub_7 = None
    mul_20: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(add_11, primals_1);  primals_1 = None
    mul_21: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_20, 768)
    sum_13: "f32[1, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True)
    mul_22: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_20, mul_19);  mul_20 = None
    sum_14: "f32[1, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True);  mul_22 = None
    mul_23: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_19, sum_14);  sum_14 = None
    sub_8: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(mul_21, sum_13);  mul_21 = sum_13 = None
    sub_9: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(sub_8, mul_23);  sub_8 = mul_23 = None
    div_3: "f32[1, 2048, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_24: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_9);  div_3 = sub_9 = None
    mul_25: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(add_11, mul_19);  mul_19 = None
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_25, [0, 1]);  mul_25 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_12: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_23, mul_24);  view_23 = mul_24 = None
    return pytree.tree_unflatten([view_19, clone, clone_1, sum_15, sum_16, permute_43, view_45, permute_39, view_42, permute_34, view_38, permute_22, view_25, sum_6, sum_7, permute_18, view_22, permute_14, view_21, add_12, None], self._out_spec)
    