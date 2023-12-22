from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1024]"; primals_2: "f32[1024]"; primals_3: "f32[1024, 1024]"; primals_4: "f32[1024]"; primals_5: "f32[1024, 1024]"; primals_6: "f32[1024]"; primals_7: "f32[1024, 1024]"; primals_8: "f32[1024]"; primals_9: "f32[1024, 1024]"; primals_10: "f32[1024]"; primals_11: "f32[1024]"; primals_12: "f32[1024]"; primals_13: "f32[1024, 1024]"; primals_14: "f32[1024]"; primals_15: "f32[1024, 1024]"; primals_16: "f32[1024]"; primals_17: "f32[1024, 1024]"; primals_18: "f32[1024]"; primals_19: "f32[1024, 1024]"; primals_20: "f32[1024]"; primals_21: "f32[1024]"; primals_22: "f32[1024]"; primals_23: "f32[4096, 1024]"; primals_24: "f32[4096]"; primals_25: "f32[1024, 4096]"; primals_26: "f32[1024]"; primals_27: "f32[1, 128, 1024]"; primals_28: "f32[1, 1, 128, 128]"; primals_29: "f32[1, 128, 1024]"; tangents_1: "f32[1, 128, 1024]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(primals_27, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_27, getitem_1)
    mul: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
    add_1: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[128, 1024]" = torch.ops.aten.view.default(add_1, [128, 1024])
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_4, view, permute);  primals_4 = None
    view_1: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm, [1, 128, 1024]);  addmm = None
    mul_2: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_2: "f32[128, 1024]" = torch.ops.aten.view.default(add_1, [128, 1024])
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_1: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_6, view_2, permute_1);  primals_6 = None
    view_3: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_1, [1, 128, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_3, [1, -1, 16, 64]);  view_3 = None
    permute_2: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_5: "f32[128, 1024]" = torch.ops.aten.view.default(add_1, [128, 1024]);  add_1 = None
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_2: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_8, view_5, permute_3);  primals_8 = None
    view_6: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_2, [1, 128, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_6, [1, -1, 16, 64]);  view_6 = None
    permute_4: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_2, [1, 128, 16, 64]);  mul_2 = None
    permute_5: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_9: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_2, [16, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_10: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone, [16, -1, 64]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_11: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_1, [16, -1, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_9, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_12: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, [1, 16, 128, 128]);  bmm = None
    add_2: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_12, primals_28);  view_12 = primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_13: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_2, [16, 128, 128]);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_13, [-1], True)
    sub_1: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_13, amax);  view_13 = amax = None
    exp: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:276, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_3: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_3, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_14: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, [1, 16, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_4: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_15: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_4, [1, 128, 1024]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_16: "f32[128, 1024]" = torch.ops.aten.view.default(view_15, [128, 1024]);  view_15 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_3: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_10, view_16, permute_8);  primals_10 = None
    view_17: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_3, [1, 128, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:439, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_2: "f32[1, 128, 1024]" = native_dropout[0]
    getitem_3: "b8[1, 128, 1024]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:440, code: hidden_states = residual + hidden_states
    add_3: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(primals_27, getitem_2);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_3, getitem_5)
    mul_3: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_4: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_3, primals_11);  mul_3 = None
    add_5: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_4, primals_12);  mul_4 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_18: "f32[128, 1024]" = torch.ops.aten.view.default(add_5, [128, 1024]);  add_5 = None
    permute_9: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_4: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_14, view_18, permute_9);  primals_14 = None
    view_19: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_4, [1, 128, 1024]);  addmm_4 = None
    mul_5: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_19, 0.125);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_20: "f32[128, 1024]" = torch.ops.aten.view.default(primals_29, [128, 1024])
    permute_10: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_5: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_16, view_20, permute_10);  primals_16 = None
    view_21: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 128, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_22: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_21, [1, -1, 16, 64]);  view_21 = None
    permute_11: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    clone_5: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_23: "f32[128, 1024]" = torch.ops.aten.view.default(primals_29, [128, 1024]);  primals_29 = None
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_6: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_18, view_23, permute_12);  primals_18 = None
    view_24: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_6, [1, 128, 1024]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_24, [1, -1, 16, 64]);  view_24 = None
    permute_13: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_6: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_26: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_5, [1, 128, 16, 64]);  mul_5 = None
    permute_14: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    clone_7: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_27: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_7, [16, -1, 64]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_28: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_5, [16, -1, 64]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_29: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_6, [16, -1, 64]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_15: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_27, permute_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_2, [-1], True)
    sub_3: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1);  bmm_2 = amax_1 = None
    exp_1: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:276, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_8: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_8, view_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_30: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_3, [1, 16, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_16: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_9: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_31: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_9, [1, 128, 1024]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_32: "f32[128, 1024]" = torch.ops.aten.view.default(view_31, [128, 1024]);  view_31 = None
    permute_17: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    addmm_7: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_20, view_32, permute_17);  primals_20 = None
    view_33: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_7, [1, 128, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:459, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_33, 0.1, True);  view_33 = None
    getitem_6: "f32[1, 128, 1024]" = native_dropout_1[0]
    getitem_7: "b8[1, 128, 1024]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:460, code: hidden_states = residual + hidden_states
    add_6: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_3, getitem_6);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_4: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_9)
    mul_6: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_7: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_6, primals_21);  mul_6 = None
    add_8: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_7, primals_22);  mul_7 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_34: "f32[128, 1024]" = torch.ops.aten.view.default(add_8, [128, 1024]);  add_8 = None
    permute_18: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_8: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_24, view_34, permute_18);  primals_24 = None
    view_35: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_8, [1, 128, 4096]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_8: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    mul_9: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476)
    erf: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_9: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_10: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_8, add_9);  mul_8 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:469, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_10: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_10);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_36: "f32[128, 4096]" = torch.ops.aten.view.default(clone_10, [128, 4096]);  clone_10 = None
    permute_19: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_9: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_26, view_36, permute_19);  primals_26 = None
    view_37: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_9, [1, 128, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:471, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_37, 0.1, True);  view_37 = None
    getitem_10: "f32[1, 128, 1024]" = native_dropout_2[0]
    getitem_11: "b8[1, 128, 1024]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:472, code: hidden_states = residual + hidden_states
    add_10: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_6, getitem_10);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:471, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_11: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_12: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_11);  mul_11 = None
    clone_11: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(mul_12, memory_format = torch.contiguous_format);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    view_38: "f32[128, 1024]" = torch.ops.aten.view.default(clone_11, [128, 1024]);  clone_11 = None
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm: "f32[128, 4096]" = torch.ops.aten.mm.default(view_38, permute_20);  permute_20 = None
    permute_21: "f32[1024, 128]" = torch.ops.aten.permute.default(view_38, [1, 0])
    mm_1: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_21, view_36);  permute_21 = view_36 = None
    permute_22: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_38, [0], True);  view_38 = None
    view_39: "f32[1024]" = torch.ops.aten.view.default(sum_3, [1024]);  sum_3 = None
    permute_23: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    view_40: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm, [1, 128, 4096]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_13: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476)
    erf_1: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_11: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_11, 0.5);  add_11 = None
    mul_15: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_35, view_35)
    mul_16: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_15, -0.5);  mul_15 = None
    exp_2: "f32[1, 128, 4096]" = torch.ops.aten.exp.default(mul_16);  mul_16 = None
    mul_17: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_18: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_35, mul_17);  view_35 = mul_17 = None
    add_12: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_14, mul_18);  mul_14 = mul_18 = None
    mul_19: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_40, add_12);  view_40 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_41: "f32[128, 4096]" = torch.ops.aten.view.default(mul_19, [128, 4096]);  mul_19 = None
    permute_24: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_2: "f32[128, 1024]" = torch.ops.aten.mm.default(view_41, permute_24);  permute_24 = None
    permute_25: "f32[4096, 128]" = torch.ops.aten.permute.default(view_41, [1, 0])
    mm_3: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_25, view_34);  permute_25 = view_34 = None
    permute_26: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
    view_42: "f32[4096]" = torch.ops.aten.view.default(sum_4, [4096]);  sum_4 = None
    permute_27: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    view_43: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_2, [1, 128, 1024]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_5: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_9);  add_6 = getitem_9 = None
    mul_20: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = None
    mul_21: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_43, primals_21);  primals_21 = None
    mul_22: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_21, 1024)
    sum_5: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_21, [2], True)
    mul_23: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_21, mul_20);  mul_21 = None
    sum_6: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_23, [2], True);  mul_23 = None
    mul_24: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, sum_6);  sum_6 = None
    sub_6: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_22, sum_5);  mul_22 = sum_5 = None
    sub_7: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_6, mul_24);  sub_6 = mul_24 = None
    div_2: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None
    mul_25: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_2, sub_7);  div_2 = sub_7 = None
    mul_26: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_43, mul_20);  mul_20 = None
    sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_26, [0, 1]);  mul_26 = None
    sum_8: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_43, [0, 1]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    add_13: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tangents_1, mul_25);  tangents_1 = mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:459, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_27: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_28: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_13, mul_27);  mul_27 = None
    clone_12: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(mul_28, memory_format = torch.contiguous_format);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_44: "f32[128, 1024]" = torch.ops.aten.view.default(clone_12, [128, 1024]);  clone_12 = None
    permute_28: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_4: "f32[128, 1024]" = torch.ops.aten.mm.default(view_44, permute_28);  permute_28 = None
    permute_29: "f32[1024, 128]" = torch.ops.aten.permute.default(view_44, [1, 0])
    mm_5: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_29, view_32);  permute_29 = view_32 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_9: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True);  view_44 = None
    view_45: "f32[1024]" = torch.ops.aten.view.default(sum_9, [1024]);  sum_9 = None
    permute_31: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    view_46: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_4, [1, 128, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_47: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_46, [1, 128, 16, 64]);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_32: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_48: "f32[16, 128, 64]" = torch.ops.aten.view.default(permute_32, [16, 128, 64]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_33: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_8, [0, 2, 1]);  clone_8 = None
    bmm_4: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_33, view_48);  permute_33 = None
    permute_34: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    bmm_5: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_48, permute_34);  view_48 = permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_2: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_29: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_5, alias_2);  bmm_5 = None
    sum_10: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [-1], True)
    mul_30: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_2, sum_10);  alias_2 = sum_10 = None
    sub_8: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_35: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    bmm_6: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_35, sub_8);  permute_35 = None
    permute_36: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_15, [0, 2, 1]);  permute_15 = None
    bmm_7: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_8, permute_36);  sub_8 = permute_36 = None
    permute_37: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_6, [0, 2, 1]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_49: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_4, [1, 16, 128, 64]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_50: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(permute_37, [1, 16, 128, 64]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_51: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_7, [1, 16, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_38: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    clone_13: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_52: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_13, [1, 128, 1024]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_39: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_14: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    view_53: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_14, [1, 128, 1024]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_54: "f32[128, 1024]" = torch.ops.aten.view.default(view_53, [128, 1024]);  view_53 = None
    permute_40: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_6: "f32[128, 1024]" = torch.ops.aten.mm.default(view_54, permute_40);  permute_40 = None
    permute_41: "f32[1024, 128]" = torch.ops.aten.permute.default(view_54, [1, 0])
    mm_7: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_41, view_23);  permute_41 = view_23 = None
    permute_42: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_54, [0], True);  view_54 = None
    view_55: "f32[1024]" = torch.ops.aten.view.default(sum_11, [1024]);  sum_11 = None
    permute_43: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_56: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_6, [1, 128, 1024]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_44: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    view_57: "f32[1, 128, 1024]" = torch.ops.aten.view.default(permute_44, [1, 128, 1024]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_58: "f32[128, 1024]" = torch.ops.aten.view.default(view_57, [128, 1024]);  view_57 = None
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_8: "f32[128, 1024]" = torch.ops.aten.mm.default(view_58, permute_45);  permute_45 = None
    permute_46: "f32[1024, 128]" = torch.ops.aten.permute.default(view_58, [1, 0])
    mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_46, view_20);  permute_46 = view_20 = None
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_58, [0], True);  view_58 = None
    view_59: "f32[1024]" = torch.ops.aten.view.default(sum_12, [1024]);  sum_12 = None
    permute_48: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    view_60: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_8, [1, 128, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_14: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_56, view_60);  view_56 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_31: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_52, 0.125);  view_52 = None
    view_61: "f32[128, 1024]" = torch.ops.aten.view.default(mul_31, [128, 1024]);  mul_31 = None
    permute_49: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_10: "f32[128, 1024]" = torch.ops.aten.mm.default(view_61, permute_49);  permute_49 = None
    permute_50: "f32[1024, 128]" = torch.ops.aten.permute.default(view_61, [1, 0])
    mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_50, view_18);  permute_50 = view_18 = None
    permute_51: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_13: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_61, [0], True);  view_61 = None
    view_62: "f32[1024]" = torch.ops.aten.view.default(sum_13, [1024]);  sum_13 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    view_63: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_10, [1, 128, 1024]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    sub_9: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_3, getitem_5);  add_3 = getitem_5 = None
    mul_32: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_1);  sub_9 = None
    mul_33: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_63, primals_11);  primals_11 = None
    mul_34: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_33, 1024)
    sum_14: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_33, [2], True)
    mul_35: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_33, mul_32);  mul_33 = None
    sum_15: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True);  mul_35 = None
    mul_36: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_32, sum_15);  sum_15 = None
    sub_10: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_34, sum_14);  mul_34 = sum_14 = None
    sub_11: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_10, mul_36);  sub_10 = mul_36 = None
    div_3: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
    mul_37: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_3, sub_11);  div_3 = sub_11 = None
    mul_38: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_63, mul_32);  mul_32 = None
    sum_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 1]);  mul_38 = None
    sum_17: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_63, [0, 1]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_15: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_13, mul_37);  add_13 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:439, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_2: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_39: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_40: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_15, mul_39);  mul_39 = None
    clone_15: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(mul_40, memory_format = torch.contiguous_format);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    view_64: "f32[128, 1024]" = torch.ops.aten.view.default(clone_15, [128, 1024]);  clone_15 = None
    permute_53: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_12: "f32[128, 1024]" = torch.ops.aten.mm.default(view_64, permute_53);  permute_53 = None
    permute_54: "f32[1024, 128]" = torch.ops.aten.permute.default(view_64, [1, 0])
    mm_13: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_54, view_16);  permute_54 = view_16 = None
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_18: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_64, [0], True);  view_64 = None
    view_65: "f32[1024]" = torch.ops.aten.view.default(sum_18, [1024]);  sum_18 = None
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    view_66: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_12, [1, 128, 1024]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:291, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_67: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_66, [1, 128, 16, 64]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:287, code: attn_output = attn_output.transpose(1, 2)
    permute_57: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:286, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_68: "f32[16, 128, 64]" = torch.ops.aten.view.default(permute_57, [16, 128, 64]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_58: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    bmm_8: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_58, view_68);  permute_58 = None
    permute_59: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_9: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_68, permute_59);  view_68 = permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_3: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_41: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_9, alias_3);  bmm_9 = None
    sum_19: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_41, [-1], True)
    mul_42: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_3, sum_19);  alias_3 = sum_19 = None
    sub_12: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_69: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(sub_12, [1, 16, 128, 128]);  sub_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:252, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_70: "f32[16, 128, 128]" = torch.ops.aten.view.default(view_69, [16, 128, 128]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_60: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_10: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_60, view_70);  permute_60 = None
    permute_61: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    bmm_11: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_70, permute_61);  view_70 = permute_61 = None
    permute_62: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_10, [0, 2, 1]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:236, code: value_states = value_states.reshape(*proj_shape)
    view_71: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_8, [1, 16, 128, 64]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:235, code: key_states = key_states.reshape(*proj_shape)
    view_72: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(permute_62, [1, 16, 128, 64]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:234, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_73: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_11, [1, 16, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_63: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    clone_16: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_74: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_16, [1, 128, 1024]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_64: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    clone_17: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    view_75: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_17, [1, 128, 1024]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_76: "f32[128, 1024]" = torch.ops.aten.view.default(view_75, [128, 1024]);  view_75 = None
    permute_65: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_14: "f32[128, 1024]" = torch.ops.aten.mm.default(view_76, permute_65);  permute_65 = None
    permute_66: "f32[1024, 128]" = torch.ops.aten.permute.default(view_76, [1, 0])
    mm_15: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_66, view_5);  permute_66 = view_5 = None
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_20: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_76, [0], True);  view_76 = None
    view_77: "f32[1024]" = torch.ops.aten.view.default(sum_20, [1024]);  sum_20 = None
    permute_68: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    view_78: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_14, [1, 128, 1024]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_69: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    view_79: "f32[1, 128, 1024]" = torch.ops.aten.view.default(permute_69, [1, 128, 1024]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_80: "f32[128, 1024]" = torch.ops.aten.view.default(view_79, [128, 1024]);  view_79 = None
    permute_70: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_16: "f32[128, 1024]" = torch.ops.aten.mm.default(view_80, permute_70);  permute_70 = None
    permute_71: "f32[1024, 128]" = torch.ops.aten.permute.default(view_80, [1, 0])
    mm_17: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_71, view_2);  permute_71 = view_2 = None
    permute_72: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_21: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_80, [0], True);  view_80 = None
    view_81: "f32[1024]" = torch.ops.aten.view.default(sum_21, [1024]);  sum_21 = None
    permute_73: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    view_82: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_16, [1, 128, 1024]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_16: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_78, view_82);  view_78 = view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_43: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_74, 0.125);  view_74 = None
    view_83: "f32[128, 1024]" = torch.ops.aten.view.default(mul_43, [128, 1024]);  mul_43 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_18: "f32[128, 1024]" = torch.ops.aten.mm.default(view_83, permute_74);  permute_74 = None
    permute_75: "f32[1024, 128]" = torch.ops.aten.permute.default(view_83, [1, 0])
    mm_19: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_75, view);  permute_75 = view = None
    permute_76: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_22: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_83, [0], True);  view_83 = None
    view_84: "f32[1024]" = torch.ops.aten.view.default(sum_22, [1024]);  sum_22 = None
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    view_85: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_18, [1, 128, 1024]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_17: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_16, view_85);  add_16 = view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_13: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_27, getitem_1);  primals_27 = getitem_1 = None
    mul_44: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt);  sub_13 = None
    mul_45: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_17, primals_1);  primals_1 = None
    mul_46: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_45, 1024)
    sum_23: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_45, [2], True)
    mul_47: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_45, mul_44);  mul_45 = None
    sum_24: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_47, [2], True);  mul_47 = None
    mul_48: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_44, sum_24);  sum_24 = None
    sub_14: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_46, sum_23);  mul_46 = sum_23 = None
    sub_15: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_14, mul_48);  sub_14 = mul_48 = None
    div_4: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
    mul_49: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_4, sub_15);  div_4 = sub_15 = None
    mul_50: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_17, mul_44);  mul_44 = None
    sum_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_50, [0, 1]);  mul_50 = None
    sum_26: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_17, [0, 1]);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:426, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_18: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_15, mul_49);  add_15 = mul_49 = None
    return pytree.tree_unflatten([add_10, sum_25, sum_26, permute_77, view_84, permute_73, view_81, permute_68, view_77, permute_56, view_65, sum_16, sum_17, permute_52, view_62, permute_48, view_59, permute_43, view_55, permute_31, view_45, sum_7, sum_8, permute_27, view_42, permute_23, view_39, add_18, None, add_14], self._out_spec)
    