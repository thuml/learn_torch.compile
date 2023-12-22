from __future__ import annotations



def forward(self, primals_1: "f32[1024]", primals_2: "f32[1024]", primals_3: "f32[1024, 1024]", primals_4: "f32[1024]", primals_5: "f32[1024, 1024]", primals_6: "f32[1024]", primals_7: "f32[1024, 1024]", primals_8: "f32[1024]", primals_9: "f32[1024, 1024]", primals_10: "f32[1024]", primals_11: "f32[1024]", primals_12: "f32[1024]", primals_13: "f32[1024, 1024]", primals_14: "f32[1024]", primals_15: "f32[1024, 1024]", primals_16: "f32[1024]", primals_17: "f32[1024, 1024]", primals_18: "f32[1024]", primals_19: "f32[1024, 1024]", primals_20: "f32[1024]", primals_21: "f32[1024]", primals_22: "f32[1024]", primals_23: "f32[4096, 1024]", primals_24: "f32[4096]", primals_25: "f32[1024, 4096]", primals_26: "f32[1024]", primals_27: "f32[1, 128, 1024]", primals_28: "f32[1, 1, 128, 128]", primals_29: "f32[1, 128, 1024]"):
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
    view: "f32[128, 1024]" = torch.ops.aten.view.default(add_1, [128, 1024]);  add_1 = None
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_4, view, permute);  primals_4 = None
    view_1: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm, [1, 128, 1024]);  addmm = None
    mul_2: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_1: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_6, view, permute_1);  primals_6 = None
    view_3: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_1, [1, 128, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_3, [1, -1, 16, 64]);  view_3 = None
    permute_2: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_2: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_8, view, permute_3);  primals_8 = None
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
    sub_2: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_3, getitem_5);  getitem_5 = None
    mul_3: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_4: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_3, primals_11)
    add_5: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_4, primals_12);  mul_4 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_18: "f32[128, 1024]" = torch.ops.aten.view.default(add_5, [128, 1024]);  add_5 = None
    permute_9: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_4: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_14, view_18, permute_9);  primals_14 = None
    view_19: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_4, [1, 128, 1024]);  addmm_4 = None
    mul_5: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_19, 0.125);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_20: "f32[128, 1024]" = torch.ops.aten.view.default(primals_29, [128, 1024]);  primals_29 = None
    permute_10: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_5: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_16, view_20, permute_10);  primals_16 = None
    view_21: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 128, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:175, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_22: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_21, [1, -1, 16, 64]);  view_21 = None
    permute_11: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    clone_5: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_6: "f32[128, 1024]" = torch.ops.aten.addmm.default(primals_18, view_20, permute_12);  primals_18 = None
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
    sub_3: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1)
    exp_1: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
    
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
    add_6: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_3, getitem_6);  add_3 = getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_4: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_9);  getitem_9 = None
    mul_6: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_7: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_6, primals_21)
    add_8: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_7, primals_22);  mul_7 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_34: "f32[128, 1024]" = torch.ops.aten.view.default(add_8, [128, 1024]);  add_8 = None
    permute_18: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_8: "f32[128, 4096]" = torch.ops.aten.addmm.default(primals_24, view_34, permute_18);  primals_24 = None
    view_35: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_8, [1, 128, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_8: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    mul_9: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476);  view_35 = None
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
    add_10: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_6, getitem_10);  add_6 = getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:470, code: hidden_states = self.fc2(hidden_states)
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:468, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_24: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:467, code: hidden_states = self.final_layer_norm(hidden_states)
    div_2: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    permute_28: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_33: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_8, [0, 2, 1]);  clone_8 = None
    permute_34: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_35: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    permute_36: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_15, [0, 2, 1]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:211, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_40: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:210, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_49: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_3: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:293, code: attn_output = self.out_proj(attn_output)
    permute_53: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:278, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_58: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    permute_59: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:255, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_3: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:239, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_60: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_61: "f32[16, 128, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:221, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_65: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_70: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:195, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [add_10, primals_1, primals_11, primals_21, primals_27, getitem_1, rsqrt, view, view_16, getitem_3, mul_3, view_18, view_20, bmm_2, amax_1, sum_2, view_32, getitem_7, mul_6, view_34, addmm_8, view_36, getitem_11, permute_20, permute_24, div_2, permute_28, permute_33, permute_34, permute_35, permute_36, permute_40, permute_45, permute_49, div_3, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74]
    