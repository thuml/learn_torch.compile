from __future__ import annotations



def forward(self, primals_1: "f32[768, 768]", primals_2: "f32[768]", primals_3: "f32[768, 768]", primals_4: "f32[768]", primals_5: "f32[768, 768]", primals_6: "f32[768]", primals_7: "f32[768, 768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[768, 768]", primals_12: "f32[768]", primals_13: "f32[768, 768]", primals_14: "f32[768]", primals_15: "f32[768, 768]", primals_16: "f32[768]", primals_17: "f32[768, 768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[3072, 768]", primals_22: "f32[3072]", primals_23: "f32[768, 3072]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[1, 1024, 768]", primals_28: "f32[1, 1, 1024, 1024]", primals_29: "f32[1, 1024, 768]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[1024, 768]" = torch.ops.aten.view.default(primals_27, [1024, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    addmm: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = None
    view_1: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm, [1, 1024, 768]);  addmm = None
    mul: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_4, view, permute_1);  primals_4 = None
    view_3: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_1, [1, 1024, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_3, [1, -1, 12, 64]);  view_3 = None
    permute_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_2: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_6, view, permute_3);  primals_6 = None
    view_6: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_2, [1, 1024, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_6, [1, -1, 12, 64]);  view_6 = None
    permute_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(mul, [1, 1024, 12, 64]);  mul = None
    permute_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_9: "f32[12, 1024, 64]" = torch.ops.aten.view.default(clone_2, [12, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_10: "f32[12, 1024, 64]" = torch.ops.aten.view.default(clone, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_11: "f32[12, 1024, 64]" = torch.ops.aten.view.default(clone_1, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_9, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_12: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm, [1, 12, 1024, 1024]);  bmm = None
    add: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_12, primals_28);  view_12 = primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_13: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(add, [12, 1024, 1024]);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_13, [-1], True)
    sub: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_13, amax);  view_13 = amax = None
    exp: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub);  sub = None
    sum_1: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[12, 1024, 1024]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:269, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    native_dropout = torch.ops.aten.native_dropout.default(div, 0.1, True);  div = None
    getitem: "f32[12, 1024, 1024]" = native_dropout[0]
    getitem_1: "b8[12, 1024, 1024]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(getitem, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_14: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_3: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_15: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_3, [1, 1024, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_16: "f32[1024, 768]" = torch.ops.aten.view.default(view_15, [1024, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_3: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_8, view_16, permute_8);  primals_8 = None
    view_17: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_3, [1, 1024, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:431, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_2: "f32[1, 1024, 768]" = native_dropout_1[0]
    getitem_3: "b8[1, 1024, 768]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_1: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(primals_27, getitem_2);  primals_27 = getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_5);  add_1 = getitem_5 = None
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_9)
    add_3: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_10);  mul_2 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_18: "f32[1024, 768]" = torch.ops.aten.view.default(add_3, [1024, 768])
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_4: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_12, view_18, permute_9);  primals_12 = None
    view_19: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_4, [1, 1024, 768]);  addmm_4 = None
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_19, 0.125);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_20: "f32[1024, 768]" = torch.ops.aten.view.default(primals_29, [1024, 768]);  primals_29 = None
    permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_5: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_14, view_20, permute_10);  primals_14 = None
    view_21: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_5, [1, 1024, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_22: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_21, [1, -1, 12, 64]);  view_21 = None
    permute_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    clone_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_6: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_16, view_20, permute_12);  primals_16 = None
    view_24: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_6, [1, 1024, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_24, [1, -1, 12, 64]);  view_24 = None
    permute_13: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_26: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(mul_3, [1, 1024, 12, 64]);  mul_3 = None
    permute_14: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    clone_6: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_27: "f32[12, 1024, 64]" = torch.ops.aten.view.default(clone_6, [12, -1, 64]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_28: "f32[12, 1024, 64]" = torch.ops.aten.view.default(clone_4, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_29: "f32[12, 1024, 64]" = torch.ops.aten.view.default(clone_5, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_27, 0);  view_27 = None
    unsqueeze_default_1: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_28, 0);  view_28 = None
    unsqueeze_default_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_29, 0);  view_29 = None
    mul_scalar: "f32[1, 12, 1024, 64]" = torch.ops.aten.mul.Scalar(unsqueeze_default, 1.0);  unsqueeze_default = None
    permute_default: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_1, [0, 1, 3, 2]);  unsqueeze_default_1 = None
    mul_scalar_1: "f32[1, 12, 64, 1024]" = torch.ops.aten.mul.Scalar(permute_default, 1.0);  permute_default = None
    expand_default: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(mul_scalar, [1, 12, 1024, 64]);  mul_scalar = None
    view_default: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_default, [12, 1024, 64]);  expand_default = None
    expand_default_1: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(mul_scalar_1, [1, 12, 64, 1024]);  mul_scalar_1 = None
    view_default_1: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_default_1, [12, 64, 1024]);  expand_default_1 = None
    bmm_default: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_default, view_default_1)
    view_default_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_default, [1, 12, 1024, 1024]);  bmm_default = None
    amax_default: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(view_default_2, [-1], True)
    sub_tensor: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_default_2, amax_default);  view_default_2 = amax_default = None
    exp_default: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
    sum_dim_int_list: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
    alias_default: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_tensor)
    native_dropout_default = torch.ops.aten.native_dropout.default(div_tensor, 0.1, True);  div_tensor = None
    getitem_16: "f32[1, 12, 1024, 1024]" = native_dropout_default[0]
    getitem_17: "b8[1, 12, 1024, 1024]" = native_dropout_default[1];  native_dropout_default = None
    expand_default_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_16, [1, 12, 1024, 1024]);  getitem_16 = None
    view_default_3: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_default_2, [12, 1024, 1024]);  expand_default_2 = None
    expand_default_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(unsqueeze_default_2, [1, 12, 1024, 64]);  unsqueeze_default_2 = None
    view_default_4: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_default_3, [12, 1024, 64]);  expand_default_3 = None
    bmm_default_1: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_default_3, view_default_4)
    view_default_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_default_1, [1, 12, 1024, 64]);  bmm_default_1 = None
    squeeze_dim: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(view_default_5, 0);  view_default_5 = None
    permute_default_1: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
    permute_default_2: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
    alias_default_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_default);  alias_default = None
    permute_default_3: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
    permute_default_4: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_30: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(squeeze_dim, [1, 12, 1024, 64]);  squeeze_dim = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_16: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_31: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_7, [1, 1024, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_32: "f32[1024, 768]" = torch.ops.aten.view.default(view_31, [1024, 768]);  view_31 = None
    permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_7: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_18, view_32, permute_17);  primals_18 = None
    view_33: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_7, [1, 1024, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:451, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_33, 0.1, True);  view_33 = None
    getitem_8: "f32[1, 1024, 768]" = native_dropout_3[0]
    getitem_9: "b8[1, 1024, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:452, code: hidden_states = residual + hidden_states
    add_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_3, getitem_8);  add_3 = getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:453, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_11: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_3: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_11);  add_4 = getitem_11 = None
    mul_4: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_5: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_4, primals_19)
    add_6: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_5, primals_20);  mul_5 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_34: "f32[1024, 768]" = torch.ops.aten.view.default(add_6, [1024, 768])
    permute_18: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_8: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_22, view_34, permute_18);  primals_22 = None
    view_35: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_8, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    mul_7: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476);  view_35 = None
    erf: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_7: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_7);  mul_6 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:461, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_8: "f32[1, 1024, 3072]" = torch.ops.aten.clone.default(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_36: "f32[1024, 3072]" = torch.ops.aten.view.default(clone_8, [1024, 3072]);  clone_8 = None
    permute_19: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_9: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_24, view_36, permute_19);  primals_24 = None
    view_37: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_9, [1, 1024, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:463, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_4 = torch.ops.aten.native_dropout.default(view_37, 0.1, True);  view_37 = None
    getitem_12: "f32[1, 1024, 768]" = native_dropout_4[0]
    getitem_13: "b8[1, 1024, 768]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_8: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_6, getitem_12);  add_6 = getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_15: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_4: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_15);  add_8 = getitem_15 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_25)
    add_10: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_26);  mul_10 = primals_26 = None
    div_2: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:453, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_3: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    permute_28: "f32[768, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_49: "f32[768, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_4: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_58: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(getitem, [0, 2, 1]);  getitem = None
    permute_59: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_3: "f32[12, 1024, 1024]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_60: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_61: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_65: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [add_10, clone, clone_1, clone_4, clone_5, primals_9, primals_19, primals_25, view, getitem_1, view_16, getitem_3, mul_1, view_18, view_20, getitem_17, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_32, getitem_9, mul_4, view_34, addmm_8, view_36, getitem_13, mul_9, div_2, permute_20, permute_24, div_3, permute_28, permute_40, permute_45, permute_49, div_4, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74]
    