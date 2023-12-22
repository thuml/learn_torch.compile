from __future__ import annotations



def forward(self, primals_1: "f32[768, 768]", primals_2: "f32[768]", primals_3: "f32[768, 768]", primals_4: "f32[768]", primals_5: "f32[768, 768]", primals_6: "f32[768]", primals_7: "f32[768, 768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[3072, 768]", primals_12: "f32[3072]", primals_13: "f32[768, 3072]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[1, 1024, 768]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[1024, 768]" = torch.ops.aten.reshape.default(primals_17, [1024, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    addmm: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = None
    view_1: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm, [1, 1024, 768]);  addmm = None
    mul: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_4, view, permute_1);  primals_4 = None
    view_3: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 1024, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_3, [1, -1, 12, 64]);  view_3 = None
    permute_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_2: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_6, view, permute_3);  primals_6 = None
    view_6: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 1024, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_6, [1, -1, 12, 64]);  view_6 = None
    permute_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul, [1, 1024, 12, 64]);  mul = None
    permute_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_9: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_2, [12, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_10: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone, [12, -1, 64]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_11: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_1, [12, -1, 64]);  clone_1 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_9, 0);  view_9 = None
    unsqueeze_default_1: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_10, 0);  view_10 = None
    unsqueeze_default_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_11, 0);  view_11 = None
    mul_scalar: "f32[1, 12, 1024, 64]" = torch.ops.aten.mul.Scalar(unsqueeze_default, 1.0);  unsqueeze_default = None
    permute_default: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_1, [0, 1, 3, 2]);  unsqueeze_default_1 = None
    mul_scalar_1: "f32[1, 12, 64, 1024]" = torch.ops.aten.mul.Scalar(permute_default, 1.0);  permute_default = None
    expand_default: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(mul_scalar, [1, 12, 1024, 64]);  mul_scalar = None
    view_default: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_default, [12, 1024, 64]);  expand_default = None
    expand_default_1: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(mul_scalar_1, [1, 12, 64, 1024]);  mul_scalar_1 = None
    view_default_1: "f32[12, 64, 1024]" = torch.ops.aten.reshape.default(expand_default_1, [12, 64, 1024]);  expand_default_1 = None
    bmm_default: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_default, view_default_1)
    view_default_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_default, [1, 12, 1024, 1024]);  bmm_default = None
    amax_default: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(view_default_2, [-1], True)
    sub_tensor: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_default_2, amax_default);  view_default_2 = amax_default = None
    exp_default: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
    sum_dim_int_list: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
    alias_default: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_tensor)
    native_dropout_default = torch.ops.aten.native_dropout.default(div_tensor, 0.1, True);  div_tensor = None
    getitem_10: "f32[1, 12, 1024, 1024]" = native_dropout_default[0]
    getitem_11: "b8[1, 12, 1024, 1024]" = native_dropout_default[1];  native_dropout_default = None
    expand_default_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_10, [1, 12, 1024, 1024]);  getitem_10 = None
    view_default_3: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(expand_default_2, [12, 1024, 1024]);  expand_default_2 = None
    expand_default_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(unsqueeze_default_2, [1, 12, 1024, 64]);  unsqueeze_default_2 = None
    view_default_4: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(expand_default_3, [12, 1024, 64]);  expand_default_3 = None
    bmm_default_1: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_default_3, view_default_4)
    view_default_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_default_1, [1, 12, 1024, 64]);  bmm_default_1 = None
    squeeze_dim: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(view_default_5, 0);  view_default_5 = None
    permute_default_1: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
    permute_default_2: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
    alias_default_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_default);  alias_default = None
    permute_default_3: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
    permute_default_4: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim, [1, 12, 1024, 64]);  squeeze_dim = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_3: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_13: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_3, [1, 1024, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_14: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_13, [1024, 768]);  view_13 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_3: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_8, view_14, permute_8);  primals_8 = None
    view_15: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 1024, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:334, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_15, 0.1, True);  view_15 = None
    getitem_2: "f32[1, 1024, 768]" = native_dropout_1[0]
    getitem_3: "b8[1, 1024, 768]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:335, code: hidden_states = residual + hidden_states
    add: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(primals_17, getitem_2);  primals_17 = getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:336, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub_1: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add, getitem_5);  add = getitem_5 = None
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_9)
    add_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_10);  mul_2 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_16: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_2, [1024, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_4: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_12, view_16, permute_9);  primals_12 = None
    view_17: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 1024, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_3: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    mul_4: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.7071067811865476);  view_17 = None
    erf: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_3: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_5: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_3, add_3);  mul_3 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_18: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_5, [1024, 3072]);  mul_5 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_5: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_14, view_18, permute_10);  primals_14 = None
    view_19: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 1024, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:342, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_19, 0.1, True);  view_19 = None
    getitem_6: "f32[1, 1024, 768]" = native_dropout_2[0]
    getitem_7: "b8[1, 1024, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:343, code: hidden_states = residual + hidden_states
    add_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_2, getitem_6);  add_2 = getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:344, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_2: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_9);  add_4 = getitem_9 = None
    mul_6: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_7: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, primals_15)
    add_6: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_7, primals_16);  mul_7 = primals_16 = None
    div_1: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_15: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:336, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_2: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [add_6, primals_9, primals_15, view, getitem_11, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_14, getitem_3, mul_1, view_16, addmm_4, view_18, getitem_7, mul_6, div_1, permute_11, permute_15, div_2, permute_19, permute_31, permute_36, permute_40]
    