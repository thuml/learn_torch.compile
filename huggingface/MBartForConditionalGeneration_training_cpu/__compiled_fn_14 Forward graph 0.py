from __future__ import annotations



def forward(self, primals_1: "f32[1024]", primals_2: "f32[1024]", primals_3: "f32[1024, 1024]", primals_4: "f32[1024]", primals_5: "f32[1024, 1024]", primals_6: "f32[1024]", primals_7: "f32[1024, 1024]", primals_8: "f32[1024]", primals_9: "f32[1024, 1024]", primals_10: "f32[1024]", primals_11: "f32[1024]", primals_12: "f32[1024]", primals_13: "f32[4096, 1024]", primals_14: "f32[4096]", primals_15: "f32[1024, 4096]", primals_16: "f32[1024]", primals_17: "f32[1, 1024, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(primals_17, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(primals_17, getitem_1)
    mul: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
    add_1: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[1024, 1024]" = torch.ops.aten.view.default(add_1, [1024, 1024]);  add_1 = None
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm: "f32[1024, 1024]" = torch.ops.aten.addmm.default(primals_4, view, permute);  primals_4 = None
    view_1: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm, [1, 1024, 1024]);  addmm = None
    mul_2: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_1: "f32[1024, 1024]" = torch.ops.aten.addmm.default(primals_6, view, permute_1);  primals_6 = None
    view_3: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_1, [1, 1024, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_3, [1, -1, 16, 64]);  view_3 = None
    permute_2: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_2: "f32[1024, 1024]" = torch.ops.aten.addmm.default(primals_8, view, permute_3);  primals_8 = None
    view_6: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_2, [1, 1024, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_6, [1, -1, 16, 64]);  view_6 = None
    permute_4: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_2, [1, 1024, 16, 64]);  mul_2 = None
    permute_5: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_9: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_2, [16, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_10: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone, [16, -1, 64]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_11: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_1, [16, -1, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_9, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(bmm, [-1], True)
    sub_1: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(bmm, amax)
    exp: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:270, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_3: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_3, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_12: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_1, [1, 16, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_4: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_13: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_4, [1, 1024, 1024]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_14: "f32[1024, 1024]" = torch.ops.aten.view.default(view_13, [1024, 1024]);  view_13 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_3: "f32[1024, 1024]" = torch.ops.aten.addmm.default(primals_10, view_14, permute_8);  primals_10 = None
    view_15: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_3, [1, 1024, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout = torch.ops.aten.native_dropout.default(view_15, 0.1, True);  view_15 = None
    getitem_2: "f32[1, 1024, 1024]" = native_dropout[0]
    getitem_3: "b8[1, 1024, 1024]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:336, code: hidden_states = residual + hidden_states
    add_2: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(primals_17, getitem_2);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_2: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_2, getitem_5);  getitem_5 = None
    mul_3: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_4: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_3, primals_11)
    add_4: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, primals_12);  mul_4 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_16: "f32[1024, 1024]" = torch.ops.aten.view.default(add_4, [1024, 1024]);  add_4 = None
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_4: "f32[1024, 4096]" = torch.ops.aten.addmm.default(primals_14, view_16, permute_9);  primals_14 = None
    view_17: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_4, [1, 1024, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    mul_6: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_17, 0.7071067811865476);  view_17 = None
    erf: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_5: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:341, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_5: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_18: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_5, [1024, 4096]);  clone_5 = None
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_5: "f32[1024, 1024]" = torch.ops.aten.addmm.default(primals_16, view_18, permute_10);  primals_16 = None
    view_19: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 1024, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_19, 0.1, True);  view_19 = None
    getitem_6: "f32[1, 1024, 1024]" = native_dropout_1[0]
    getitem_7: "b8[1, 1024, 1024]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:344, code: hidden_states = residual + hidden_states
    add_6: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_2, getitem_6);  add_2 = getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    permute_11: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_15: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    div_1: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_24: "f32[16, 1024, 1024]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    permute_25: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_26: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_27: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_31: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_40: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [add_6, primals_1, primals_11, primals_17, getitem_1, rsqrt, view, bmm, amax, sum_1, view_14, getitem_3, mul_3, view_16, addmm_4, view_18, getitem_7, permute_11, permute_15, div_1, permute_19, permute_24, permute_25, permute_26, permute_27, permute_31, permute_36, permute_40]
    