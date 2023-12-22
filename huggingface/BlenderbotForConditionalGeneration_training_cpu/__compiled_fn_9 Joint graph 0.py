from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[2560]"; primals_2: "f32[2560]"; primals_3: "f32[2560, 2560]"; primals_4: "f32[2560]"; primals_5: "f32[2560, 2560]"; primals_6: "f32[2560]"; primals_7: "f32[2560, 2560]"; primals_8: "f32[2560]"; primals_9: "f32[2560, 2560]"; primals_10: "f32[2560]"; primals_11: "f32[2560]"; primals_12: "f32[2560]"; primals_13: "f32[10240, 2560]"; primals_14: "f32[10240]"; primals_15: "f32[2560, 10240]"; primals_16: "f32[2560]"; primals_17: "f32[1, 128, 2560]"; tangents_1: "f32[1, 128, 2560]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:320, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(primals_17, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(primals_17, getitem_1)
    mul: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
    add_1: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[128, 2560]" = torch.ops.aten.view.default(add_1, [128, 2560])
    permute: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm: "f32[128, 2560]" = torch.ops.aten.addmm.default(primals_4, view, permute);  primals_4 = None
    view_1: "f32[1, 128, 2560]" = torch.ops.aten.view.default(addmm, [1, 128, 2560]);  addmm = None
    mul_2: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_1, 0.11180339887498948);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_2: "f32[128, 2560]" = torch.ops.aten.view.default(add_1, [128, 2560])
    permute_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_1: "f32[128, 2560]" = torch.ops.aten.addmm.default(primals_6, view_2, permute_1);  primals_6 = None
    view_3: "f32[1, 128, 2560]" = torch.ops.aten.view.default(addmm_1, [1, 128, 2560]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 128, 32, 80]" = torch.ops.aten.view.default(view_3, [1, -1, 32, 80]);  view_3 = None
    permute_2: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_5: "f32[128, 2560]" = torch.ops.aten.view.default(add_1, [128, 2560]);  add_1 = None
    permute_3: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_2: "f32[128, 2560]" = torch.ops.aten.addmm.default(primals_8, view_5, permute_3);  primals_8 = None
    view_6: "f32[1, 128, 2560]" = torch.ops.aten.view.default(addmm_2, [1, 128, 2560]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 128, 32, 80]" = torch.ops.aten.view.default(view_6, [1, -1, 32, 80]);  view_6 = None
    permute_4: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 128, 32, 80]" = torch.ops.aten.view.default(mul_2, [1, 128, 32, 80]);  mul_2 = None
    permute_5: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 32, 128, 80]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_9: "f32[32, 128, 80]" = torch.ops.aten.view.default(clone_2, [32, -1, 80]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_10: "f32[32, 128, 80]" = torch.ops.aten.view.default(clone, [32, -1, 80]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_11: "f32[32, 128, 80]" = torch.ops.aten.view.default(clone_1, [32, -1, 80]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_9, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[32, 128, 1]" = torch.ops.aten.amax.default(bmm, [-1], True)
    sub_1: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(bmm, amax);  bmm = amax = None
    exp: "f32[32, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[32, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[32, 128, 128]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:261, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_3: "f32[32, 128, 128]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(clone_3, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_12: "f32[1, 32, 128, 80]" = torch.ops.aten.view.default(bmm_1, [1, 32, 128, 80]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_4: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_13: "f32[1, 128, 2560]" = torch.ops.aten.view.default(clone_4, [1, 128, 2560]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_14: "f32[128, 2560]" = torch.ops.aten.view.default(view_13, [128, 2560]);  view_13 = None
    permute_8: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_3: "f32[128, 2560]" = torch.ops.aten.addmm.default(primals_10, view_14, permute_8);  primals_10 = None
    view_15: "f32[1, 128, 2560]" = torch.ops.aten.view.default(addmm_3, [1, 128, 2560]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:327, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout = torch.ops.aten.native_dropout.default(view_15, 0.1, True);  view_15 = None
    getitem_2: "f32[1, 128, 2560]" = native_dropout[0]
    getitem_3: "b8[1, 128, 2560]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:328, code: hidden_states = residual + hidden_states
    add_2: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(primals_17, getitem_2);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:331, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_2: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_2, getitem_5)
    mul_3: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_4: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_3, primals_11);  mul_3 = None
    add_4: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(mul_4, primals_12);  mul_4 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:332, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_16: "f32[128, 2560]" = torch.ops.aten.view.default(add_4, [128, 2560]);  add_4 = None
    permute_9: "f32[2560, 10240]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_4: "f32[128, 10240]" = torch.ops.aten.addmm.default(primals_14, view_16, permute_9);  primals_14 = None
    view_17: "f32[1, 128, 10240]" = torch.ops.aten.view.default(addmm_4, [1, 128, 10240]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    mul_6: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_17, 0.7071067811865476)
    erf: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_5: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:333, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_5: "f32[1, 128, 10240]" = torch.ops.aten.clone.default(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:334, code: hidden_states = self.fc2(hidden_states)
    view_18: "f32[128, 10240]" = torch.ops.aten.view.default(clone_5, [128, 10240]);  clone_5 = None
    permute_10: "f32[10240, 2560]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_5: "f32[128, 2560]" = torch.ops.aten.addmm.default(primals_16, view_18, permute_10);  primals_16 = None
    view_19: "f32[1, 128, 2560]" = torch.ops.aten.view.default(addmm_5, [1, 128, 2560]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_19, 0.1, True);  view_19 = None
    getitem_6: "f32[1, 128, 2560]" = native_dropout_1[0]
    getitem_7: "b8[1, 128, 2560]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:336, code: hidden_states = residual + hidden_states
    add_6: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_2, getitem_6);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[1, 128, 2560]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_8: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_9: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(tangents_1, mul_8);  mul_8 = None
    clone_6: "f32[1, 128, 2560]" = torch.ops.aten.clone.default(mul_9, memory_format = torch.contiguous_format);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:334, code: hidden_states = self.fc2(hidden_states)
    view_20: "f32[128, 2560]" = torch.ops.aten.view.default(clone_6, [128, 2560]);  clone_6 = None
    permute_11: "f32[2560, 10240]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm: "f32[128, 10240]" = torch.ops.aten.mm.default(view_20, permute_11);  permute_11 = None
    permute_12: "f32[2560, 128]" = torch.ops.aten.permute.default(view_20, [1, 0])
    mm_1: "f32[2560, 10240]" = torch.ops.aten.mm.default(permute_12, view_18);  permute_12 = view_18 = None
    permute_13: "f32[10240, 2560]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_2: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_20, [0], True);  view_20 = None
    view_21: "f32[2560]" = torch.ops.aten.view.default(sum_2, [2560]);  sum_2 = None
    permute_14: "f32[2560, 10240]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    view_22: "f32[1, 128, 10240]" = torch.ops.aten.view.default(mm, [1, 128, 10240]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_10: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_17, 0.7071067811865476)
    erf_1: "f32[1, 128, 10240]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
    add_7: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_11: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
    mul_12: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_17, view_17)
    mul_13: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_12, -0.5);  mul_12 = None
    exp_1: "f32[1, 128, 10240]" = torch.ops.aten.exp.default(mul_13);  mul_13 = None
    mul_14: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_15: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_17, mul_14);  view_17 = mul_14 = None
    add_8: "f32[1, 128, 10240]" = torch.ops.aten.add.Tensor(mul_11, mul_15);  mul_11 = mul_15 = None
    mul_16: "f32[1, 128, 10240]" = torch.ops.aten.mul.Tensor(view_22, add_8);  view_22 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:332, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_23: "f32[128, 10240]" = torch.ops.aten.view.default(mul_16, [128, 10240]);  mul_16 = None
    permute_15: "f32[10240, 2560]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_2: "f32[128, 2560]" = torch.ops.aten.mm.default(view_23, permute_15);  permute_15 = None
    permute_16: "f32[10240, 128]" = torch.ops.aten.permute.default(view_23, [1, 0])
    mm_3: "f32[10240, 2560]" = torch.ops.aten.mm.default(permute_16, view_16);  permute_16 = view_16 = None
    permute_17: "f32[2560, 10240]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_3: "f32[1, 10240]" = torch.ops.aten.sum.dim_IntList(view_23, [0], True);  view_23 = None
    view_24: "f32[10240]" = torch.ops.aten.view.default(sum_3, [10240]);  sum_3 = None
    permute_18: "f32[10240, 2560]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    view_25: "f32[1, 128, 2560]" = torch.ops.aten.view.default(mm_2, [1, 128, 2560]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:331, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_3: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(add_2, getitem_5);  add_2 = getitem_5 = None
    mul_17: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_18: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_25, primals_11);  primals_11 = None
    mul_19: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_18, 2560)
    sum_4: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
    mul_20: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_18, mul_17);  mul_18 = None
    sum_5: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
    mul_21: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_17, sum_5);  sum_5 = None
    sub_4: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_19, sum_4);  mul_19 = sum_4 = None
    sub_5: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_4, mul_21);  sub_4 = mul_21 = None
    div_1: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2560);  rsqrt_1 = None
    mul_22: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
    mul_23: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_25, mul_17);  mul_17 = None
    sum_6: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
    sum_7: "f32[2560]" = torch.ops.aten.sum.dim_IntList(view_25, [0, 1]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:331, code: hidden_states = self.final_layer_norm(hidden_states)
    add_9: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(tangents_1, mul_22);  tangents_1 = mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:327, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 128, 2560]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_24: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_25: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(add_9, mul_24);  mul_24 = None
    clone_7: "f32[1, 128, 2560]" = torch.ops.aten.clone.default(mul_25, memory_format = torch.contiguous_format);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:278, code: attn_output = self.out_proj(attn_output)
    view_26: "f32[128, 2560]" = torch.ops.aten.view.default(clone_7, [128, 2560]);  clone_7 = None
    permute_19: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_4: "f32[128, 2560]" = torch.ops.aten.mm.default(view_26, permute_19);  permute_19 = None
    permute_20: "f32[2560, 128]" = torch.ops.aten.permute.default(view_26, [1, 0])
    mm_5: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_20, view_14);  permute_20 = view_14 = None
    permute_21: "f32[2560, 2560]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_8: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_26, [0], True);  view_26 = None
    view_27: "f32[2560]" = torch.ops.aten.view.default(sum_8, [2560]);  sum_8 = None
    permute_22: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    view_28: "f32[1, 128, 2560]" = torch.ops.aten.view.default(mm_4, [1, 128, 2560]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:276, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_29: "f32[1, 128, 32, 80]" = torch.ops.aten.view.default(view_28, [1, 128, 32, 80]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:272, code: attn_output = attn_output.transpose(1, 2)
    permute_23: "f32[1, 32, 128, 80]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:271, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_30: "f32[32, 128, 80]" = torch.ops.aten.view.default(permute_23, [32, 128, 80]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:263, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_24: "f32[32, 128, 128]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    bmm_2: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(permute_24, view_30);  permute_24 = None
    permute_25: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_3: "f32[32, 128, 128]" = torch.ops.aten.bmm.default(view_30, permute_25);  view_30 = permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_1: "f32[32, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_26: "f32[32, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_3, alias_1);  bmm_3 = None
    sum_9: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_26, [-1], True)
    mul_27: "f32[32, 128, 128]" = torch.ops.aten.mul.Tensor(alias_1, sum_9);  alias_1 = sum_9 = None
    sub_6: "f32[32, 128, 128]" = torch.ops.aten.sub.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:224, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_26: "f32[32, 80, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_4: "f32[32, 80, 128]" = torch.ops.aten.bmm.default(permute_26, sub_6);  permute_26 = None
    permute_27: "f32[32, 128, 80]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    bmm_5: "f32[32, 128, 80]" = torch.ops.aten.bmm.default(sub_6, permute_27);  sub_6 = permute_27 = None
    permute_28: "f32[32, 128, 80]" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:221, code: value_states = value_states.reshape(*proj_shape)
    view_31: "f32[1, 32, 128, 80]" = torch.ops.aten.view.default(bmm_2, [1, 32, 128, 80]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:220, code: key_states = key_states.reshape(*proj_shape)
    view_32: "f32[1, 32, 128, 80]" = torch.ops.aten.view.default(permute_28, [1, 32, 128, 80]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:219, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_33: "f32[1, 32, 128, 80]" = torch.ops.aten.view.default(bmm_5, [1, 32, 128, 80]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_29: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    clone_8: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_34: "f32[1, 128, 2560]" = torch.ops.aten.view.default(clone_8, [1, 128, 2560]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_30: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    clone_9: "f32[1, 128, 32, 80]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_35: "f32[1, 128, 2560]" = torch.ops.aten.view.default(clone_9, [1, 128, 2560]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:206, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_36: "f32[128, 2560]" = torch.ops.aten.view.default(view_35, [128, 2560]);  view_35 = None
    permute_31: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_6: "f32[128, 2560]" = torch.ops.aten.mm.default(view_36, permute_31);  permute_31 = None
    permute_32: "f32[2560, 128]" = torch.ops.aten.permute.default(view_36, [1, 0])
    mm_7: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_32, view_5);  permute_32 = view_5 = None
    permute_33: "f32[2560, 2560]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_10: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_36, [0], True);  view_36 = None
    view_37: "f32[2560]" = torch.ops.aten.view.default(sum_10, [2560]);  sum_10 = None
    permute_34: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    view_38: "f32[1, 128, 2560]" = torch.ops.aten.view.default(mm_6, [1, 128, 2560]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:160, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_35: "f32[1, 128, 32, 80]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    view_39: "f32[1, 128, 2560]" = torch.ops.aten.view.default(permute_35, [1, 128, 2560]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_40: "f32[128, 2560]" = torch.ops.aten.view.default(view_39, [128, 2560]);  view_39 = None
    permute_36: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_8: "f32[128, 2560]" = torch.ops.aten.mm.default(view_40, permute_36);  permute_36 = None
    permute_37: "f32[2560, 128]" = torch.ops.aten.permute.default(view_40, [1, 0])
    mm_9: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_37, view_2);  permute_37 = view_2 = None
    permute_38: "f32[2560, 2560]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_11: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
    view_41: "f32[2560]" = torch.ops.aten.view.default(sum_11, [2560]);  sum_11 = None
    permute_39: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_42: "f32[1, 128, 2560]" = torch.ops.aten.view.default(mm_8, [1, 128, 2560]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:205, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_10: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(view_38, view_42);  view_38 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_28: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(view_34, 0.11180339887498948);  view_34 = None
    view_43: "f32[128, 2560]" = torch.ops.aten.view.default(mul_28, [128, 2560]);  mul_28 = None
    permute_40: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_10: "f32[128, 2560]" = torch.ops.aten.mm.default(view_43, permute_40);  permute_40 = None
    permute_41: "f32[2560, 128]" = torch.ops.aten.permute.default(view_43, [1, 0])
    mm_11: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
    permute_42: "f32[2560, 2560]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_12: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_43, [0], True);  view_43 = None
    view_44: "f32[2560]" = torch.ops.aten.view.default(sum_12, [2560]);  sum_12 = None
    permute_43: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_45: "f32[1, 128, 2560]" = torch.ops.aten.view.default(mm_10, [1, 128, 2560]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:180, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_11: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_10, view_45);  add_10 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:320, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_7: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(primals_17, getitem_1);  primals_17 = getitem_1 = None
    mul_29: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt);  sub_7 = None
    mul_30: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(add_11, primals_1);  primals_1 = None
    mul_31: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_30, 2560)
    sum_13: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_30, [2], True)
    mul_32: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_30, mul_29);  mul_30 = None
    sum_14: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_32, [2], True);  mul_32 = None
    mul_33: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_29, sum_14);  sum_14 = None
    sub_8: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_31, sum_13);  mul_31 = sum_13 = None
    sub_9: "f32[1, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_8, mul_33);  sub_8 = mul_33 = None
    div_2: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2560);  rsqrt = None
    mul_34: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(div_2, sub_9);  div_2 = sub_9 = None
    mul_35: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(add_11, mul_29);  mul_29 = None
    sum_15: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_35, [0, 1]);  mul_35 = None
    sum_16: "f32[2560]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:320, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_12: "f32[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_9, mul_34);  add_9 = mul_34 = None
    return pytree.tree_unflatten([add_6, sum_15, sum_16, permute_43, view_44, permute_39, view_41, permute_34, view_37, permute_22, view_27, sum_6, sum_7, permute_18, view_24, permute_14, view_21, add_12], self._out_spec)
    