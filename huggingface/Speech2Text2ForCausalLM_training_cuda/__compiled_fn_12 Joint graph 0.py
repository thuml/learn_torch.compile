from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[256, 256]"; primals_2: "f32[256]"; primals_3: "f32[256, 256]"; primals_4: "f32[256]"; primals_5: "f32[256, 256]"; primals_6: "f32[256]"; primals_7: "f32[256, 256]"; primals_8: "f32[256]"; primals_9: "f32[256]"; primals_10: "f32[256]"; primals_11: "f32[2048, 256]"; primals_12: "f32[2048]"; primals_13: "f32[256, 2048]"; primals_14: "f32[256]"; primals_15: "f32[256]"; primals_16: "f32[256]"; primals_17: "f32[1, 128, 256]"; primals_18: "f32[1, 1, 128, 128]"; tangents_1: "f32[1, 128, 256]"; tangents_2: "f32[1, 4, 128, 64]"; tangents_3: "f32[1, 4, 128, 64]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, tangents_1, tangents_2, tangents_3, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[128, 256]" = torch.ops.aten.view.default(primals_17, [128, 256])
    permute: "f32[256, 256]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    addmm: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = None
    view_1: "f32[1, 128, 256]" = torch.ops.aten.view.default(addmm, [1, 128, 256]);  addmm = None
    mul: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_2: "f32[128, 256]" = torch.ops.aten.view.default(primals_17, [128, 256])
    permute_1: "f32[256, 256]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm_1: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_4, view_2, permute_1);  primals_4 = None
    view_3: "f32[1, 128, 256]" = torch.ops.aten.view.default(addmm_1, [1, 128, 256]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 128, 4, 64]" = torch.ops.aten.view.default(view_3, [1, -1, 4, 64]);  view_3 = None
    permute_2: "f32[1, 4, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 4, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_5: "f32[128, 256]" = torch.ops.aten.view.default(primals_17, [128, 256])
    permute_3: "f32[256, 256]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_2: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_6, view_5, permute_3);  primals_6 = None
    view_6: "f32[1, 128, 256]" = torch.ops.aten.view.default(addmm_2, [1, 128, 256]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 128, 4, 64]" = torch.ops.aten.view.default(view_6, [1, -1, 4, 64]);  view_6 = None
    permute_4: "f32[1, 4, 128, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 4, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 128, 4, 64]" = torch.ops.aten.view.default(mul, [1, 128, 4, 64]);  mul = None
    permute_5: "f32[1, 4, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 4, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:240, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_9: "f32[4, 128, 64]" = torch.ops.aten.view.default(clone_2, [4, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:241, code: key_states = key_states.reshape(*proj_shape)
    view_10: "f32[4, 128, 64]" = torch.ops.aten.view.default(clone, [4, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:242, code: value_states = value_states.reshape(*proj_shape)
    view_11: "f32[4, 128, 64]" = torch.ops.aten.view.default(clone_1, [4, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:245, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[4, 64, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm: "f32[4, 128, 128]" = torch.ops.aten.bmm.default(view_9, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:258, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_12: "f32[1, 4, 128, 128]" = torch.ops.aten.view.default(bmm, [1, 4, 128, 128]);  bmm = None
    add: "f32[1, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_12, primals_18);  view_12 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:259, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_13: "f32[4, 128, 128]" = torch.ops.aten.view.default(add, [4, 128, 128]);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:261, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[4, 128, 1]" = torch.ops.aten.amax.default(view_13, [-1], True)
    sub: "f32[4, 128, 128]" = torch.ops.aten.sub.Tensor(view_13, amax);  view_13 = amax = None
    exp: "f32[4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
    sum_1: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[4, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[4, 128, 128]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:282, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_3: "f32[4, 128, 128]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:284, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[4, 128, 64]" = torch.ops.aten.bmm.default(clone_3, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:292, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_14: "f32[1, 4, 128, 64]" = torch.ops.aten.view.default(bmm_1, [1, 4, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:293, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 4, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:297, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_4: "f32[1, 128, 4, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_15: "f32[1, 128, 256]" = torch.ops.aten.view.default(clone_4, [1, 128, 256]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:299, code: attn_output = self.out_proj(attn_output)
    view_16: "f32[128, 256]" = torch.ops.aten.view.default(view_15, [128, 256]);  view_15 = None
    permute_8: "f32[256, 256]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_3: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_8, view_16, permute_8);  primals_8 = None
    view_17: "f32[1, 128, 256]" = torch.ops.aten.view.default(addmm_3, [1, 128, 256]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:377, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem: "f32[1, 128, 256]" = native_dropout[0]
    getitem_1: "b8[1, 128, 256]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:378, code: hidden_states = residual + hidden_states
    add_1: "f32[1, 128, 256]" = torch.ops.aten.add.Tensor(primals_17, getitem);  primals_17 = getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:379, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean[0]
    getitem_3: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(add_1, getitem_3)
    mul_1: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_1, primals_9);  mul_1 = None
    add_3: "f32[1, 128, 256]" = torch.ops.aten.add.Tensor(mul_2, primals_10);  mul_2 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_18: "f32[128, 256]" = torch.ops.aten.view.default(add_3, [128, 256])
    permute_9: "f32[256, 2048]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_4: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_12, view_18, permute_9);  primals_12 = None
    view_19: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_4, [1, 128, 2048]);  addmm_4 = None
    relu: "f32[1, 128, 2048]" = torch.ops.aten.relu.default(view_19);  view_19 = None
    alias_1: "f32[1, 128, 2048]" = torch.ops.aten.alias.default(relu)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:407, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_5: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(relu);  relu = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:408, code: hidden_states = self.fc2(hidden_states)
    view_20: "f32[128, 2048]" = torch.ops.aten.view.default(clone_5, [128, 2048]);  clone_5 = None
    permute_10: "f32[2048, 256]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_5: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_14, view_20, permute_10);  primals_14 = None
    view_21: "f32[1, 128, 256]" = torch.ops.aten.view.default(addmm_5, [1, 128, 256]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:409, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_21, 0.1, True);  view_21 = None
    getitem_4: "f32[1, 128, 256]" = native_dropout_1[0]
    getitem_5: "b8[1, 128, 256]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:410, code: hidden_states = residual + hidden_states
    add_4: "f32[1, 128, 256]" = torch.ops.aten.add.Tensor(add_3, getitem_4);  add_3 = getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:411, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_2: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(add_4, getitem_7)
    mul_3: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_4: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_3, primals_15);  mul_3 = None
    add_6: "f32[1, 128, 256]" = torch.ops.aten.add.Tensor(mul_4, primals_16);  mul_4 = primals_16 = None
    sub_3: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
    mul_5: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_6: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(tangents_1, primals_15);  primals_15 = None
    mul_7: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_6, 256)
    sum_2: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2], True)
    mul_8: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_6, mul_5);  mul_6 = None
    sum_3: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True);  mul_8 = None
    mul_9: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_5, sum_3);  sum_3 = None
    sub_4: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(mul_7, sum_2);  mul_7 = sum_2 = None
    sub_5: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(sub_4, mul_9);  sub_4 = mul_9 = None
    div_1: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 256);  rsqrt_1 = None
    mul_10: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
    mul_11: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(tangents_1, mul_5);  mul_5 = None
    sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 1]);  mul_11 = None
    sum_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:409, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[1, 128, 256]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_12: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_13: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_10, mul_12);  mul_12 = None
    clone_6: "f32[1, 128, 256]" = torch.ops.aten.clone.default(mul_13, memory_format = torch.contiguous_format);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:408, code: hidden_states = self.fc2(hidden_states)
    view_22: "f32[128, 256]" = torch.ops.aten.view.default(clone_6, [128, 256]);  clone_6 = None
    permute_11: "f32[256, 2048]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm: "f32[128, 2048]" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
    permute_12: "f32[256, 128]" = torch.ops.aten.permute.default(view_22, [1, 0])
    mm_1: "f32[256, 2048]" = torch.ops.aten.mm.default(permute_12, view_20);  permute_12 = view_20 = None
    permute_13: "f32[2048, 256]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_6: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
    view_23: "f32[256]" = torch.ops.aten.view.default(sum_6, [256]);  sum_6 = None
    permute_14: "f32[256, 2048]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    view_24: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm, [1, 128, 2048]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    alias_2: "f32[1, 128, 2048]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    le: "b8[1, 128, 2048]" = torch.ops.aten.le.Scalar(alias_2, 0);  alias_2 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[1, 128, 2048]" = torch.ops.aten.where.self(le, scalar_tensor, view_24);  le = scalar_tensor = view_24 = None
    view_25: "f32[128, 2048]" = torch.ops.aten.view.default(where, [128, 2048]);  where = None
    permute_15: "f32[2048, 256]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_2: "f32[128, 256]" = torch.ops.aten.mm.default(view_25, permute_15);  permute_15 = None
    permute_16: "f32[2048, 128]" = torch.ops.aten.permute.default(view_25, [1, 0])
    mm_3: "f32[2048, 256]" = torch.ops.aten.mm.default(permute_16, view_18);  permute_16 = view_18 = None
    permute_17: "f32[256, 2048]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_7: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True);  view_25 = None
    view_26: "f32[2048]" = torch.ops.aten.view.default(sum_7, [2048]);  sum_7 = None
    permute_18: "f32[2048, 256]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    view_27: "f32[1, 128, 256]" = torch.ops.aten.view.default(mm_2, [1, 128, 256]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_7: "f32[1, 128, 256]" = torch.ops.aten.add.Tensor(mul_10, view_27);  mul_10 = view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:379, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub_6: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(add_1, getitem_3);  add_1 = getitem_3 = None
    mul_14: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = None
    mul_15: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(add_7, primals_9);  primals_9 = None
    mul_16: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_15, 256)
    sum_8: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)
    mul_17: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_15, mul_14);  mul_15 = None
    sum_9: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [2], True);  mul_17 = None
    mul_18: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_14, sum_9);  sum_9 = None
    sub_7: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(mul_16, sum_8);  mul_16 = sum_8 = None
    sub_8: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(sub_7, mul_18);  sub_7 = mul_18 = None
    div_2: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 256);  rsqrt = None
    mul_19: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
    mul_20: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(add_7, mul_14);  mul_14 = None
    sum_10: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_20, [0, 1]);  mul_20 = None
    sum_11: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_7, [0, 1]);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:377, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 128, 256]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_21: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_22: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_19, mul_21);  mul_21 = None
    clone_7: "f32[1, 128, 256]" = torch.ops.aten.clone.default(mul_22, memory_format = torch.contiguous_format);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:299, code: attn_output = self.out_proj(attn_output)
    view_28: "f32[128, 256]" = torch.ops.aten.view.default(clone_7, [128, 256]);  clone_7 = None
    permute_19: "f32[256, 256]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_4: "f32[128, 256]" = torch.ops.aten.mm.default(view_28, permute_19);  permute_19 = None
    permute_20: "f32[256, 128]" = torch.ops.aten.permute.default(view_28, [1, 0])
    mm_5: "f32[256, 256]" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
    permute_21: "f32[256, 256]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_12: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
    view_29: "f32[256]" = torch.ops.aten.view.default(sum_12, [256]);  sum_12 = None
    permute_22: "f32[256, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    view_30: "f32[1, 128, 256]" = torch.ops.aten.view.default(mm_4, [1, 128, 256]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:297, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_31: "f32[1, 128, 4, 64]" = torch.ops.aten.view.default(view_30, [1, 128, 4, 64]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:293, code: attn_output = attn_output.transpose(1, 2)
    permute_23: "f32[1, 4, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:292, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_32: "f32[4, 128, 64]" = torch.ops.aten.view.default(permute_23, [4, 128, 64]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:284, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_24: "f32[4, 128, 128]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    bmm_2: "f32[4, 128, 64]" = torch.ops.aten.bmm.default(permute_24, view_32);  permute_24 = None
    permute_25: "f32[4, 64, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_3: "f32[4, 128, 128]" = torch.ops.aten.bmm.default(view_32, permute_25);  view_32 = permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:261, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_3: "f32[4, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_23: "f32[4, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_3, alias_3);  bmm_3 = None
    sum_13: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_23, [-1], True)
    mul_24: "f32[4, 128, 128]" = torch.ops.aten.mul.Tensor(alias_3, sum_13);  alias_3 = sum_13 = None
    sub_9: "f32[4, 128, 128]" = torch.ops.aten.sub.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:259, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_33: "f32[1, 4, 128, 128]" = torch.ops.aten.view.default(sub_9, [1, 4, 128, 128]);  sub_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:258, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_34: "f32[4, 128, 128]" = torch.ops.aten.view.default(view_33, [4, 128, 128]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:245, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_26: "f32[4, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_4: "f32[4, 64, 128]" = torch.ops.aten.bmm.default(permute_26, view_34);  permute_26 = None
    permute_27: "f32[4, 128, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    bmm_5: "f32[4, 128, 64]" = torch.ops.aten.bmm.default(view_34, permute_27);  view_34 = permute_27 = None
    permute_28: "f32[4, 128, 64]" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:242, code: value_states = value_states.reshape(*proj_shape)
    view_35: "f32[1, 4, 128, 64]" = torch.ops.aten.view.default(bmm_2, [1, 4, 128, 64]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:242, code: value_states = value_states.reshape(*proj_shape)
    add_8: "f32[1, 4, 128, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_35);  tangents_3 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:241, code: key_states = key_states.reshape(*proj_shape)
    view_36: "f32[1, 4, 128, 64]" = torch.ops.aten.view.default(permute_28, [1, 4, 128, 64]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:241, code: key_states = key_states.reshape(*proj_shape)
    add_9: "f32[1, 4, 128, 64]" = torch.ops.aten.add.Tensor(tangents_2, view_36);  tangents_2 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:240, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_37: "f32[1, 4, 128, 64]" = torch.ops.aten.view.default(bmm_5, [1, 4, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_29: "f32[1, 128, 4, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
    clone_8: "f32[1, 128, 4, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_38: "f32[1, 128, 256]" = torch.ops.aten.view.default(clone_8, [1, 128, 256]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_30: "f32[1, 128, 4, 64]" = torch.ops.aten.permute.default(add_8, [0, 2, 1, 3]);  add_8 = None
    clone_9: "f32[1, 128, 4, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_39: "f32[1, 128, 256]" = torch.ops.aten.view.default(clone_9, [1, 128, 256]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_40: "f32[128, 256]" = torch.ops.aten.view.default(view_39, [128, 256]);  view_39 = None
    permute_31: "f32[256, 256]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_6: "f32[128, 256]" = torch.ops.aten.mm.default(view_40, permute_31);  permute_31 = None
    permute_32: "f32[256, 128]" = torch.ops.aten.permute.default(view_40, [1, 0])
    mm_7: "f32[256, 256]" = torch.ops.aten.mm.default(permute_32, view_5);  permute_32 = view_5 = None
    permute_33: "f32[256, 256]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_14: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
    view_41: "f32[256]" = torch.ops.aten.view.default(sum_14, [256]);  sum_14 = None
    permute_34: "f32[256, 256]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    view_42: "f32[1, 128, 256]" = torch.ops.aten.view.default(mm_6, [1, 128, 256]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_10: "f32[1, 128, 256]" = torch.ops.aten.add.Tensor(mul_19, view_42);  mul_19 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_35: "f32[1, 128, 4, 64]" = torch.ops.aten.permute.default(add_9, [0, 2, 1, 3]);  add_9 = None
    clone_10: "f32[1, 128, 4, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    view_43: "f32[1, 128, 256]" = torch.ops.aten.view.default(clone_10, [1, 128, 256]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_44: "f32[128, 256]" = torch.ops.aten.view.default(view_43, [128, 256]);  view_43 = None
    permute_36: "f32[256, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_8: "f32[128, 256]" = torch.ops.aten.mm.default(view_44, permute_36);  permute_36 = None
    permute_37: "f32[256, 128]" = torch.ops.aten.permute.default(view_44, [1, 0])
    mm_9: "f32[256, 256]" = torch.ops.aten.mm.default(permute_37, view_2);  permute_37 = view_2 = None
    permute_38: "f32[256, 256]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_15: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True);  view_44 = None
    view_45: "f32[256]" = torch.ops.aten.view.default(sum_15, [256]);  sum_15 = None
    permute_39: "f32[256, 256]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_46: "f32[1, 128, 256]" = torch.ops.aten.view.default(mm_8, [1, 128, 256]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_11: "f32[1, 128, 256]" = torch.ops.aten.add.Tensor(add_10, view_46);  add_10 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_25: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(view_38, 0.125);  view_38 = None
    view_47: "f32[128, 256]" = torch.ops.aten.view.default(mul_25, [128, 256]);  mul_25 = None
    permute_40: "f32[256, 256]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_10: "f32[128, 256]" = torch.ops.aten.mm.default(view_47, permute_40);  permute_40 = None
    permute_41: "f32[256, 128]" = torch.ops.aten.permute.default(view_47, [1, 0])
    mm_11: "f32[256, 256]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
    permute_42: "f32[256, 256]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_16: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True);  view_47 = None
    view_48: "f32[256]" = torch.ops.aten.view.default(sum_16, [256]);  sum_16 = None
    permute_43: "f32[256, 256]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_49: "f32[1, 128, 256]" = torch.ops.aten.view.default(mm_10, [1, 128, 256]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_12: "f32[1, 128, 256]" = torch.ops.aten.add.Tensor(add_11, view_49);  add_11 = view_49 = None
    return pytree.tree_unflatten([add_6, clone, clone_1, permute_43, view_48, permute_39, view_45, permute_34, view_41, permute_22, view_29, sum_10, sum_11, permute_18, view_26, permute_14, view_23, sum_4, sum_5, add_12, None], self._out_spec)
    