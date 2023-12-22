from __future__ import annotations



def forward(self, primals_1: "f32[256, 256]", primals_2: "f32[256]", primals_3: "f32[256, 256]", primals_4: "f32[256]", primals_5: "f32[256, 256]", primals_6: "f32[256]", primals_7: "f32[256, 256]", primals_8: "f32[256]", primals_9: "f32[256]", primals_10: "f32[256]", primals_11: "f32[2048, 256]", primals_12: "f32[2048]", primals_13: "f32[256, 2048]", primals_14: "f32[256]", primals_15: "f32[256]", primals_16: "f32[256]", primals_17: "f32[1, 128, 256]", primals_18: "f32[1, 1, 128, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[128, 256]" = torch.ops.aten.view.default(primals_17, [128, 256])
    permute: "f32[256, 256]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    addmm: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = None
    view_1: "f32[1, 128, 256]" = torch.ops.aten.view.default(addmm, [1, 128, 256]);  addmm = None
    mul: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1: "f32[256, 256]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    addmm_1: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_4, view, permute_1);  primals_4 = None
    view_3: "f32[1, 128, 256]" = torch.ops.aten.view.default(addmm_1, [1, 128, 256]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:181, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 128, 4, 64]" = torch.ops.aten.view.default(view_3, [1, -1, 4, 64]);  view_3 = None
    permute_2: "f32[1, 4, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 4, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_3: "f32[256, 256]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    addmm_2: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_6, view, permute_3);  primals_6 = None
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
    sub_1: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(add_1, getitem_3);  add_1 = getitem_3 = None
    mul_1: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_1, primals_9)
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
    sub_2: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
    mul_3: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_4: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_3, primals_15)
    add_6: "f32[1, 128, 256]" = torch.ops.aten.add.Tensor(mul_4, primals_16);  mul_4 = primals_16 = None
    div_1: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 256);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:408, code: hidden_states = self.fc2(hidden_states)
    permute_11: "f32[256, 2048]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:406, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    alias_2: "f32[1, 128, 2048]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    le: "b8[1, 128, 2048]" = torch.ops.aten.le.Scalar(alias_2, 0);  alias_2 = None
    permute_15: "f32[2048, 256]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:379, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_2: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 256);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:299, code: attn_output = self.out_proj(attn_output)
    permute_19: "f32[256, 256]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:284, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_24: "f32[4, 128, 128]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    permute_25: "f32[4, 64, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:261, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_3: "f32[4, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:245, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_26: "f32[4, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_27: "f32[4, 128, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:227, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_31: "f32[256, 256]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:226, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_36: "f32[256, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:201, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_40: "f32[256, 256]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [add_6, clone, clone_1, primals_9, primals_15, view, view_16, getitem_1, mul_1, view_18, view_20, getitem_5, mul_3, div_1, permute_11, le, permute_15, div_2, permute_19, permute_24, permute_25, alias_3, permute_26, permute_27, permute_31, permute_36, permute_40]
    