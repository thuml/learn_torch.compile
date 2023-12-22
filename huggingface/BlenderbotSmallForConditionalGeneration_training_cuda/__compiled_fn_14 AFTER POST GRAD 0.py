from __future__ import annotations



def forward(self, primals_1: "f32[512, 512]", primals_2: "f32[512]", primals_3: "f32[512, 512]", primals_4: "f32[512]", primals_5: "f32[512, 512]", primals_6: "f32[512]", primals_7: "f32[512, 512]", primals_8: "f32[512]", primals_9: "f32[512]", primals_10: "f32[512]", primals_11: "f32[2048, 512]", primals_12: "f32[2048]", primals_13: "f32[512, 2048]", primals_14: "f32[512]", primals_15: "f32[512]", primals_16: "f32[512]", primals_17: "f32[1, 128, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[128, 512]" = torch.ops.aten.reshape.default(primals_17, [128, 512])
    permute: "f32[512, 512]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[128, 512]" = torch.ops.aten.mm.default(view, permute)
    add_tensor_2: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_2, primals_2);  mm_default_2 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_1: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 128, 512]);  add_tensor_2 = None
    mul: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.1767766952966369);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[128, 512]" = torch.ops.aten.mm.default(view, permute_1)
    add_tensor_1: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_1, primals_4);  mm_default_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_3: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 128, 512]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_3, [1, -1, 16, 32]);  view_3 = None
    permute_2: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_3: "f32[512, 512]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[128, 512]" = torch.ops.aten.mm.default(view, permute_3)
    add_tensor: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default, primals_6);  mm_default = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_6: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor, [1, 128, 512]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_7: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_6, [1, -1, 16, 32]);  view_6 = None
    permute_4: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone_1: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul, [1, 128, 16, 32]);  mul = None
    permute_5: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_9: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_2, [16, -1, 32]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_10: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone, [16, -1, 32]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_11: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_1, [16, -1, 32]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_9, permute_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm, [-1], True)
    sub: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm, amax)
    exp: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
    sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_12: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 128, 32]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_4: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_13: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_4, [1, 128, 512]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_14: "f32[128, 512]" = torch.ops.aten.reshape.default(view_13, [128, 512]);  view_13 = None
    permute_8: "f32[512, 512]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_3: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_8, view_14, permute_8);  primals_8 = None
    view_15: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 512]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:323, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout = torch.ops.aten.native_dropout.default(view_15, 0.1, True);  view_15 = None
    getitem: "f32[1, 128, 512]" = native_dropout[0]
    getitem_1: "b8[1, 128, 512]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:324, code: hidden_states = residual + hidden_states
    add: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(primals_17, getitem);  primals_17 = getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean[0]
    getitem_3: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub_1: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add, getitem_3);  add = getitem_3 = None
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, primals_9)
    add_2: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_2, primals_10);  mul_2 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_16: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute_9: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm_4: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_12, view_16, permute_9);  primals_12 = None
    view_17: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_3: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    mul_4: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_17, 0.7071067811865476);  view_17 = None
    erf: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_3: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_5: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_3, add_3);  mul_3 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_18: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_5, [128, 2048]);  mul_5 = None
    permute_10: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_5: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_14, view_18, permute_10);  primals_14 = None
    view_19: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 512]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:331, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_19, 0.1, True);  view_19 = None
    getitem_4: "f32[1, 128, 512]" = native_dropout_1[0]
    getitem_5: "b8[1, 128, 512]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:332, code: hidden_states = residual + hidden_states
    add_4: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_2, getitem_4);  add_2 = getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_2: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
    mul_6: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_7: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_6, primals_15)
    add_6: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_7, primals_16);  mul_7 = primals_16 = None
    div_1: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 512);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    permute_11: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_15: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_2: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 512);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    permute_19: "f32[512, 512]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_24: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
    permute_25: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_26: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_27: "f32[16, 128, 32]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_31: "f32[512, 512]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_36: "f32[512, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_40: "f32[512, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [add_6, primals_9, primals_15, view, bmm, amax, sum_1, view_14, getitem_1, mul_1, view_16, addmm_4, view_18, getitem_5, mul_6, div_1, permute_11, permute_15, div_2, permute_19, permute_24, permute_25, permute_26, permute_27, permute_31, permute_36, permute_40]
    