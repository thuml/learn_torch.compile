from __future__ import annotations



def forward(self, primals_1: "f32[512, 512]", primals_2: "f32[512]", primals_3: "f32[512, 512]", primals_4: "f32[512]", primals_5: "f32[512, 512]", primals_6: "f32[512]", primals_7: "f32[512, 512]", primals_8: "f32[512]", primals_9: "f32[512]", primals_10: "f32[512]", primals_11: "f32[512, 512]", primals_12: "f32[512]", primals_13: "f32[512, 512]", primals_14: "f32[512]", primals_15: "f32[512, 512]", primals_16: "f32[512]", primals_17: "f32[512, 512]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "f32[512]", primals_21: "f32[2048, 512]", primals_22: "f32[2048]", primals_23: "f32[512, 2048]", primals_24: "f32[512]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[1, 128, 512]", primals_28: "f32[1, 1, 128, 128]", primals_29: "f32[1, 128, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view: "f32[128, 512]" = torch.ops.aten.reshape.default(primals_27, [128, 512])
    permute: "f32[512, 512]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[128, 512]" = torch.ops.aten.mm.default(view, permute)
    add_tensor_5: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_5, primals_2);  mm_default_5 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_1: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 128, 512]);  add_tensor_5 = None
    mul: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.1767766952966369);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[128, 512]" = torch.ops.aten.mm.default(view, permute_1)
    add_tensor_4: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_4, primals_4);  mm_default_4 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_3: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 128, 512]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_4: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_3, [1, -1, 16, 32]);  view_3 = None
    permute_2: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    clone: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_3: "f32[512, 512]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[128, 512]" = torch.ops.aten.mm.default(view, permute_3)
    add_tensor_3: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_3, primals_6);  mm_default_3 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_6: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 128, 512]);  add_tensor_3 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_12: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 16, 128, 128]);  bmm = None
    add: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_12, primals_28);  view_12 = primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_13: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add, [16, 128, 128]);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_13, [-1], True)
    sub: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_13, amax);  view_13 = amax = None
    exp: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
    sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[16, 128, 128]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_3: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(clone_3, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_14: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 128, 32]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_4: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_15: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_4, [1, 128, 512]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_16: "f32[128, 512]" = torch.ops.aten.reshape.default(view_15, [128, 512]);  view_15 = None
    permute_8: "f32[512, 512]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm_3: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_8, view_16, permute_8);  primals_8 = None
    view_17: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 512]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem: "f32[1, 128, 512]" = native_dropout[0]
    getitem_1: "b8[1, 128, 512]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_1: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(primals_27, getitem);  primals_27 = getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean[0]
    getitem_3: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_1, getitem_3);  add_1 = getitem_3 = None
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, primals_9)
    add_3: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_2, primals_10);  mul_2 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_18: "f32[128, 512]" = torch.ops.aten.reshape.default(add_3, [128, 512])
    permute_9: "f32[512, 512]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[128, 512]" = torch.ops.aten.mm.default(view_18, permute_9)
    add_tensor_2: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_2, primals_12);  mm_default_2 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_19: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 128, 512]);  add_tensor_2 = None
    mul_3: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_19, 0.1767766952966369);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_20: "f32[128, 512]" = torch.ops.aten.reshape.default(primals_29, [128, 512]);  primals_29 = None
    permute_10: "f32[512, 512]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[128, 512]" = torch.ops.aten.mm.default(view_20, permute_10)
    add_tensor_1: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_1, primals_14);  mm_default_1 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_21: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 128, 512]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_22: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_21, [1, -1, 16, 32]);  view_21 = None
    permute_11: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    clone_5: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[128, 512]" = torch.ops.aten.mm.default(view_20, permute_12)
    add_tensor: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default, primals_16);  mm_default = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_24: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor, [1, 128, 512]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_24, [1, -1, 16, 32]);  view_24 = None
    permute_13: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_6: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_26: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_3, [1, 128, 16, 32]);  mul_3 = None
    permute_14: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    clone_7: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_27: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_7, [16, -1, 32]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_28: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_5, [16, -1, 32]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_29: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_6, [16, -1, 32]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_15: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_27, permute_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[16, 128, 1]" = torch.ops.aten.amax.default(bmm_2, [-1], True)
    sub_2: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1)
    exp_1: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_2: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div_1, view_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_30: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_3, [1, 16, 128, 32]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_16: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_9: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_31: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_9, [1, 128, 512]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_32: "f32[128, 512]" = torch.ops.aten.reshape.default(view_31, [128, 512]);  view_31 = None
    permute_17: "f32[512, 512]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_7: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_18, view_32, permute_17);  primals_18 = None
    view_33: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 512]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:440, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_33, 0.1, True);  view_33 = None
    getitem_4: "f32[1, 128, 512]" = native_dropout_1[0]
    getitem_5: "b8[1, 128, 512]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    add_4: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_3, getitem_4);  add_3 = getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_3: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
    mul_4: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_5: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_4, primals_19)
    add_6: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_5, primals_20);  mul_5 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_34: "f32[128, 512]" = torch.ops.aten.reshape.default(add_6, [128, 512])
    permute_18: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_8: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_22, view_34, permute_18);  primals_22 = None
    view_35: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    mul_7: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476);  view_35 = None
    erf: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_7: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_6, add_7);  mul_6 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_36: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_8, [128, 2048]);  mul_8 = None
    permute_19: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_9: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_24, view_36, permute_19);  primals_24 = None
    view_37: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 512]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_37, 0.1, True);  view_37 = None
    getitem_8: "f32[1, 128, 512]" = native_dropout_2[0]
    getitem_9: "b8[1, 128, 512]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_8: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_6, getitem_8);  add_6 = getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_4: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_8, getitem_11);  add_8 = getitem_11 = None
    mul_9: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_10: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_9, primals_25)
    add_10: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_10, primals_26);  mul_10 = primals_26 = None
    div_2: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 512);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    permute_20: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    permute_24: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    div_3: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 512);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    permute_28: "f32[512, 512]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_33: "f32[16, 128, 128]" = torch.ops.aten.permute.default(div_1, [0, 2, 1]);  div_1 = None
    permute_34: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_35: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    permute_36: "f32[16, 128, 32]" = torch.ops.aten.permute.default(permute_15, [0, 2, 1]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    permute_40: "f32[512, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    permute_45: "f32[512, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_49: "f32[512, 512]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    div_4: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 512);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    permute_58: "f32[16, 128, 128]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    permute_59: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_3: "f32[16, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_60: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_61: "f32[16, 128, 32]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    permute_65: "f32[512, 512]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    permute_70: "f32[512, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    permute_74: "f32[512, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [add_10, primals_9, primals_19, primals_25, view, view_16, getitem_1, mul_1, view_18, view_20, bmm_2, amax_1, sum_2, view_32, getitem_5, mul_4, view_34, addmm_8, view_36, getitem_9, mul_9, div_2, permute_20, permute_24, div_3, permute_28, permute_33, permute_34, permute_35, permute_36, permute_40, permute_45, permute_49, div_4, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74]
    