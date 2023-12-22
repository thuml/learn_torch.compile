from __future__ import annotations



def forward(self, primals_9: "f32[512]", primals_15: "f32[512]", view: "f32[128, 512]", view_16: "f32[128, 512]", getitem_1: "b8[1, 128, 512]", mul_1: "f32[1, 128, 512]", view_18: "f32[128, 512]", addmm_4: "f32[128, 2048]", view_20: "f32[128, 2048]", getitem_5: "b8[1, 128, 512]", mul_6: "f32[1, 128, 512]", div_1: "f32[1, 128, 1]", permute_11: "f32[512, 2048]", permute_15: "f32[2048, 512]", div_2: "f32[1, 128, 1]", permute_19: "f32[512, 512]", permute_24: "f32[16, 128, 128]", permute_25: "f32[16, 32, 128]", alias_1: "f32[16, 128, 128]", permute_26: "f32[16, 32, 128]", permute_27: "f32[16, 128, 32]", permute_31: "f32[512, 512]", permute_36: "f32[512, 512]", permute_40: "f32[512, 512]", tangents_1: "f32[1, 128, 512]", tangents_2: "f32[1, 16, 128, 32]", tangents_3: "f32[1, 16, 128, 32]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_19: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_4, [1, 128, 2048]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_4: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_4: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_9: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(tangents_1, primals_15);  primals_15 = None
    mul_10: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_9, 512)
    sum_2: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True)
    mul_11: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_9, mul_6);  mul_9 = None
    sum_3: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_11, [2], True);  mul_11 = None
    mul_12: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_6, sum_3);  sum_3 = None
    sub_4: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(mul_10, sum_2);  mul_10 = sum_2 = None
    sub_5: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(sub_4, mul_12);  sub_4 = mul_12 = None
    mul_13: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
    mul_14: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(tangents_1, mul_6);  mul_6 = None
    sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_14, [0, 1]);  mul_14 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_15: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_16: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_13, mul_15);  mul_15 = None
    clone_6: "f32[1, 128, 512]" = torch.ops.aten.clone.default(mul_16, memory_format = torch.contiguous_format);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_22: "f32[128, 512]" = torch.ops.aten.view.default(clone_6, [128, 512]);  clone_6 = None
    mm: "f32[128, 2048]" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
    permute_12: "f32[512, 128]" = torch.ops.aten.permute.default(view_22, [1, 0])
    mm_1: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_12, view_20);  permute_12 = view_20 = None
    permute_13: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_6: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
    view_23: "f32[512]" = torch.ops.aten.view.default(sum_6, [512]);  sum_6 = None
    permute_14: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    view_24: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm, [1, 128, 2048]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_18: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
    mul_19: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_20: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_19, -0.5);  mul_19 = None
    exp_1: "f32[1, 128, 2048]" = torch.ops.aten.exp.default(mul_20);  mul_20 = None
    mul_21: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_22: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_19, mul_21);  view_19 = mul_21 = None
    add_9: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_18, mul_22);  mul_18 = mul_22 = None
    mul_23: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_24, add_9);  view_24 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_25: "f32[128, 2048]" = torch.ops.aten.view.default(mul_23, [128, 2048]);  mul_23 = None
    mm_2: "f32[128, 512]" = torch.ops.aten.mm.default(view_25, permute_15);  permute_15 = None
    permute_16: "f32[2048, 128]" = torch.ops.aten.permute.default(view_25, [1, 0])
    mm_3: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_16, view_18);  permute_16 = view_18 = None
    permute_17: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_7: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True);  view_25 = None
    view_26: "f32[2048]" = torch.ops.aten.view.default(sum_7, [2048]);  sum_7 = None
    permute_18: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    view_27: "f32[1, 128, 512]" = torch.ops.aten.view.default(mm_2, [1, 128, 512]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_10: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_13, view_27);  mul_13 = view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_25: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_10, primals_9);  primals_9 = None
    mul_26: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_25, 512)
    sum_8: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [2], True)
    mul_27: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_25, mul_1);  mul_25 = None
    sum_9: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True);  mul_27 = None
    mul_28: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, sum_9);  sum_9 = None
    sub_7: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(mul_26, sum_8);  mul_26 = sum_8 = None
    sub_8: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(sub_7, mul_28);  sub_7 = mul_28 = None
    mul_29: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
    mul_30: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_10, mul_1);  mul_1 = None
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_30, [0, 1]);  mul_30 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_10, [0, 1]);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_31: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_32: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_29, mul_31);  mul_31 = None
    clone_7: "f32[1, 128, 512]" = torch.ops.aten.clone.default(mul_32, memory_format = torch.contiguous_format);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_28: "f32[128, 512]" = torch.ops.aten.view.default(clone_7, [128, 512]);  clone_7 = None
    mm_4: "f32[128, 512]" = torch.ops.aten.mm.default(view_28, permute_19);  permute_19 = None
    permute_20: "f32[512, 128]" = torch.ops.aten.permute.default(view_28, [1, 0])
    mm_5: "f32[512, 512]" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
    permute_21: "f32[512, 512]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_12: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
    view_29: "f32[512]" = torch.ops.aten.view.default(sum_12, [512]);  sum_12 = None
    permute_22: "f32[512, 512]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    view_30: "f32[1, 128, 512]" = torch.ops.aten.view.default(mm_4, [1, 128, 512]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_31: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_30, [1, 128, 16, 32]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_23: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_32: "f32[16, 128, 32]" = torch.ops.aten.view.default(permute_23, [16, 128, 32]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_2: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(permute_24, view_32);  permute_24 = None
    bmm_3: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_32, permute_25);  view_32 = permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_33: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_3, alias_1);  bmm_3 = None
    sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_33, [-1], True)
    mul_34: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_1, sum_13);  alias_1 = sum_13 = None
    sub_9: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_33: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(sub_9, [1, 16, 128, 128]);  sub_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_34: "f32[16, 128, 128]" = torch.ops.aten.view.default(view_33, [16, 128, 128]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_4: "f32[16, 32, 128]" = torch.ops.aten.bmm.default(permute_26, view_34);  permute_26 = None
    bmm_5: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(view_34, permute_27);  view_34 = permute_27 = None
    permute_28: "f32[16, 128, 32]" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_35: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_2, [1, 16, 128, 32]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    add_11: "f32[1, 16, 128, 32]" = torch.ops.aten.add.Tensor(tangents_3, view_35);  tangents_3 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_36: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(permute_28, [1, 16, 128, 32]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    add_12: "f32[1, 16, 128, 32]" = torch.ops.aten.add.Tensor(tangents_2, view_36);  tangents_2 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_37: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_5, [1, 16, 128, 32]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_29: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
    clone_8: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_38: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_8, [1, 128, 512]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_30: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(add_11, [0, 2, 1, 3]);  add_11 = None
    clone_9: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_39: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_9, [1, 128, 512]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_40: "f32[128, 512]" = torch.ops.aten.view.default(view_39, [128, 512]);  view_39 = None
    mm_6: "f32[128, 512]" = torch.ops.aten.mm.default(view_40, permute_31);  permute_31 = None
    permute_32: "f32[512, 128]" = torch.ops.aten.permute.default(view_40, [1, 0])
    mm_7: "f32[512, 512]" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
    permute_33: "f32[512, 512]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_14: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
    view_41: "f32[512]" = torch.ops.aten.view.default(sum_14, [512]);  sum_14 = None
    permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    view_42: "f32[1, 128, 512]" = torch.ops.aten.view.default(mm_6, [1, 128, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_13: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_29, view_42);  mul_29 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_35: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(add_12, [0, 2, 1, 3]);  add_12 = None
    clone_10: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    view_43: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_10, [1, 128, 512]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_44: "f32[128, 512]" = torch.ops.aten.view.default(view_43, [128, 512]);  view_43 = None
    mm_8: "f32[128, 512]" = torch.ops.aten.mm.default(view_44, permute_36);  permute_36 = None
    permute_37: "f32[512, 128]" = torch.ops.aten.permute.default(view_44, [1, 0])
    mm_9: "f32[512, 512]" = torch.ops.aten.mm.default(permute_37, view);  permute_37 = None
    permute_38: "f32[512, 512]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_15: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True);  view_44 = None
    view_45: "f32[512]" = torch.ops.aten.view.default(sum_15, [512]);  sum_15 = None
    permute_39: "f32[512, 512]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_46: "f32[1, 128, 512]" = torch.ops.aten.view.default(mm_8, [1, 128, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_14: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_13, view_46);  add_13 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_35: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_38, 0.1767766952966369);  view_38 = None
    view_47: "f32[128, 512]" = torch.ops.aten.view.default(mul_35, [128, 512]);  mul_35 = None
    mm_10: "f32[128, 512]" = torch.ops.aten.mm.default(view_47, permute_40);  permute_40 = None
    permute_41: "f32[512, 128]" = torch.ops.aten.permute.default(view_47, [1, 0])
    mm_11: "f32[512, 512]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
    permute_42: "f32[512, 512]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_16: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True);  view_47 = None
    view_48: "f32[512]" = torch.ops.aten.view.default(sum_16, [512]);  sum_16 = None
    permute_43: "f32[512, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_49: "f32[1, 128, 512]" = torch.ops.aten.view.default(mm_10, [1, 128, 512]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_15: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_14, view_49);  add_14 = view_49 = None
    return [permute_43, view_48, permute_39, view_45, permute_34, view_41, permute_22, view_29, sum_10, sum_11, permute_18, view_26, permute_14, view_23, sum_4, sum_5, add_15, None]
    