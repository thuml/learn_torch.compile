from __future__ import annotations



def forward(self, primals_9: "f32[512]", primals_19: "f32[512]", primals_25: "f32[512]", view: "f32[128, 512]", view_16: "f32[128, 512]", getitem_1: "b8[1, 128, 512]", mul_1: "f32[1, 128, 512]", view_18: "f32[128, 512]", view_20: "f32[128, 512]", bmm_2: "f32[16, 128, 128]", amax_1: "f32[16, 128, 1]", sum_2: "f32[16, 128, 1]", view_32: "f32[128, 512]", getitem_5: "b8[1, 128, 512]", mul_4: "f32[1, 128, 512]", view_34: "f32[128, 512]", addmm_8: "f32[128, 2048]", view_36: "f32[128, 2048]", getitem_9: "b8[1, 128, 512]", mul_9: "f32[1, 128, 512]", div_2: "f32[1, 128, 1]", permute_20: "f32[512, 2048]", permute_24: "f32[2048, 512]", div_3: "f32[1, 128, 1]", permute_28: "f32[512, 512]", permute_33: "f32[16, 128, 128]", permute_34: "f32[16, 32, 128]", permute_35: "f32[16, 32, 128]", permute_36: "f32[16, 128, 32]", permute_40: "f32[512, 512]", permute_45: "f32[512, 512]", permute_49: "f32[512, 512]", div_4: "f32[1, 128, 1]", permute_53: "f32[512, 512]", permute_58: "f32[16, 128, 128]", permute_59: "f32[16, 32, 128]", alias_3: "f32[16, 128, 128]", permute_60: "f32[16, 32, 128]", permute_61: "f32[16, 128, 32]", permute_65: "f32[512, 512]", permute_70: "f32[512, 512]", permute_74: "f32[512, 512]", tangents_1: "f32[1, 128, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_2: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1);  bmm_2 = amax_1 = None
    exp_1: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    div_1: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_35: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 2048]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_7: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476)
    erf: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_7: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_12: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(tangents_1, primals_25);  primals_25 = None
    mul_13: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_12, 512)
    sum_3: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_12, [2], True)
    mul_14: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_12, mul_9);  mul_12 = None
    sum_4: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_14, [2], True);  mul_14 = None
    mul_15: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_9, sum_4);  sum_4 = None
    sub_6: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(mul_13, sum_3);  mul_13 = sum_3 = None
    sub_7: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(sub_6, mul_15);  sub_6 = mul_15 = None
    mul_16: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_2, sub_7);  div_2 = sub_7 = None
    mul_17: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(tangents_1, mul_9);  mul_9 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_17, [0, 1]);  mul_17 = None
    sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_9, torch.float32);  getitem_9 = None
    mul_18: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_19: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_16, mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_38: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_19, [128, 512]);  mul_19 = None
    mm: "f32[128, 2048]" = torch.ops.aten.mm.default(view_38, permute_20);  permute_20 = None
    permute_21: "f32[512, 128]" = torch.ops.aten.permute.default(view_38, [1, 0])
    mm_1: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_21, view_36);  permute_21 = view_36 = None
    permute_22: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_7: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_38, [0], True);  view_38 = None
    view_39: "f32[512]" = torch.ops.aten.reshape.default(sum_7, [512]);  sum_7 = None
    permute_23: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    view_40: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm, [1, 128, 2048]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_21: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
    mul_22: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_35, view_35)
    mul_23: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_22, -0.5);  mul_22 = None
    exp_2: "f32[1, 128, 2048]" = torch.ops.aten.exp.default(mul_23);  mul_23 = None
    mul_24: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_25: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_35, mul_24);  view_35 = mul_24 = None
    add_12: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_21, mul_25);  mul_21 = mul_25 = None
    mul_26: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_40, add_12);  view_40 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_41: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_26, [128, 2048]);  mul_26 = None
    mm_2: "f32[128, 512]" = torch.ops.aten.mm.default(view_41, permute_24);  permute_24 = None
    permute_25: "f32[2048, 128]" = torch.ops.aten.permute.default(view_41, [1, 0])
    mm_3: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_25, view_34);  permute_25 = view_34 = None
    permute_26: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_8: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
    view_42: "f32[2048]" = torch.ops.aten.reshape.default(sum_8, [2048]);  sum_8 = None
    permute_27: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    view_43: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 512]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_13: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_16, view_43);  mul_16 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_28: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_13, primals_19);  primals_19 = None
    mul_29: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_28, 512)
    sum_9: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_28, [2], True)
    mul_30: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_28, mul_4);  mul_28 = None
    sum_10: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_30, [2], True);  mul_30 = None
    mul_31: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_4, sum_10);  sum_10 = None
    sub_9: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(mul_29, sum_9);  mul_29 = sum_9 = None
    sub_10: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(sub_9, mul_31);  sub_9 = mul_31 = None
    mul_32: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_3, sub_10);  div_3 = sub_10 = None
    mul_33: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_13, mul_4);  mul_4 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_33, [0, 1]);  mul_33 = None
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_13, [0, 1]);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:440, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_34: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_35: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_32, mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_44: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_35, [128, 512]);  mul_35 = None
    mm_4: "f32[128, 512]" = torch.ops.aten.mm.default(view_44, permute_28);  permute_28 = None
    permute_29: "f32[512, 128]" = torch.ops.aten.permute.default(view_44, [1, 0])
    mm_5: "f32[512, 512]" = torch.ops.aten.mm.default(permute_29, view_32);  permute_29 = view_32 = None
    permute_30: "f32[512, 512]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_13: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True);  view_44 = None
    view_45: "f32[512]" = torch.ops.aten.reshape.default(sum_13, [512]);  sum_13 = None
    permute_31: "f32[512, 512]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    view_46: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 512]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_47: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_46, [1, 128, 16, 32]);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_32: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_48: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(permute_32, [16, 128, 32]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_4: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(permute_33, view_48);  permute_33 = None
    bmm_5: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_48, permute_34);  view_48 = permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_36: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_5, div_1);  bmm_5 = None
    sum_14: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_36, [-1], True)
    mul_37: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_1, sum_14);  div_1 = sum_14 = None
    sub_11: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_6: "f32[16, 32, 128]" = torch.ops.aten.bmm.default(permute_35, sub_11);  permute_35 = None
    bmm_7: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(sub_11, permute_36);  sub_11 = permute_36 = None
    permute_37: "f32[16, 128, 32]" = torch.ops.aten.permute.default(bmm_6, [0, 2, 1]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_49: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_4, [1, 16, 128, 32]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_50: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(permute_37, [1, 16, 128, 32]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_51: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_7, [1, 16, 128, 32]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_38: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    clone_13: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_52: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_13, [1, 128, 512]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_39: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_14: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    view_53: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_14, [1, 128, 512]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_54: "f32[128, 512]" = torch.ops.aten.reshape.default(view_53, [128, 512]);  view_53 = None
    mm_6: "f32[128, 512]" = torch.ops.aten.mm.default(view_54, permute_40);  permute_40 = None
    permute_41: "f32[512, 128]" = torch.ops.aten.permute.default(view_54, [1, 0])
    mm_7: "f32[512, 512]" = torch.ops.aten.mm.default(permute_41, view_20);  permute_41 = None
    permute_42: "f32[512, 512]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_15: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_54, [0], True);  view_54 = None
    view_55: "f32[512]" = torch.ops.aten.reshape.default(sum_15, [512]);  sum_15 = None
    permute_43: "f32[512, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_56: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_44: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    view_57: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(permute_44, [1, 128, 512]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_58: "f32[128, 512]" = torch.ops.aten.reshape.default(view_57, [128, 512]);  view_57 = None
    mm_8: "f32[128, 512]" = torch.ops.aten.mm.default(view_58, permute_45);  permute_45 = None
    permute_46: "f32[512, 128]" = torch.ops.aten.permute.default(view_58, [1, 0])
    mm_9: "f32[512, 512]" = torch.ops.aten.mm.default(permute_46, view_20);  permute_46 = view_20 = None
    permute_47: "f32[512, 512]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_16: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_58, [0], True);  view_58 = None
    view_59: "f32[512]" = torch.ops.aten.reshape.default(sum_16, [512]);  sum_16 = None
    permute_48: "f32[512, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    view_60: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_14: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_56, view_60);  view_56 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_38: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_52, 0.1767766952966369);  view_52 = None
    view_61: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_38, [128, 512]);  mul_38 = None
    mm_10: "f32[128, 512]" = torch.ops.aten.mm.default(view_61, permute_49);  permute_49 = None
    permute_50: "f32[512, 128]" = torch.ops.aten.permute.default(view_61, [1, 0])
    mm_11: "f32[512, 512]" = torch.ops.aten.mm.default(permute_50, view_18);  permute_50 = view_18 = None
    permute_51: "f32[512, 512]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_17: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_61, [0], True);  view_61 = None
    view_62: "f32[512]" = torch.ops.aten.reshape.default(sum_17, [512]);  sum_17 = None
    permute_52: "f32[512, 512]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    view_63: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 512]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_15: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_32, view_63);  mul_32 = view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_40: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_15, primals_9);  primals_9 = None
    mul_41: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_40, 512)
    sum_18: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_40, [2], True)
    mul_42: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_40, mul_1);  mul_40 = None
    sum_19: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_42, [2], True);  mul_42 = None
    mul_43: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, sum_19);  sum_19 = None
    sub_13: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(mul_41, sum_18);  mul_41 = sum_18 = None
    sub_14: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(sub_13, mul_43);  sub_13 = mul_43 = None
    mul_44: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_4, sub_14);  div_4 = sub_14 = None
    mul_45: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_15, mul_1);  mul_1 = None
    sum_20: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_45, [0, 1]);  mul_45 = None
    sum_21: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_15, [0, 1]);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_2: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_46: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_47: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_44, mul_46);  mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_64: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_47, [128, 512]);  mul_47 = None
    mm_12: "f32[128, 512]" = torch.ops.aten.mm.default(view_64, permute_53);  permute_53 = None
    permute_54: "f32[512, 128]" = torch.ops.aten.permute.default(view_64, [1, 0])
    mm_13: "f32[512, 512]" = torch.ops.aten.mm.default(permute_54, view_16);  permute_54 = view_16 = None
    permute_55: "f32[512, 512]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_22: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_64, [0], True);  view_64 = None
    view_65: "f32[512]" = torch.ops.aten.reshape.default(sum_22, [512]);  sum_22 = None
    permute_56: "f32[512, 512]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    view_66: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 512]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_67: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_66, [1, 128, 16, 32]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_57: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_68: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(permute_57, [16, 128, 32]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_8: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(permute_58, view_68);  permute_58 = None
    bmm_9: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_68, permute_59);  view_68 = permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_48: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_9, alias_3);  bmm_9 = None
    sum_23: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_48, [-1], True)
    mul_49: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_3, sum_23);  alias_3 = sum_23 = None
    sub_15: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_69: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_15, [1, 16, 128, 128]);  sub_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_70: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_69, [16, 128, 128]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_10: "f32[16, 32, 128]" = torch.ops.aten.bmm.default(permute_60, view_70);  permute_60 = None
    bmm_11: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(view_70, permute_61);  view_70 = permute_61 = None
    permute_62: "f32[16, 128, 32]" = torch.ops.aten.permute.default(bmm_10, [0, 2, 1]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_71: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_8, [1, 16, 128, 32]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_72: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(permute_62, [1, 16, 128, 32]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_73: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_11, [1, 16, 128, 32]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_63: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    clone_16: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_74: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_16, [1, 128, 512]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_64: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    clone_17: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    view_75: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_17, [1, 128, 512]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_76: "f32[128, 512]" = torch.ops.aten.reshape.default(view_75, [128, 512]);  view_75 = None
    mm_14: "f32[128, 512]" = torch.ops.aten.mm.default(view_76, permute_65);  permute_65 = None
    permute_66: "f32[512, 128]" = torch.ops.aten.permute.default(view_76, [1, 0])
    mm_15: "f32[512, 512]" = torch.ops.aten.mm.default(permute_66, view);  permute_66 = None
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_24: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_76, [0], True);  view_76 = None
    view_77: "f32[512]" = torch.ops.aten.reshape.default(sum_24, [512]);  sum_24 = None
    permute_68: "f32[512, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    view_78: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_16: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_44, view_78);  mul_44 = view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_69: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    view_79: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(permute_69, [1, 128, 512]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_80: "f32[128, 512]" = torch.ops.aten.reshape.default(view_79, [128, 512]);  view_79 = None
    mm_16: "f32[128, 512]" = torch.ops.aten.mm.default(view_80, permute_70);  permute_70 = None
    permute_71: "f32[512, 128]" = torch.ops.aten.permute.default(view_80, [1, 0])
    mm_17: "f32[512, 512]" = torch.ops.aten.mm.default(permute_71, view);  permute_71 = None
    permute_72: "f32[512, 512]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_25: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_80, [0], True);  view_80 = None
    view_81: "f32[512]" = torch.ops.aten.reshape.default(sum_25, [512]);  sum_25 = None
    permute_73: "f32[512, 512]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    view_82: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 512]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_17: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_16, view_82);  add_16 = view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_50: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_74, 0.1767766952966369);  view_74 = None
    view_83: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_50, [128, 512]);  mul_50 = None
    mm_18: "f32[128, 512]" = torch.ops.aten.mm.default(view_83, permute_74);  permute_74 = None
    permute_75: "f32[512, 128]" = torch.ops.aten.permute.default(view_83, [1, 0])
    mm_19: "f32[512, 512]" = torch.ops.aten.mm.default(permute_75, view);  permute_75 = view = None
    permute_76: "f32[512, 512]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_26: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_83, [0], True);  view_83 = None
    view_84: "f32[512]" = torch.ops.aten.reshape.default(sum_26, [512]);  sum_26 = None
    permute_77: "f32[512, 512]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    view_85: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 512]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_18: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_17, view_85);  add_17 = view_85 = None
    return [permute_77, view_84, permute_73, view_81, permute_68, view_77, permute_56, view_65, sum_20, sum_21, permute_52, view_62, permute_48, view_59, permute_43, view_55, permute_31, view_45, sum_11, sum_12, permute_27, view_42, permute_23, view_39, sum_5, sum_6, add_18, None, add_14]
    