from __future__ import annotations



def forward(self, primals_1: "f32[768]", primals_11: "f32[768]", primals_17: "f32[1, 2048, 768]", getitem_1: "f32[1, 2048, 1]", rsqrt: "f32[1, 2048, 1]", view: "f32[2048, 768]", view_16: "f32[2048, 768]", getitem_3: "b8[1, 2048, 768]", mul_3: "f32[2048, 768]", add_5: "f32[2048, 768]", relu: "f32[2048, 3072]", getitem_7: "b8[2048, 768]", permute_11: "f32[768, 3072]", permute_15: "f32[3072, 768]", div_1: "f32[2048, 1]", permute_19: "f32[768, 768]", permute_24: "f32[12, 2048, 2048]", permute_25: "f32[12, 64, 2048]", alias_3: "f32[12, 2048, 2048]", eq: "b8[1, 12, 2048, 2048]", lt: "b8[1, 12, 2048, 2048]", permute_26: "f32[12, 64, 2048]", permute_27: "f32[12, 2048, 64]", permute_31: "f32[768, 768]", permute_36: "f32[768, 768]", permute_40: "f32[768, 768]", tangents_1: "f32[1, 2048, 768]", tangents_2: "f32[1, 12, 2048, 64]", tangents_3: "f32[1, 12, 2048, 64]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(primals_17, getitem_1);  primals_17 = getitem_1 = None
    mul: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    view_20: "f32[2048, 768]" = torch.ops.aten.reshape.default(tangents_1, [2048, 768]);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[2048, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_5: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_6: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(view_20, mul_5);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    mm: "f32[2048, 3072]" = torch.ops.aten.mm.default(mul_6, permute_11);  permute_11 = None
    permute_12: "f32[768, 2048]" = torch.ops.aten.permute.default(mul_6, [1, 0])
    mm_1: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_12, relu);  permute_12 = None
    permute_13: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_2: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_6, [0], True);  mul_6 = None
    view_21: "f32[768]" = torch.ops.aten.reshape.default(sum_2, [768]);  sum_2 = None
    permute_14: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    le: "b8[2048, 3072]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[2048, 3072]" = torch.ops.aten.where.self(le, full_default_1, mm);  le = mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    mm_2: "f32[2048, 768]" = torch.ops.aten.mm.default(where, permute_15);  permute_15 = None
    permute_16: "f32[3072, 2048]" = torch.ops.aten.permute.default(where, [1, 0])
    mm_3: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_16, add_5);  permute_16 = add_5 = None
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_3: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
    view_22: "f32[3072]" = torch.ops.aten.reshape.default(sum_3, [3072]);  sum_3 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_8: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mm_2, primals_11);  primals_11 = None
    mul_9: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_8, 768)
    sum_4: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [1], True)
    mul_10: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_8, mul_3);  mul_8 = None
    sum_5: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_10, [1], True);  mul_10 = None
    mul_11: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_5);  sum_5 = None
    sub_4: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(mul_9, sum_4);  mul_9 = sum_4 = None
    sub_5: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(sub_4, mul_11);  sub_4 = mul_11 = None
    mul_12: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
    mul_13: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mm_2, mul_3);  mul_3 = None
    sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_13, [0]);  mul_13 = None
    sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(mm_2, [0]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    add_7: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_20, mul_12);  view_20 = mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_23: "f32[1, 2048, 768]" = torch.ops.aten.reshape.default(add_7, [1, 2048, 768]);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 2048, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_14: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_15: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_23, mul_14);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_24: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_15, [2048, 768]);  mul_15 = None
    mm_4: "f32[2048, 768]" = torch.ops.aten.mm.default(view_24, permute_19);  permute_19 = None
    permute_20: "f32[768, 2048]" = torch.ops.aten.permute.default(view_24, [1, 0])
    mm_5: "f32[768, 768]" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
    permute_21: "f32[768, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_24, [0], True);  view_24 = None
    view_25: "f32[768]" = torch.ops.aten.reshape.default(sum_8, [768]);  sum_8 = None
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    view_26: "f32[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_4, [1, 2048, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_27: "f32[1, 2048, 12, 64]" = torch.ops.aten.reshape.default(view_26, [1, 2048, 12, 64]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_23: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_28: "f32[12, 2048, 64]" = torch.ops.aten.reshape.default(permute_23, [12, 2048, 64]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_2: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(permute_24, view_28);  permute_24 = None
    bmm_3: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_28, permute_25);  view_28 = permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_16: "f32[12, 2048, 2048]" = torch.ops.aten.mul.Tensor(bmm_3, alias_3);  bmm_3 = None
    sum_9: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_16, [-1], True)
    mul_17: "f32[12, 2048, 2048]" = torch.ops.aten.mul.Tensor(alias_3, sum_9);  alias_3 = sum_9 = None
    sub_6: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_29: "f32[1, 12, 2048, 2048]" = torch.ops.aten.reshape.default(sub_6, [1, 12, 2048, 2048]);  sub_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    div_2: "f32[1, 12, 2048, 2048]" = torch.ops.aten.div.Scalar(view_29, 2)
    where_1: "f32[1, 12, 2048, 2048]" = torch.ops.aten.where.self(eq, div_2, view_29);  eq = div_2 = view_29 = None
    where_2: "f32[1, 12, 2048, 2048]" = torch.ops.aten.where.self(lt, full_default_1, where_1);  lt = full_default_1 = where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    view_31: "f32[12, 2048, 2048]" = torch.ops.aten.reshape.default(where_2, [12, 2048, 2048]);  where_2 = None
    bmm_4: "f32[12, 64, 2048]" = torch.ops.aten.bmm.default(permute_26, view_31);  permute_26 = None
    bmm_5: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(view_31, permute_27);  view_31 = permute_27 = None
    permute_28: "f32[12, 2048, 64]" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_32: "f32[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 2048, 64]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    add_8: "f32[1, 12, 2048, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_32);  tangents_3 = view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_33: "f32[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(permute_28, [1, 12, 2048, 64]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    add_9: "f32[1, 12, 2048, 64]" = torch.ops.aten.add.Tensor(tangents_2, view_33);  tangents_2 = view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_34: "f32[1, 12, 2048, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 2048, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_29: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
    clone_7: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_35: "f32[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 2048, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_30: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(add_8, [0, 2, 1, 3]);  add_8 = None
    clone_8: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_36: "f32[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_8, [1, 2048, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_37: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_36, [2048, 768]);  view_36 = None
    mm_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_37, permute_31);  permute_31 = None
    permute_32: "f32[768, 2048]" = torch.ops.aten.permute.default(view_37, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True);  view_37 = None
    view_38: "f32[768]" = torch.ops.aten.reshape.default(sum_10, [768]);  sum_10 = None
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    view_39: "f32[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_6, [1, 2048, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_35: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(add_9, [0, 2, 1, 3]);  add_9 = None
    clone_9: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    view_40: "f32[1, 2048, 768]" = torch.ops.aten.reshape.default(clone_9, [1, 2048, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_41: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_40, [2048, 768]);  view_40 = None
    mm_8: "f32[2048, 768]" = torch.ops.aten.mm.default(view_41, permute_36);  permute_36 = None
    permute_37: "f32[768, 2048]" = torch.ops.aten.permute.default(view_41, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_37, view);  permute_37 = None
    permute_38: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
    view_42: "f32[768]" = torch.ops.aten.reshape.default(sum_11, [768]);  sum_11 = None
    permute_39: "f32[768, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_43: "f32[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_8, [1, 2048, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_10: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_39, view_43);  view_39 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_18: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_35, 0.125);  view_35 = None
    view_44: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_18, [2048, 768]);  mul_18 = None
    mm_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_44, permute_40);  permute_40 = None
    permute_41: "f32[768, 2048]" = torch.ops.aten.permute.default(view_44, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True);  view_44 = None
    view_45: "f32[768]" = torch.ops.aten.reshape.default(sum_12, [768]);  sum_12 = None
    permute_43: "f32[768, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_46: "f32[1, 2048, 768]" = torch.ops.aten.reshape.default(mm_10, [1, 2048, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_11: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(add_10, view_46);  add_10 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_20: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(add_11, primals_1);  primals_1 = None
    mul_21: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_20, 768)
    sum_13: "f32[1, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True)
    mul_22: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_20, mul);  mul_20 = None
    sum_14: "f32[1, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True);  mul_22 = None
    mul_23: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul, sum_14);  sum_14 = None
    sub_8: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(mul_21, sum_13);  mul_21 = sum_13 = None
    sub_9: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(sub_8, mul_23);  sub_8 = mul_23 = None
    div_3: "f32[1, 2048, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_24: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_9);  div_3 = sub_9 = None
    mul_25: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(add_11, mul);  mul = None
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_25, [0, 1]);  mul_25 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_12: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_23, mul_24);  view_23 = mul_24 = None
    return [sum_15, sum_16, permute_43, view_45, permute_39, view_42, permute_34, view_38, permute_22, view_25, sum_6, sum_7, permute_18, view_22, permute_14, view_21, add_12, None]
    