from __future__ import annotations



def forward(self, primals_1: "f32[1024]", primals_11: "f32[1024]", primals_17: "f32[1, 128, 1024]", getitem_1: "f32[1, 128, 1]", rsqrt: "f32[1, 128, 1]", view: "f32[128, 1024]", getitem_3: "b8[16, 128, 128]", view_16: "f32[128, 1024]", getitem_5: "b8[1, 128, 1024]", mul_3: "f32[1, 128, 1024]", view_18: "f32[128, 1024]", addmm_4: "f32[128, 4096]", view_20: "f32[128, 4096]", getitem_9: "b8[1, 128, 1024]", permute_11: "f32[1024, 4096]", permute_15: "f32[4096, 1024]", div_1: "f32[1, 128, 1]", permute_19: "f32[1024, 1024]", permute_24: "f32[16, 128, 128]", permute_25: "f32[16, 64, 128]", alias_1: "f32[16, 128, 128]", eq: "b8[1, 16, 128, 128]", lt: "b8[1, 16, 128, 128]", permute_26: "f32[16, 64, 128]", permute_27: "f32[16, 128, 64]", permute_31: "f32[1024, 1024]", permute_36: "f32[1024, 1024]", permute_40: "f32[1024, 1024]", tangents_1: "f32[1, 128, 1024]", tangents_2: "f32[1, 16, 128, 64]", tangents_3: "f32[1, 16, 128, 64]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_17, getitem_1);  primals_17 = getitem_1 = None
    mul: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_19: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_6: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_9, torch.float32);  getitem_9 = None
    mul_8: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_9: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_22: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_9, [128, 1024]);  mul_9 = None
    mm: "f32[128, 4096]" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
    permute_12: "f32[1024, 128]" = torch.ops.aten.permute.default(view_22, [1, 0])
    mm_1: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_12, view_20);  permute_12 = view_20 = None
    permute_13: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
    view_23: "f32[1024]" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
    permute_14: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    view_24: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm, [1, 128, 4096]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_11: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
    mul_12: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_13: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_12, -0.5);  mul_12 = None
    exp_1: "f32[1, 128, 4096]" = torch.ops.aten.exp.default(mul_13);  mul_13 = None
    mul_14: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_15: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_19, mul_14);  view_19 = mul_14 = None
    add_9: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_11, mul_15);  mul_11 = mul_15 = None
    mul_16: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_24, add_9);  view_24 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_25: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_16, [128, 4096]);  mul_16 = None
    mm_2: "f32[128, 1024]" = torch.ops.aten.mm.default(view_25, permute_15);  permute_15 = None
    permute_16: "f32[4096, 128]" = torch.ops.aten.permute.default(view_25, [1, 0])
    mm_3: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_16, view_18);  permute_16 = view_18 = None
    permute_17: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True);  view_25 = None
    view_26: "f32[4096]" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
    permute_18: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    view_27: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 1024]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_18: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_27, primals_11);  primals_11 = None
    mul_19: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_18, 1024)
    sum_4: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
    mul_20: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_18, mul_3);  mul_18 = None
    sum_5: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
    mul_21: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_3, sum_5);  sum_5 = None
    sub_4: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_19, sum_4);  mul_19 = sum_4 = None
    sub_5: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_4, mul_21);  sub_4 = mul_21 = None
    mul_22: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
    mul_23: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_27, mul_3);  mul_3 = None
    sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
    sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_27, [0, 1]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    add_10: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tangents_1, mul_22);  tangents_1 = mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_24: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_25: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_10, mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_28: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_25, [128, 1024]);  mul_25 = None
    mm_4: "f32[128, 1024]" = torch.ops.aten.mm.default(view_28, permute_19);  permute_19 = None
    permute_20: "f32[1024, 128]" = torch.ops.aten.permute.default(view_28, [1, 0])
    mm_5: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
    permute_21: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_8: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
    view_29: "f32[1024]" = torch.ops.aten.reshape.default(sum_8, [1024]);  sum_8 = None
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    view_30: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_31: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_30, [1, 128, 16, 64]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_23: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_32: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_23, [16, 128, 64]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_2: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_24, view_32);  permute_24 = None
    bmm_3: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_32, permute_25);  view_32 = permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    convert_element_type_2: "f32[16, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_26: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_27: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_3, mul_26);  bmm_3 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_28: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_27, alias_1);  mul_27 = None
    sum_9: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_28, [-1], True)
    mul_29: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_1, sum_9);  alias_1 = sum_9 = None
    sub_6: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_33: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_6, [1, 16, 128, 128]);  sub_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    div_2: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Scalar(view_33, 2)
    where: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(eq, div_2, view_33);  eq = div_2 = view_33 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(lt, full_default_1, where);  lt = full_default_1 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    view_35: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_1, [16, 128, 128]);  where_1 = None
    bmm_4: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_26, view_35);  permute_26 = None
    bmm_5: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_35, permute_27);  view_35 = permute_27 = None
    permute_28: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_36: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_2, [1, 16, 128, 64]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    add_11: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_36);  tangents_3 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_37: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_28, [1, 16, 128, 64]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    add_12: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_2, view_37);  tangents_2 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_38: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_29: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_8: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_39: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_8, [1, 128, 1024]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_30: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_11, [0, 2, 1, 3]);  add_11 = None
    clone_9: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_40: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_9, [1, 128, 1024]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_41: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_40, [128, 1024]);  view_40 = None
    mm_6: "f32[128, 1024]" = torch.ops.aten.mm.default(view_41, permute_31);  permute_31 = None
    permute_32: "f32[1024, 128]" = torch.ops.aten.permute.default(view_41, [1, 0])
    mm_7: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_10: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
    view_42: "f32[1024]" = torch.ops.aten.reshape.default(sum_10, [1024]);  sum_10 = None
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    view_43: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 1024]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_35: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_12, [0, 2, 1, 3]);  add_12 = None
    clone_10: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    view_44: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_10, [1, 128, 1024]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_45: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_44, [128, 1024]);  view_44 = None
    mm_8: "f32[128, 1024]" = torch.ops.aten.mm.default(view_45, permute_36);  permute_36 = None
    permute_37: "f32[1024, 128]" = torch.ops.aten.permute.default(view_45, [1, 0])
    mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_37, view);  permute_37 = None
    permute_38: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_45, [0], True);  view_45 = None
    view_46: "f32[1024]" = torch.ops.aten.reshape.default(sum_11, [1024]);  sum_11 = None
    permute_39: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_47: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_13: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_43, view_47);  view_43 = view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_30: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_39, 0.125);  view_39 = None
    view_48: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_30, [128, 1024]);  mul_30 = None
    mm_10: "f32[128, 1024]" = torch.ops.aten.mm.default(view_48, permute_40);  permute_40 = None
    permute_41: "f32[1024, 128]" = torch.ops.aten.permute.default(view_48, [1, 0])
    mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
    permute_42: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_48, [0], True);  view_48 = None
    view_49: "f32[1024]" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
    permute_43: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_50: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 1024]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_14: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_13, view_50);  add_13 = view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_32: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_14, primals_1);  primals_1 = None
    mul_33: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_32, 1024)
    sum_13: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_32, [2], True)
    mul_34: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_32, mul);  mul_32 = None
    sum_14: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_34, [2], True);  mul_34 = None
    mul_35: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, sum_14);  sum_14 = None
    sub_8: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_33, sum_13);  mul_33 = sum_13 = None
    sub_9: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_8, mul_35);  sub_8 = mul_35 = None
    div_3: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
    mul_36: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_3, sub_9);  div_3 = sub_9 = None
    mul_37: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_14, mul);  mul = None
    sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_37, [0, 1]);  mul_37 = None
    sum_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_15: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_10, mul_36);  add_10 = mul_36 = None
    return [sum_15, sum_16, permute_43, view_49, permute_39, view_46, permute_34, view_42, permute_22, view_29, sum_6, sum_7, permute_18, view_26, permute_14, view_23, add_15, None]
    