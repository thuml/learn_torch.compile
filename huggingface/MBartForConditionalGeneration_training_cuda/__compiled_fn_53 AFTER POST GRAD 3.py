from __future__ import annotations



def forward(self, primals_1: "f32[1024]", primals_11: "f32[1024]", primals_17: "f32[1, 1024, 1024]", getitem_1: "f32[1, 1024, 1]", rsqrt: "f32[1, 1024, 1]", view: "f32[1024, 1024]", bmm: "f32[16, 1024, 1024]", amax: "f32[16, 1024, 1]", sum_1: "f32[16, 1024, 1]", view_14: "f32[1024, 1024]", getitem_3: "b8[1, 1024, 1024]", mul_3: "f32[1, 1024, 1024]", view_16: "f32[1024, 1024]", addmm_4: "f32[1024, 4096]", view_18: "f32[1024, 4096]", getitem_7: "b8[1, 1024, 1024]", permute_11: "f32[1024, 4096]", permute_15: "f32[4096, 1024]", div_1: "f32[1, 1024, 1]", permute_19: "f32[1024, 1024]", permute_24: "f32[16, 1024, 1024]", permute_25: "f32[16, 64, 1024]", permute_26: "f32[16, 64, 1024]", permute_27: "f32[16, 1024, 64]", permute_31: "f32[1024, 1024]", permute_36: "f32[1024, 1024]", permute_40: "f32[1024, 1024]", tangents_1: "f32[1, 1024, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    sub: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(primals_17, getitem_1);  primals_17 = getitem_1 = None
    mul: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_1: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(bmm, amax);  bmm = amax = None
    exp: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    div: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_17: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_4, [1, 1024, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_17, 0.7071067811865476)
    erf: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_5: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:343, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[1, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_8: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_9: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:342, code: hidden_states = self.fc2(hidden_states)
    view_20: "f32[1024, 1024]" = torch.ops.aten.reshape.default(mul_9, [1024, 1024]);  mul_9 = None
    mm: "f32[1024, 4096]" = torch.ops.aten.mm.default(view_20, permute_11);  permute_11 = None
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_20, [1, 0])
    mm_1: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_12, view_18);  permute_12 = view_18 = None
    permute_13: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_20, [0], True);  view_20 = None
    view_21: "f32[1024]" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
    permute_14: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    view_22: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(mm, [1, 1024, 4096]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_11: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
    mul_12: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_17, view_17)
    mul_13: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_12, -0.5);  mul_12 = None
    exp_1: "f32[1, 1024, 4096]" = torch.ops.aten.exp.default(mul_13);  mul_13 = None
    mul_14: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_15: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_17, mul_14);  view_17 = mul_14 = None
    add_8: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(mul_11, mul_15);  mul_11 = mul_15 = None
    mul_16: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_22, add_8);  view_22 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:340, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_23: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_16, [1024, 4096]);  mul_16 = None
    mm_2: "f32[1024, 1024]" = torch.ops.aten.mm.default(view_23, permute_15);  permute_15 = None
    permute_16: "f32[4096, 1024]" = torch.ops.aten.permute.default(view_23, [1, 0])
    mm_3: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_16, view_16);  permute_16 = view_16 = None
    permute_17: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_23, [0], True);  view_23 = None
    view_24: "f32[4096]" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
    permute_18: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    view_25: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(mm_2, [1, 1024, 1024]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_18: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_25, primals_11);  primals_11 = None
    mul_19: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_18, 1024)
    sum_4: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
    mul_20: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_18, mul_3);  mul_18 = None
    sum_5: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
    mul_21: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_3, sum_5);  sum_5 = None
    sub_4: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_19, sum_4);  mul_19 = sum_4 = None
    sub_5: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(sub_4, mul_21);  sub_4 = mul_21 = None
    mul_22: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
    mul_23: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_25, mul_3);  mul_3 = None
    sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
    sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_25, [0, 1]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:339, code: hidden_states = self.final_layer_norm(hidden_states)
    add_9: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(tangents_1, mul_22);  tangents_1 = mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:335, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_24: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_25: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_9, mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_26: "f32[1024, 1024]" = torch.ops.aten.reshape.default(mul_25, [1024, 1024]);  mul_25 = None
    mm_4: "f32[1024, 1024]" = torch.ops.aten.mm.default(view_26, permute_19);  permute_19 = None
    permute_20: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_26, [1, 0])
    mm_5: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_20, view_14);  permute_20 = view_14 = None
    permute_21: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_8: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_26, [0], True);  view_26 = None
    view_27: "f32[1024]" = torch.ops.aten.reshape.default(sum_8, [1024]);  sum_8 = None
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    view_28: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(mm_4, [1, 1024, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_29: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_28, [1, 1024, 16, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_23: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_30: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(permute_23, [16, 1024, 64]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_2: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(permute_24, view_30);  permute_24 = None
    bmm_3: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_30, permute_25);  view_30 = permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_26: "f32[16, 1024, 1024]" = torch.ops.aten.mul.Tensor(bmm_3, div);  bmm_3 = None
    sum_9: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_26, [-1], True)
    mul_27: "f32[16, 1024, 1024]" = torch.ops.aten.mul.Tensor(div, sum_9);  div = sum_9 = None
    sub_6: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_4: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_26, sub_6);  permute_26 = None
    bmm_5: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(sub_6, permute_27);  sub_6 = permute_27 = None
    permute_28: "f32[16, 1024, 64]" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_31: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_2, [1, 16, 1024, 64]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_32: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(permute_28, [1, 16, 1024, 64]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_33: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_29: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    clone_8: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_34: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_8, [1, 1024, 1024]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_30: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    clone_9: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_35: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_9, [1, 1024, 1024]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_36: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_35, [1024, 1024]);  view_35 = None
    mm_6: "f32[1024, 1024]" = torch.ops.aten.mm.default(view_36, permute_31);  permute_31 = None
    permute_32: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_36, [1, 0])
    mm_7: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_10: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_36, [0], True);  view_36 = None
    view_37: "f32[1024]" = torch.ops.aten.reshape.default(sum_10, [1024]);  sum_10 = None
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    view_38: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(mm_6, [1, 1024, 1024]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_35: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    view_39: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_35, [1, 1024, 1024]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_40: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_39, [1024, 1024]);  view_39 = None
    mm_8: "f32[1024, 1024]" = torch.ops.aten.mm.default(view_40, permute_36);  permute_36 = None
    permute_37: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_40, [1, 0])
    mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_37, view);  permute_37 = None
    permute_38: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_11: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
    view_41: "f32[1024]" = torch.ops.aten.reshape.default(sum_11, [1024]);  sum_11 = None
    permute_39: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_42: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(mm_8, [1, 1024, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_10: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(view_38, view_42);  view_38 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_28: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_34, 0.125);  view_34 = None
    view_43: "f32[1024, 1024]" = torch.ops.aten.reshape.default(mul_28, [1024, 1024]);  mul_28 = None
    mm_10: "f32[1024, 1024]" = torch.ops.aten.mm.default(view_43, permute_40);  permute_40 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(view_43, [1, 0])
    mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
    permute_42: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_43, [0], True);  view_43 = None
    view_44: "f32[1024]" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
    permute_43: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_45: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(mm_10, [1, 1024, 1024]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_11: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_10, view_45);  add_10 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_30: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_11, primals_1);  primals_1 = None
    mul_31: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_30, 1024)
    sum_13: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_30, [2], True)
    mul_32: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_30, mul);  mul_30 = None
    sum_14: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_32, [2], True);  mul_32 = None
    mul_33: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, sum_14);  sum_14 = None
    sub_8: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_31, sum_13);  mul_31 = sum_13 = None
    sub_9: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(sub_8, mul_33);  sub_8 = mul_33 = None
    div_2: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
    mul_34: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(div_2, sub_9);  div_2 = sub_9 = None
    mul_35: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(add_11, mul);  mul = None
    sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_35, [0, 1]);  mul_35 = None
    sum_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:328, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_12: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_9, mul_34);  add_9 = mul_34 = None
    return [sum_15, sum_16, permute_43, view_44, permute_39, view_41, permute_34, view_37, permute_22, view_27, sum_6, sum_7, permute_18, view_24, permute_14, view_21, add_12]
    