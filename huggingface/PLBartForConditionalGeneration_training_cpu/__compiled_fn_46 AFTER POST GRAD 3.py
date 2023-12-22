from __future__ import annotations



def forward(self, primals_9: "f32[768]", primals_15: "f32[768]", view: "f32[1024, 768]", getitem_11: "b8[1, 12, 1024, 1024]", permute_default_1: "f32[12, 1024, 1024]", permute_default_2: "f32[12, 64, 1024]", alias_default_1: "f32[1, 12, 1024, 1024]", permute_default_3: "f32[12, 64, 1024]", permute_default_4: "f32[12, 1024, 64]", view_14: "f32[1024, 768]", getitem_3: "b8[1, 1024, 768]", mul_1: "f32[1, 1024, 768]", view_16: "f32[1024, 768]", addmm_4: "f32[1024, 3072]", view_18: "f32[1024, 3072]", getitem_7: "b8[1, 1024, 768]", mul_6: "f32[1, 1024, 768]", div_1: "f32[1, 1024, 1]", permute_11: "f32[768, 3072]", permute_15: "f32[3072, 768]", div_2: "f32[1, 1024, 1]", permute_19: "f32[768, 768]", permute_31: "f32[768, 768]", permute_36: "f32[768, 768]", permute_40: "f32[768, 768]", tangents_1: "f32[1, 1024, 768]"):
    # No stacktrace found for following nodes
    convert_element_type_default: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_tensor: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_17: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 1024, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_4: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.7071067811865476)
    erf: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_3: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:344, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, primals_15);  primals_15 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, 768)
    sum_2: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True)
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, mul_6);  mul_9 = None
    sum_3: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_11, [2], True);  mul_11 = None
    mul_12: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, sum_3);  sum_3 = None
    sub_4: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_10, sum_2);  mul_10 = sum_2 = None
    sub_5: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_4, mul_12);  sub_4 = mul_12 = None
    mul_13: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
    mul_14: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, mul_6);  mul_6 = None
    sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_14, [0, 1]);  mul_14 = None
    sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:342, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_15: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_16: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_13, mul_15);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_20: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_16, [1024, 768]);  mul_16 = None
    mm: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_20, permute_11);  permute_11 = None
    permute_12: "f32[768, 1024]" = torch.ops.aten.permute.default(view_20, [1, 0])
    mm_1: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_12, view_18);  permute_12 = view_18 = None
    permute_13: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_20, [0], True);  view_20 = None
    view_21: "f32[768]" = torch.ops.aten.reshape.default(sum_6, [768]);  sum_6 = None
    permute_14: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    view_22: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(mm, [1, 1024, 3072]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_18: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
    mul_19: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, view_17)
    mul_20: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_19, -0.5);  mul_19 = None
    exp_1: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_20);  mul_20 = None
    mul_21: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_22: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, mul_21);  view_17 = mul_21 = None
    add_8: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_18, mul_22);  mul_18 = mul_22 = None
    mul_23: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_22, add_8);  view_22 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_23: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_23, [1024, 3072]);  mul_23 = None
    mm_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_23, permute_15);  permute_15 = None
    permute_16: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_23, [1, 0])
    mm_3: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_16, view_16);  permute_16 = view_16 = None
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_23, [0], True);  view_23 = None
    view_24: "f32[3072]" = torch.ops.aten.reshape.default(sum_7, [3072]);  sum_7 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    view_25: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_2, [1, 1024, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    add_9: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_13, view_25);  mul_13 = view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:336, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_25: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_9, primals_9);  primals_9 = None
    mul_26: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, 768)
    sum_8: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [2], True)
    mul_27: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, mul_1);  mul_25 = None
    sum_9: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True);  mul_27 = None
    mul_28: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_9);  sum_9 = None
    sub_7: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_26, sum_8);  mul_26 = sum_8 = None
    sub_8: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_7, mul_28);  sub_7 = mul_28 = None
    mul_29: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
    mul_30: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_9, mul_1);  mul_1 = None
    sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_30, [0, 1]);  mul_30 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:334, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    convert_element_type_1: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_31: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_32: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_29, mul_31);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_26: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_32, [1024, 768]);  mul_32 = None
    mm_4: "f32[1024, 768]" = torch.ops.aten.mm.default(view_26, permute_19);  permute_19 = None
    permute_20: "f32[768, 1024]" = torch.ops.aten.permute.default(view_26, [1, 0])
    mm_5: "f32[768, 768]" = torch.ops.aten.mm.default(permute_20, view_14);  permute_20 = view_14 = None
    permute_21: "f32[768, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_26, [0], True);  view_26 = None
    view_27: "f32[768]" = torch.ops.aten.reshape.default(sum_12, [768]);  sum_12 = None
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    view_28: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_4, [1, 1024, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_28, [1, 1024, 12, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_23: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_30: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(permute_23, [12, 1024, 64]);  permute_23 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
    view_default_6: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(unsqueeze_default_3, [12, 1024, 64]);  unsqueeze_default_3 = None
    bmm_default_2: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_default_1, view_default_6);  permute_default_1 = None
    view_default_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_default_2, [1, 12, 1024, 64]);  bmm_default_2 = None
    squeeze_dim_1: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(view_default_7, 0);  view_default_7 = None
    bmm_default_3: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_default_6, permute_default_2);  view_default_6 = permute_default_2 = None
    view_default_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_default_3, [1, 12, 1024, 1024]);  bmm_default_3 = None
    mul_tensor_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_default_8, mul_tensor);  view_default_8 = mul_tensor = None
    mul_tensor_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, alias_default_1);  mul_tensor_1 = None
    sum_dim_int_list_1: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
    mul_tensor_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_default_1, sum_dim_int_list_1);  alias_default_1 = sum_dim_int_list_1 = None
    sub_tensor_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
    view_default_9: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(sub_tensor_1, [12, 1024, 1024]);  sub_tensor_1 = None
    bmm_default_4: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_default_3, view_default_9);  permute_default_3 = None
    view_default_10: "f32[1, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_default_4, [1, 12, 64, 1024]);  bmm_default_4 = None
    mul_scalar_2: "f32[1, 12, 64, 1024]" = torch.ops.aten.mul.Scalar(view_default_10, 1.0);  view_default_10 = None
    permute_default_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
    squeeze_dim_2: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(permute_default_5, 0);  permute_default_5 = None
    bmm_default_5: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_default_9, permute_default_4);  view_default_9 = permute_default_4 = None
    view_default_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_default_5, [1, 12, 1024, 64]);  bmm_default_5 = None
    mul_scalar_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.mul.Scalar(view_default_11, 1.0);  view_default_11 = None
    squeeze_dim_3: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(mul_scalar_3, 0);  mul_scalar_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_31: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_1, [1, 12, 1024, 64]);  squeeze_dim_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_32: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_2, [1, 12, 1024, 64]);  squeeze_dim_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_33: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_3, [1, 12, 1024, 64]);  squeeze_dim_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    clone_8: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_34: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_8, [1, 1024, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_30: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    clone_9: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_35: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_9, [1, 1024, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_36: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_35, [1024, 768]);  view_35 = None
    mm_6: "f32[1024, 768]" = torch.ops.aten.mm.default(view_36, permute_31);  permute_31 = None
    permute_32: "f32[768, 1024]" = torch.ops.aten.permute.default(view_36, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_36, [0], True);  view_36 = None
    view_37: "f32[768]" = torch.ops.aten.reshape.default(sum_14, [768]);  sum_14 = None
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    view_38: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_6, [1, 1024, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    add_10: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_29, view_38);  mul_29 = view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_35: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    view_39: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(permute_35, [1, 1024, 768]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_40: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_39, [1024, 768]);  view_39 = None
    mm_8: "f32[1024, 768]" = torch.ops.aten.mm.default(view_40, permute_36);  permute_36 = None
    permute_37: "f32[768, 1024]" = torch.ops.aten.permute.default(view_40, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_37, view);  permute_37 = None
    permute_38: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
    view_41: "f32[768]" = torch.ops.aten.reshape.default(sum_15, [768]);  sum_15 = None
    permute_39: "f32[768, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_42: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_8, [1, 1024, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_11: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_10, view_42);  add_10 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_37: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_34, 0.125);  view_34 = None
    view_43: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_37, [1024, 768]);  mul_37 = None
    mm_10: "f32[1024, 768]" = torch.ops.aten.mm.default(view_43, permute_40);  permute_40 = None
    permute_41: "f32[768, 1024]" = torch.ops.aten.permute.default(view_43, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_43, [0], True);  view_43 = None
    view_44: "f32[768]" = torch.ops.aten.reshape.default(sum_16, [768]);  sum_16 = None
    permute_43: "f32[768, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_45: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_10, [1, 1024, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_12: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_11, view_45);  add_11 = view_45 = None
    return [permute_43, view_44, permute_39, view_41, permute_34, view_37, permute_22, view_27, sum_10, sum_11, permute_18, view_24, permute_14, view_21, sum_4, sum_5, add_12]
    