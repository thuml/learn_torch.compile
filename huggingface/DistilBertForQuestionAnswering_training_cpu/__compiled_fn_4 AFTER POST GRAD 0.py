from __future__ import annotations



def forward(self, primals_3: "f32[768]", primals_13: "f32[768]", primals_19: "f32[768]", primals_29: "f32[768]", primals_35: "f32[768]", primals_45: "f32[768]", primals_51: "f32[768]", primals_61: "f32[768]", primals_67: "f32[768]", primals_77: "f32[768]", primals_83: "f32[768]", primals_93: "f32[768]", primals_99: "f32[768]", primals_104: "i64[1, 128]", slice_2: "i64[1, 128]", mul: "f32[1, 128, 768]", getitem_3: "b8[1, 128, 768]", view: "f32[128, 768]", view_12: "b8[1, 1, 1, 128]", getitem_5: "b8[1, 12, 128, 128]", view_17: "f32[128, 768]", mul_2: "f32[1, 128, 768]", view_19: "f32[128, 768]", addmm_4: "f32[128, 3072]", view_21: "f32[128, 3072]", getitem_9: "b8[1, 128, 768]", mul_7: "f32[1, 128, 768]", view_23: "f32[128, 768]", getitem_13: "b8[1, 12, 128, 128]", view_40: "f32[128, 768]", mul_9: "f32[1, 128, 768]", view_42: "f32[128, 768]", addmm_10: "f32[128, 3072]", view_44: "f32[128, 3072]", getitem_17: "b8[1, 128, 768]", mul_14: "f32[1, 128, 768]", view_46: "f32[128, 768]", getitem_21: "b8[1, 12, 128, 128]", view_63: "f32[128, 768]", mul_16: "f32[1, 128, 768]", view_65: "f32[128, 768]", addmm_16: "f32[128, 3072]", view_67: "f32[128, 3072]", getitem_25: "b8[1, 128, 768]", mul_21: "f32[1, 128, 768]", view_69: "f32[128, 768]", getitem_29: "b8[1, 12, 128, 128]", view_86: "f32[128, 768]", mul_23: "f32[1, 128, 768]", view_88: "f32[128, 768]", addmm_22: "f32[128, 3072]", view_90: "f32[128, 3072]", getitem_33: "b8[1, 128, 768]", mul_28: "f32[1, 128, 768]", view_92: "f32[128, 768]", getitem_37: "b8[1, 12, 128, 128]", view_109: "f32[128, 768]", mul_30: "f32[1, 128, 768]", view_111: "f32[128, 768]", addmm_28: "f32[128, 3072]", view_113: "f32[128, 3072]", getitem_41: "b8[1, 128, 768]", mul_35: "f32[1, 128, 768]", view_115: "f32[128, 768]", getitem_45: "b8[1, 12, 128, 128]", view_132: "f32[128, 768]", mul_37: "f32[1, 128, 768]", view_134: "f32[128, 768]", addmm_34: "f32[128, 3072]", view_136: "f32[128, 3072]", getitem_49: "b8[1, 128, 768]", mul_42: "f32[1, 128, 768]", getitem_53: "b8[1, 128, 768]", view_138: "f32[128, 768]", sub_20: "f32[1, 128]", ne: "b8[1]", sub_22: "f32[1, 128]", ne_3: "b8[1]", ne_6: "b8[1, 1]", where_10: "i64[1, 1]", ne_8: "b8[1, 1]", where_12: "i64[1, 1]", permute_67: "f32[2, 768]", div_18: "f32[1, 128, 1]", permute_71: "f32[768, 3072]", permute_75: "f32[3072, 768]", div_19: "f32[1, 128, 1]", permute_79: "f32[768, 768]", permute_84: "f32[12, 128, 128]", permute_85: "f32[12, 64, 128]", alias_10: "f32[1, 12, 128, 128]", permute_86: "f32[12, 64, 128]", permute_87: "f32[12, 128, 64]", permute_90: "f32[768, 768]", permute_95: "f32[768, 768]", permute_100: "f32[768, 768]", div_21: "f32[1, 128, 1]", permute_104: "f32[768, 3072]", permute_108: "f32[3072, 768]", div_22: "f32[1, 128, 1]", permute_112: "f32[768, 768]", permute_117: "f32[12, 128, 128]", permute_118: "f32[12, 64, 128]", alias_11: "f32[1, 12, 128, 128]", permute_119: "f32[12, 64, 128]", permute_120: "f32[12, 128, 64]", permute_123: "f32[768, 768]", permute_128: "f32[768, 768]", permute_133: "f32[768, 768]", div_24: "f32[1, 128, 1]", permute_137: "f32[768, 3072]", permute_141: "f32[3072, 768]", div_25: "f32[1, 128, 1]", permute_145: "f32[768, 768]", permute_150: "f32[12, 128, 128]", permute_151: "f32[12, 64, 128]", alias_12: "f32[1, 12, 128, 128]", permute_152: "f32[12, 64, 128]", permute_153: "f32[12, 128, 64]", permute_156: "f32[768, 768]", permute_161: "f32[768, 768]", permute_166: "f32[768, 768]", div_27: "f32[1, 128, 1]", permute_170: "f32[768, 3072]", permute_174: "f32[3072, 768]", div_28: "f32[1, 128, 1]", permute_178: "f32[768, 768]", permute_183: "f32[12, 128, 128]", permute_184: "f32[12, 64, 128]", alias_13: "f32[1, 12, 128, 128]", permute_185: "f32[12, 64, 128]", permute_186: "f32[12, 128, 64]", permute_189: "f32[768, 768]", permute_194: "f32[768, 768]", permute_199: "f32[768, 768]", div_30: "f32[1, 128, 1]", permute_203: "f32[768, 3072]", permute_207: "f32[3072, 768]", div_31: "f32[1, 128, 1]", permute_211: "f32[768, 768]", permute_216: "f32[12, 128, 128]", permute_217: "f32[12, 64, 128]", alias_14: "f32[1, 12, 128, 128]", permute_218: "f32[12, 64, 128]", permute_219: "f32[12, 128, 64]", permute_222: "f32[768, 768]", permute_227: "f32[768, 768]", permute_232: "f32[768, 768]", div_33: "f32[1, 128, 1]", permute_236: "f32[768, 3072]", permute_240: "f32[3072, 768]", div_34: "f32[1, 128, 1]", permute_244: "f32[768, 768]", permute_249: "f32[12, 128, 128]", permute_250: "f32[12, 64, 128]", alias_15: "f32[1, 12, 128, 128]", permute_251: "f32[12, 64, 128]", permute_252: "f32[12, 128, 64]", permute_255: "f32[768, 768]", permute_260: "f32[768, 768]", permute_265: "f32[768, 768]", div_36: "f32[1, 128, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 128]", tangents_3: "f32[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:221, code: mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    expand_2: "b8[1, 12, 128, 128]" = torch.ops.aten.expand.default(view_12, [1, 12, 128, 128]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_20: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476)
    erf: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_43: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_1: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_66: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476)
    erf_2: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_89: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.7071067811865476)
    erf_3: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_27: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_112: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.7071067811865476)
    erf_4: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_34: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_135: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476)
    erf_5: "f32[1, 128, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_41: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:941, code: start_loss = loss_fct(start_logits, start_positions)
    full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sum_8: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:942, code: end_loss = loss_fct(end_logits, end_positions)
    sum_11: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_11, torch.float32);  sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:943, code: total_loss = (start_loss + end_loss) / 2
    div_15: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:942, code: end_loss = loss_fct(end_logits, end_positions)
    div_16: "f32[]" = torch.ops.aten.div.Tensor(div_15, convert_element_type_1);  convert_element_type_1 = None
    full_default_11: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[1, 128]" = torch.ops.aten.scatter.value(full_default_11, 1, where_10, -1.0);  where_10 = None
    where_11: "f32[1, 1]" = torch.ops.aten.where.self(ne_6, div_16, full_default_7);  ne_6 = div_16 = None
    mul_44: "f32[1, 128]" = torch.ops.aten.mul.Tensor(scatter, where_11);  scatter = where_11 = None
    exp_8: "f32[1, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_13: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_44, [1], True)
    mul_45: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_8, sum_13);  exp_8 = sum_13 = None
    sub_23: "f32[1, 128]" = torch.ops.aten.sub.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:942, code: end_loss = loss_fct(end_logits, end_positions)
    add_46: "f32[1, 128]" = torch.ops.aten.add.Tensor(tangents_3, sub_23);  tangents_3 = sub_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:941, code: start_loss = loss_fct(start_logits, start_positions)
    div_17: "f32[]" = torch.ops.aten.div.Tensor(div_15, convert_element_type);  div_15 = convert_element_type = None
    scatter_1: "f32[1, 128]" = torch.ops.aten.scatter.value(full_default_11, 1, where_12, -1.0);  full_default_11 = where_12 = None
    where_13: "f32[1, 1]" = torch.ops.aten.where.self(ne_8, div_17, full_default_7);  ne_8 = div_17 = None
    mul_46: "f32[1, 128]" = torch.ops.aten.mul.Tensor(scatter_1, where_13);  scatter_1 = where_13 = None
    exp_9: "f32[1, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_14: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_46, [1], True)
    mul_47: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_9, sum_14);  exp_9 = sum_14 = None
    sub_24: "f32[1, 128]" = torch.ops.aten.sub.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:941, code: start_loss = loss_fct(start_logits, start_positions)
    add_47: "f32[1, 128]" = torch.ops.aten.add.Tensor(tangents_2, sub_24);  tangents_2 = sub_24 = None
    
    # No stacktrace found for following nodes
    unsqueeze_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_46, 2);  add_46 = None
    unsqueeze_5: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_47, 2);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:924, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat: "f32[1, 128, 2]" = torch.ops.aten.cat.default([unsqueeze_5, unsqueeze_4], 2);  unsqueeze_5 = unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:923, code: logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
    view_140: "f32[128, 2]" = torch.ops.aten.reshape.default(cat, [128, 2]);  cat = None
    mm: "f32[128, 768]" = torch.ops.aten.mm.default(view_140, permute_67);  permute_67 = None
    permute_68: "f32[2, 128]" = torch.ops.aten.permute.default(view_140, [1, 0])
    mm_1: "f32[2, 768]" = torch.ops.aten.mm.default(permute_68, view_138);  permute_68 = view_138 = None
    permute_69: "f32[768, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_15: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True);  view_140 = None
    view_141: "f32[2]" = torch.ops.aten.reshape.default(sum_15, [2]);  sum_15 = None
    permute_70: "f32[2, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    view_142: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm, [1, 128, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:922, code: hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
    convert_element_type_2: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
    mul_48: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_49: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(view_142, mul_48);  view_142 = mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    mul_51: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_99);  primals_99 = None
    mul_52: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_51, 768)
    sum_16: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_51, [2], True)
    mul_53: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_51, mul_42);  mul_51 = None
    sum_17: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_53, [2], True);  mul_53 = None
    mul_54: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_17);  sum_17 = None
    sub_26: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_52, sum_16);  mul_52 = sum_16 = None
    sub_27: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_54);  sub_26 = mul_54 = None
    mul_55: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_27);  div_18 = sub_27 = None
    mul_56: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_49, mul_42);  mul_42 = None
    sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_56, [0, 1]);  mul_56 = None
    sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_49, [0, 1]);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_3: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_57: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_58: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_55, mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_143: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_58, [128, 768]);  mul_58 = None
    mm_2: "f32[128, 3072]" = torch.ops.aten.mm.default(view_143, permute_71);  permute_71 = None
    permute_72: "f32[768, 128]" = torch.ops.aten.permute.default(view_143, [1, 0])
    mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_72, view_136);  permute_72 = view_136 = None
    permute_73: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_143, [0], True);  view_143 = None
    view_144: "f32[768]" = torch.ops.aten.reshape.default(sum_20, [768]);  sum_20 = None
    permute_74: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    view_145: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 3072]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_60: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
    mul_61: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, view_135)
    mul_62: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_61, -0.5);  mul_61 = None
    exp_10: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_62);  mul_62 = None
    mul_63: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_64: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_135, mul_63);  view_135 = mul_63 = None
    add_49: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_60, mul_64);  mul_60 = mul_64 = None
    mul_65: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_145, add_49);  view_145 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_146: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_65, [128, 3072]);  mul_65 = None
    mm_4: "f32[128, 768]" = torch.ops.aten.mm.default(view_146, permute_75);  permute_75 = None
    permute_76: "f32[3072, 128]" = torch.ops.aten.permute.default(view_146, [1, 0])
    mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_76, view_134);  permute_76 = view_134 = None
    permute_77: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_21: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_146, [0], True);  view_146 = None
    view_147: "f32[3072]" = torch.ops.aten.reshape.default(sum_21, [3072]);  sum_21 = None
    permute_78: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    view_148: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_50: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_55, view_148);  mul_55 = view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    mul_67: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_50, primals_93);  primals_93 = None
    mul_68: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_67, 768)
    sum_22: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_67, [2], True)
    mul_69: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_67, mul_37);  mul_67 = None
    sum_23: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_69, [2], True);  mul_69 = None
    mul_70: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_37, sum_23);  sum_23 = None
    sub_29: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_68, sum_22);  mul_68 = sum_22 = None
    sub_30: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_29, mul_70);  sub_29 = mul_70 = None
    mul_71: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_30);  div_19 = sub_30 = None
    mul_72: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_50, mul_37);  mul_37 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_72, [0, 1]);  mul_72 = None
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_50, [0, 1]);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_149: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_71, [128, 768])
    mm_6: "f32[128, 768]" = torch.ops.aten.mm.default(view_149, permute_79);  permute_79 = None
    permute_80: "f32[768, 128]" = torch.ops.aten.permute.default(view_149, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_80, view_132);  permute_80 = view_132 = None
    permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True);  view_149 = None
    view_150: "f32[768]" = torch.ops.aten.reshape.default(sum_26, [768]);  sum_26 = None
    permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    view_151: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_152: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_151, [1, 128, 12, 64]);  view_151 = None
    permute_83: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_153: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(permute_83, [12, 128, 64]);  permute_83 = None
    bmm_12: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_84, view_153);  permute_84 = None
    bmm_13: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_153, permute_85);  view_153 = permute_85 = None
    view_154: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_12, [1, 12, 128, 64]);  bmm_12 = None
    view_155: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_13, [1, 12, 128, 128]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_4: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_73: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_74: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_155, mul_73);  view_155 = mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    mul_75: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_74, alias_10);  mul_74 = None
    sum_27: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_75, [-1], True)
    mul_76: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_10, sum_27);  alias_10 = sum_27 = None
    sub_31: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_14: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_7, sub_31);  sub_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_156: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(where_14, [12, 128, 128]);  where_14 = None
    bmm_14: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_86, view_156);  permute_86 = None
    bmm_15: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_156, permute_87);  view_156 = permute_87 = None
    view_157: "f32[1, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_14, [1, 12, 64, 128]);  bmm_14 = None
    view_158: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 12, 128, 64]);  bmm_15 = None
    permute_88: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_157, [0, 1, 3, 2]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_20: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_158, 8.0);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_89: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
    clone_11: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
    view_159: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_11, [1, 128, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_160: "f32[128, 768]" = torch.ops.aten.reshape.default(view_159, [128, 768]);  view_159 = None
    mm_8: "f32[128, 768]" = torch.ops.aten.mm.default(view_160, permute_90);  permute_90 = None
    permute_91: "f32[768, 128]" = torch.ops.aten.permute.default(view_160, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_91, view_115);  permute_91 = None
    permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True);  view_160 = None
    view_161: "f32[768]" = torch.ops.aten.reshape.default(sum_28, [768]);  sum_28 = None
    permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    view_162: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_51: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_71, view_162);  mul_71 = view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_94: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_88, [0, 2, 1, 3]);  permute_88 = None
    view_163: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(permute_94, [1, 128, 768]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_164: "f32[128, 768]" = torch.ops.aten.reshape.default(view_163, [128, 768]);  view_163 = None
    mm_10: "f32[128, 768]" = torch.ops.aten.mm.default(view_164, permute_95);  permute_95 = None
    permute_96: "f32[768, 128]" = torch.ops.aten.permute.default(view_164, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_96, view_115);  permute_96 = None
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_164, [0], True);  view_164 = None
    view_165: "f32[768]" = torch.ops.aten.reshape.default(sum_29, [768]);  sum_29 = None
    permute_98: "f32[768, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    view_166: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_52: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_51, view_166);  add_51 = view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_99: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_20, [0, 2, 1, 3]);  div_20 = None
    clone_12: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
    view_167: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_12, [1, 128, 768]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_168: "f32[128, 768]" = torch.ops.aten.reshape.default(view_167, [128, 768]);  view_167 = None
    mm_12: "f32[128, 768]" = torch.ops.aten.mm.default(view_168, permute_100);  permute_100 = None
    permute_101: "f32[768, 128]" = torch.ops.aten.permute.default(view_168, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_101, view_115);  permute_101 = view_115 = None
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
    view_169: "f32[768]" = torch.ops.aten.reshape.default(sum_30, [768]);  sum_30 = None
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    view_170: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_53: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_52, view_170);  add_52 = view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    mul_78: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_53, primals_83);  primals_83 = None
    mul_79: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_78, 768)
    sum_31: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_78, [2], True)
    mul_80: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_78, mul_35);  mul_78 = None
    sum_32: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_80, [2], True);  mul_80 = None
    mul_81: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_35, sum_32);  sum_32 = None
    sub_33: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_79, sum_31);  mul_79 = sum_31 = None
    sub_34: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_33, mul_81);  sub_33 = mul_81 = None
    mul_82: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_34);  div_21 = sub_34 = None
    mul_83: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_53, mul_35);  mul_35 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_83, [0, 1]);  mul_83 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_53, [0, 1]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_5: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_84: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_85: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_82, mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_171: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_85, [128, 768]);  mul_85 = None
    mm_14: "f32[128, 3072]" = torch.ops.aten.mm.default(view_171, permute_104);  permute_104 = None
    permute_105: "f32[768, 128]" = torch.ops.aten.permute.default(view_171, [1, 0])
    mm_15: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_105, view_113);  permute_105 = view_113 = None
    permute_106: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
    view_172: "f32[768]" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
    permute_107: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    view_173: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 3072]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
    mul_88: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, view_112)
    mul_89: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_88, -0.5);  mul_88 = None
    exp_11: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_89);  mul_89 = None
    mul_90: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_91: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_112, mul_90);  view_112 = mul_90 = None
    add_55: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_87, mul_91);  mul_87 = mul_91 = None
    mul_92: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, add_55);  view_173 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_174: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_92, [128, 3072]);  mul_92 = None
    mm_16: "f32[128, 768]" = torch.ops.aten.mm.default(view_174, permute_108);  permute_108 = None
    permute_109: "f32[3072, 128]" = torch.ops.aten.permute.default(view_174, [1, 0])
    mm_17: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_109, view_111);  permute_109 = view_111 = None
    permute_110: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_36: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_174, [0], True);  view_174 = None
    view_175: "f32[3072]" = torch.ops.aten.reshape.default(sum_36, [3072]);  sum_36 = None
    permute_111: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    view_176: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_56: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_82, view_176);  mul_82 = view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    mul_94: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_56, primals_77);  primals_77 = None
    mul_95: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_94, 768)
    sum_37: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_94, [2], True)
    mul_96: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_94, mul_30);  mul_94 = None
    sum_38: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True);  mul_96 = None
    mul_97: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_38);  sum_38 = None
    sub_36: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_95, sum_37);  mul_95 = sum_37 = None
    sub_37: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_36, mul_97);  sub_36 = mul_97 = None
    mul_98: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_37);  div_22 = sub_37 = None
    mul_99: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_56, mul_30);  mul_30 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_99, [0, 1]);  mul_99 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_56, [0, 1]);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_177: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_98, [128, 768])
    mm_18: "f32[128, 768]" = torch.ops.aten.mm.default(view_177, permute_112);  permute_112 = None
    permute_113: "f32[768, 128]" = torch.ops.aten.permute.default(view_177, [1, 0])
    mm_19: "f32[768, 768]" = torch.ops.aten.mm.default(permute_113, view_109);  permute_113 = view_109 = None
    permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_177, [0], True);  view_177 = None
    view_178: "f32[768]" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    view_179: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_180: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_179, [1, 128, 12, 64]);  view_179 = None
    permute_116: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_181: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(permute_116, [12, 128, 64]);  permute_116 = None
    bmm_16: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_117, view_181);  permute_117 = None
    bmm_17: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_181, permute_118);  view_181 = permute_118 = None
    view_182: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_16, [1, 12, 128, 64]);  bmm_16 = None
    view_183: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_17, [1, 12, 128, 128]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_6: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_100: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_101: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_183, mul_100);  view_183 = mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    mul_102: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_101, alias_11);  mul_101 = None
    sum_42: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_102, [-1], True)
    mul_103: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_11, sum_42);  alias_11 = sum_42 = None
    sub_38: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_15: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_7, sub_38);  sub_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_184: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(where_15, [12, 128, 128]);  where_15 = None
    bmm_18: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_119, view_184);  permute_119 = None
    bmm_19: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_184, permute_120);  view_184 = permute_120 = None
    view_185: "f32[1, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_18, [1, 12, 64, 128]);  bmm_18 = None
    view_186: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 12, 128, 64]);  bmm_19 = None
    permute_121: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_185, [0, 1, 3, 2]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_23: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_186, 8.0);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_122: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    clone_15: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_187: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_15, [1, 128, 768]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_188: "f32[128, 768]" = torch.ops.aten.reshape.default(view_187, [128, 768]);  view_187 = None
    mm_20: "f32[128, 768]" = torch.ops.aten.mm.default(view_188, permute_123);  permute_123 = None
    permute_124: "f32[768, 128]" = torch.ops.aten.permute.default(view_188, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_124, view_92);  permute_124 = None
    permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
    view_189: "f32[768]" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
    permute_126: "f32[768, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    view_190: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_20, [1, 128, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_57: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_98, view_190);  mul_98 = view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_127: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_121, [0, 2, 1, 3]);  permute_121 = None
    view_191: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(permute_127, [1, 128, 768]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_192: "f32[128, 768]" = torch.ops.aten.reshape.default(view_191, [128, 768]);  view_191 = None
    mm_22: "f32[128, 768]" = torch.ops.aten.mm.default(view_192, permute_128);  permute_128 = None
    permute_129: "f32[768, 128]" = torch.ops.aten.permute.default(view_192, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_129, view_92);  permute_129 = None
    permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
    view_193: "f32[768]" = torch.ops.aten.reshape.default(sum_44, [768]);  sum_44 = None
    permute_131: "f32[768, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    view_194: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_22, [1, 128, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_58: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_57, view_194);  add_57 = view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_132: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_23, [0, 2, 1, 3]);  div_23 = None
    clone_16: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_195: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_16, [1, 128, 768]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_196: "f32[128, 768]" = torch.ops.aten.reshape.default(view_195, [128, 768]);  view_195 = None
    mm_24: "f32[128, 768]" = torch.ops.aten.mm.default(view_196, permute_133);  permute_133 = None
    permute_134: "f32[768, 128]" = torch.ops.aten.permute.default(view_196, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_134, view_92);  permute_134 = view_92 = None
    permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_196, [0], True);  view_196 = None
    view_197: "f32[768]" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
    permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    view_198: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_24, [1, 128, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_59: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_58, view_198);  add_58 = view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    mul_105: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_59, primals_67);  primals_67 = None
    mul_106: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_105, 768)
    sum_46: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_105, [2], True)
    mul_107: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_105, mul_28);  mul_105 = None
    sum_47: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_107, [2], True);  mul_107 = None
    mul_108: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, sum_47);  sum_47 = None
    sub_40: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_106, sum_46);  mul_106 = sum_46 = None
    sub_41: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_40, mul_108);  sub_40 = mul_108 = None
    mul_109: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_41);  div_24 = sub_41 = None
    mul_110: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_59, mul_28);  mul_28 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_110, [0, 1]);  mul_110 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_59, [0, 1]);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_7: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_33, torch.float32);  getitem_33 = None
    mul_111: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_112: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_109, mul_111);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_199: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_112, [128, 768]);  mul_112 = None
    mm_26: "f32[128, 3072]" = torch.ops.aten.mm.default(view_199, permute_137);  permute_137 = None
    permute_138: "f32[768, 128]" = torch.ops.aten.permute.default(view_199, [1, 0])
    mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_138, view_90);  permute_138 = view_90 = None
    permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_199, [0], True);  view_199 = None
    view_200: "f32[768]" = torch.ops.aten.reshape.default(sum_50, [768]);  sum_50 = None
    permute_140: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    view_201: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(mm_26, [1, 128, 3072]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_114: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
    mul_115: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, view_89)
    mul_116: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_115, -0.5);  mul_115 = None
    exp_12: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_116);  mul_116 = None
    mul_117: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_118: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_89, mul_117);  view_89 = mul_117 = None
    add_61: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_114, mul_118);  mul_114 = mul_118 = None
    mul_119: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_201, add_61);  view_201 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_202: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_119, [128, 3072]);  mul_119 = None
    mm_28: "f32[128, 768]" = torch.ops.aten.mm.default(view_202, permute_141);  permute_141 = None
    permute_142: "f32[3072, 128]" = torch.ops.aten.permute.default(view_202, [1, 0])
    mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_142, view_88);  permute_142 = view_88 = None
    permute_143: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_51: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_202, [0], True);  view_202 = None
    view_203: "f32[3072]" = torch.ops.aten.reshape.default(sum_51, [3072]);  sum_51 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    view_204: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_28, [1, 128, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_62: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_109, view_204);  mul_109 = view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    mul_121: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_62, primals_61);  primals_61 = None
    mul_122: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_121, 768)
    sum_52: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_121, mul_23);  mul_121 = None
    sum_53: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_23, sum_53);  sum_53 = None
    sub_43: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_122, sum_52);  mul_122 = sum_52 = None
    sub_44: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_43, mul_124);  sub_43 = mul_124 = None
    mul_125: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_44);  div_25 = sub_44 = None
    mul_126: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_62, mul_23);  mul_23 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_62, [0, 1]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_205: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_125, [128, 768])
    mm_30: "f32[128, 768]" = torch.ops.aten.mm.default(view_205, permute_145);  permute_145 = None
    permute_146: "f32[768, 128]" = torch.ops.aten.permute.default(view_205, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_146, view_86);  permute_146 = view_86 = None
    permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_205, [0], True);  view_205 = None
    view_206: "f32[768]" = torch.ops.aten.reshape.default(sum_56, [768]);  sum_56 = None
    permute_148: "f32[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    view_207: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_30, [1, 128, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_208: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_207, [1, 128, 12, 64]);  view_207 = None
    permute_149: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_209: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(permute_149, [12, 128, 64]);  permute_149 = None
    bmm_20: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_150, view_209);  permute_150 = None
    bmm_21: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_209, permute_151);  view_209 = permute_151 = None
    view_210: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_20, [1, 12, 128, 64]);  bmm_20 = None
    view_211: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_21, [1, 12, 128, 128]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_8: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_29, torch.float32);  getitem_29 = None
    mul_127: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_128: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_211, mul_127);  view_211 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    mul_129: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_128, alias_12);  mul_128 = None
    sum_57: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [-1], True)
    mul_130: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_12, sum_57);  alias_12 = sum_57 = None
    sub_45: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_16: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_7, sub_45);  sub_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_212: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(where_16, [12, 128, 128]);  where_16 = None
    bmm_22: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_152, view_212);  permute_152 = None
    bmm_23: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_212, permute_153);  view_212 = permute_153 = None
    view_213: "f32[1, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_22, [1, 12, 64, 128]);  bmm_22 = None
    view_214: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 12, 128, 64]);  bmm_23 = None
    permute_154: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_213, [0, 1, 3, 2]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_26: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_214, 8.0);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_155: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_19: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    view_215: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_19, [1, 128, 768]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_216: "f32[128, 768]" = torch.ops.aten.reshape.default(view_215, [128, 768]);  view_215 = None
    mm_32: "f32[128, 768]" = torch.ops.aten.mm.default(view_216, permute_156);  permute_156 = None
    permute_157: "f32[768, 128]" = torch.ops.aten.permute.default(view_216, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_157, view_69);  permute_157 = None
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
    view_217: "f32[768]" = torch.ops.aten.reshape.default(sum_58, [768]);  sum_58 = None
    permute_159: "f32[768, 768]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    view_218: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_32, [1, 128, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_63: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_125, view_218);  mul_125 = view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_160: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_154, [0, 2, 1, 3]);  permute_154 = None
    view_219: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(permute_160, [1, 128, 768]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_220: "f32[128, 768]" = torch.ops.aten.reshape.default(view_219, [128, 768]);  view_219 = None
    mm_34: "f32[128, 768]" = torch.ops.aten.mm.default(view_220, permute_161);  permute_161 = None
    permute_162: "f32[768, 128]" = torch.ops.aten.permute.default(view_220, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_162, view_69);  permute_162 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_220, [0], True);  view_220 = None
    view_221: "f32[768]" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    view_222: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_34, [1, 128, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_64: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_63, view_222);  add_63 = view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_165: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_26, [0, 2, 1, 3]);  div_26 = None
    clone_20: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    view_223: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_20, [1, 128, 768]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_224: "f32[128, 768]" = torch.ops.aten.reshape.default(view_223, [128, 768]);  view_223 = None
    mm_36: "f32[128, 768]" = torch.ops.aten.mm.default(view_224, permute_166);  permute_166 = None
    permute_167: "f32[768, 128]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_167, view_69);  permute_167 = view_69 = None
    permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[768]" = torch.ops.aten.reshape.default(sum_60, [768]);  sum_60 = None
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_226: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_36, [1, 128, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_65: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_64, view_226);  add_64 = view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    mul_132: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_65, primals_51);  primals_51 = None
    mul_133: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_132, 768)
    sum_61: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True)
    mul_134: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_132, mul_21);  mul_132 = None
    sum_62: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_134, [2], True);  mul_134 = None
    mul_135: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_21, sum_62);  sum_62 = None
    sub_47: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_133, sum_61);  mul_133 = sum_61 = None
    sub_48: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_135);  sub_47 = mul_135 = None
    mul_136: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_48);  div_27 = sub_48 = None
    mul_137: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_65, mul_21);  mul_21 = None
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_137, [0, 1]);  mul_137 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_65, [0, 1]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_9: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_138: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_139: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_136, mul_138);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_227: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_139, [128, 768]);  mul_139 = None
    mm_38: "f32[128, 3072]" = torch.ops.aten.mm.default(view_227, permute_170);  permute_170 = None
    permute_171: "f32[768, 128]" = torch.ops.aten.permute.default(view_227, [1, 0])
    mm_39: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_171, view_67);  permute_171 = view_67 = None
    permute_172: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
    permute_173: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_229: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(mm_38, [1, 128, 3072]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_141: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
    mul_142: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, view_66)
    mul_143: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_142, -0.5);  mul_142 = None
    exp_13: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_143);  mul_143 = None
    mul_144: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_145: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_66, mul_144);  view_66 = mul_144 = None
    add_67: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_141, mul_145);  mul_141 = mul_145 = None
    mul_146: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_229, add_67);  view_229 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_230: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_146, [128, 3072]);  mul_146 = None
    mm_40: "f32[128, 768]" = torch.ops.aten.mm.default(view_230, permute_174);  permute_174 = None
    permute_175: "f32[3072, 128]" = torch.ops.aten.permute.default(view_230, [1, 0])
    mm_41: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_175, view_65);  permute_175 = view_65 = None
    permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_66: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True);  view_230 = None
    view_231: "f32[3072]" = torch.ops.aten.reshape.default(sum_66, [3072]);  sum_66 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    view_232: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_40, [1, 128, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_68: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_136, view_232);  mul_136 = view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    mul_148: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_68, primals_45);  primals_45 = None
    mul_149: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_148, 768)
    sum_67: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
    mul_150: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_148, mul_16);  mul_148 = None
    sum_68: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
    mul_151: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_68);  sum_68 = None
    sub_50: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_149, sum_67);  mul_149 = sum_67 = None
    sub_51: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_151);  sub_50 = mul_151 = None
    mul_152: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_51);  div_28 = sub_51 = None
    mul_153: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_68, mul_16);  mul_16 = None
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_68, [0, 1]);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_233: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_152, [128, 768])
    mm_42: "f32[128, 768]" = torch.ops.aten.mm.default(view_233, permute_178);  permute_178 = None
    permute_179: "f32[768, 128]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_43: "f32[768, 768]" = torch.ops.aten.mm.default(permute_179, view_63);  permute_179 = view_63 = None
    permute_180: "f32[768, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[768]" = torch.ops.aten.reshape.default(sum_71, [768]);  sum_71 = None
    permute_181: "f32[768, 768]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    view_235: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_42, [1, 128, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_236: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_235, [1, 128, 12, 64]);  view_235 = None
    permute_182: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_237: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(permute_182, [12, 128, 64]);  permute_182 = None
    bmm_24: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_183, view_237);  permute_183 = None
    bmm_25: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_237, permute_184);  view_237 = permute_184 = None
    view_238: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_24, [1, 12, 128, 64]);  bmm_24 = None
    view_239: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_25, [1, 12, 128, 128]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_10: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_154: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_155: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_239, mul_154);  view_239 = mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    mul_156: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_155, alias_13);  mul_155 = None
    sum_72: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [-1], True)
    mul_157: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_13, sum_72);  alias_13 = sum_72 = None
    sub_52: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_17: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_7, sub_52);  sub_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_240: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(where_17, [12, 128, 128]);  where_17 = None
    bmm_26: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_185, view_240);  permute_185 = None
    bmm_27: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_240, permute_186);  view_240 = permute_186 = None
    view_241: "f32[1, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_26, [1, 12, 64, 128]);  bmm_26 = None
    view_242: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_27, [1, 12, 128, 64]);  bmm_27 = None
    permute_187: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_241, [0, 1, 3, 2]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_29: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_242, 8.0);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_188: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
    clone_23: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_243: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_23, [1, 128, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_244: "f32[128, 768]" = torch.ops.aten.reshape.default(view_243, [128, 768]);  view_243 = None
    mm_44: "f32[128, 768]" = torch.ops.aten.mm.default(view_244, permute_189);  permute_189 = None
    permute_190: "f32[768, 128]" = torch.ops.aten.permute.default(view_244, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_190, view_46);  permute_190 = None
    permute_191: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True);  view_244 = None
    view_245: "f32[768]" = torch.ops.aten.reshape.default(sum_73, [768]);  sum_73 = None
    permute_192: "f32[768, 768]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    view_246: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_44, [1, 128, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_69: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_152, view_246);  mul_152 = view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_193: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_187, [0, 2, 1, 3]);  permute_187 = None
    view_247: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(permute_193, [1, 128, 768]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_248: "f32[128, 768]" = torch.ops.aten.reshape.default(view_247, [128, 768]);  view_247 = None
    mm_46: "f32[128, 768]" = torch.ops.aten.mm.default(view_248, permute_194);  permute_194 = None
    permute_195: "f32[768, 128]" = torch.ops.aten.permute.default(view_248, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_195, view_46);  permute_195 = None
    permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
    view_249: "f32[768]" = torch.ops.aten.reshape.default(sum_74, [768]);  sum_74 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_250: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_46, [1, 128, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_70: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_69, view_250);  add_69 = view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_198: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_29, [0, 2, 1, 3]);  div_29 = None
    clone_24: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
    view_251: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_24, [1, 128, 768]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_252: "f32[128, 768]" = torch.ops.aten.reshape.default(view_251, [128, 768]);  view_251 = None
    mm_48: "f32[128, 768]" = torch.ops.aten.mm.default(view_252, permute_199);  permute_199 = None
    permute_200: "f32[768, 128]" = torch.ops.aten.permute.default(view_252, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_46);  permute_200 = view_46 = None
    permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
    view_253: "f32[768]" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_254: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_48, [1, 128, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_71: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_70, view_254);  add_70 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    mul_159: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_71, primals_35);  primals_35 = None
    mul_160: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_159, 768)
    sum_76: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True)
    mul_161: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_159, mul_14);  mul_159 = None
    sum_77: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_161, [2], True);  mul_161 = None
    mul_162: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_77);  sum_77 = None
    sub_54: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_160, sum_76);  mul_160 = sum_76 = None
    sub_55: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_54, mul_162);  sub_54 = mul_162 = None
    mul_163: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_55);  div_30 = sub_55 = None
    mul_164: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_71, mul_14);  mul_14 = None
    sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_164, [0, 1]);  mul_164 = None
    sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_71, [0, 1]);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_11: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_165: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_166: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_163, mul_165);  mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_255: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_166, [128, 768]);  mul_166 = None
    mm_50: "f32[128, 3072]" = torch.ops.aten.mm.default(view_255, permute_203);  permute_203 = None
    permute_204: "f32[768, 128]" = torch.ops.aten.permute.default(view_255, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_204, view_44);  permute_204 = view_44 = None
    permute_205: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True);  view_255 = None
    view_256: "f32[768]" = torch.ops.aten.reshape.default(sum_80, [768]);  sum_80 = None
    permute_206: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    view_257: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(mm_50, [1, 128, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_168: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
    mul_169: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, view_43)
    mul_170: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_169, -0.5);  mul_169 = None
    exp_14: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_170);  mul_170 = None
    mul_171: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_172: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_43, mul_171);  view_43 = mul_171 = None
    add_73: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_168, mul_172);  mul_168 = mul_172 = None
    mul_173: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_257, add_73);  view_257 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_258: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_173, [128, 3072]);  mul_173 = None
    mm_52: "f32[128, 768]" = torch.ops.aten.mm.default(view_258, permute_207);  permute_207 = None
    permute_208: "f32[3072, 128]" = torch.ops.aten.permute.default(view_258, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_208, view_42);  permute_208 = view_42 = None
    permute_209: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_81: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
    view_259: "f32[3072]" = torch.ops.aten.reshape.default(sum_81, [3072]);  sum_81 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    view_260: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_52, [1, 128, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_74: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_163, view_260);  mul_163 = view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    mul_175: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_74, primals_29);  primals_29 = None
    mul_176: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_175, 768)
    sum_82: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True)
    mul_177: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_175, mul_9);  mul_175 = None
    sum_83: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True);  mul_177 = None
    mul_178: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_83);  sum_83 = None
    sub_57: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_176, sum_82);  mul_176 = sum_82 = None
    sub_58: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_57, mul_178);  sub_57 = mul_178 = None
    mul_179: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_58);  div_31 = sub_58 = None
    mul_180: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_74, mul_9);  mul_9 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_180, [0, 1]);  mul_180 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_74, [0, 1]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_261: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_179, [128, 768])
    mm_54: "f32[128, 768]" = torch.ops.aten.mm.default(view_261, permute_211);  permute_211 = None
    permute_212: "f32[768, 128]" = torch.ops.aten.permute.default(view_261, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_212, view_40);  permute_212 = view_40 = None
    permute_213: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
    view_262: "f32[768]" = torch.ops.aten.reshape.default(sum_86, [768]);  sum_86 = None
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    view_263: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_54, [1, 128, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_264: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_263, [1, 128, 12, 64]);  view_263 = None
    permute_215: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_265: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(permute_215, [12, 128, 64]);  permute_215 = None
    bmm_28: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_216, view_265);  permute_216 = None
    bmm_29: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_265, permute_217);  view_265 = permute_217 = None
    view_266: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_28, [1, 12, 128, 64]);  bmm_28 = None
    view_267: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_29, [1, 12, 128, 128]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_12: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
    mul_181: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_182: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_267, mul_181);  view_267 = mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    mul_183: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_182, alias_14);  mul_182 = None
    sum_87: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [-1], True)
    mul_184: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_14, sum_87);  alias_14 = sum_87 = None
    sub_59: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_18: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_7, sub_59);  sub_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_268: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(where_18, [12, 128, 128]);  where_18 = None
    bmm_30: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_218, view_268);  permute_218 = None
    bmm_31: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_268, permute_219);  view_268 = permute_219 = None
    view_269: "f32[1, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_30, [1, 12, 64, 128]);  bmm_30 = None
    view_270: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_31, [1, 12, 128, 64]);  bmm_31 = None
    permute_220: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_269, [0, 1, 3, 2]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_32: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_270, 8.0);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_221: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
    clone_27: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    view_271: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_27, [1, 128, 768]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_272: "f32[128, 768]" = torch.ops.aten.reshape.default(view_271, [128, 768]);  view_271 = None
    mm_56: "f32[128, 768]" = torch.ops.aten.mm.default(view_272, permute_222);  permute_222 = None
    permute_223: "f32[768, 128]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_223, view_23);  permute_223 = None
    permute_224: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[768]" = torch.ops.aten.reshape.default(sum_88, [768]);  sum_88 = None
    permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_274: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_56, [1, 128, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_75: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_179, view_274);  mul_179 = view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_226: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_220, [0, 2, 1, 3]);  permute_220 = None
    view_275: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(permute_226, [1, 128, 768]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_276: "f32[128, 768]" = torch.ops.aten.reshape.default(view_275, [128, 768]);  view_275 = None
    mm_58: "f32[128, 768]" = torch.ops.aten.mm.default(view_276, permute_227);  permute_227 = None
    permute_228: "f32[768, 128]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_228, view_23);  permute_228 = None
    permute_229: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[768]" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    view_278: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_58, [1, 128, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_76: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_75, view_278);  add_75 = view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_231: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_32, [0, 2, 1, 3]);  div_32 = None
    clone_28: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
    view_279: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_28, [1, 128, 768]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_280: "f32[128, 768]" = torch.ops.aten.reshape.default(view_279, [128, 768]);  view_279 = None
    mm_60: "f32[128, 768]" = torch.ops.aten.mm.default(view_280, permute_232);  permute_232 = None
    permute_233: "f32[768, 128]" = torch.ops.aten.permute.default(view_280, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_233, view_23);  permute_233 = view_23 = None
    permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
    view_281: "f32[768]" = torch.ops.aten.reshape.default(sum_90, [768]);  sum_90 = None
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    view_282: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_60, [1, 128, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_77: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_76, view_282);  add_76 = view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:314, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    mul_186: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_77, primals_19);  primals_19 = None
    mul_187: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_186, 768)
    sum_91: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True)
    mul_188: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_186, mul_7);  mul_186 = None
    sum_92: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True);  mul_188 = None
    mul_189: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_92);  sum_92 = None
    sub_61: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_187, sum_91);  mul_187 = sum_91 = None
    sub_62: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_61, mul_189);  sub_61 = mul_189 = None
    mul_190: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_62);  div_33 = sub_62 = None
    mul_191: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_77, mul_7);  mul_7 = None
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1]);  mul_191 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_77, [0, 1]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:260, code: x = self.dropout(x)
    convert_element_type_13: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_9, torch.float32);  getitem_9 = None
    mul_192: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_193: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_190, mul_192);  mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:259, code: x = self.lin2(x)
    view_283: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_193, [128, 768]);  mul_193 = None
    mm_62: "f32[128, 3072]" = torch.ops.aten.mm.default(view_283, permute_236);  permute_236 = None
    permute_237: "f32[768, 128]" = torch.ops.aten.permute.default(view_283, [1, 0])
    mm_63: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_237, view_21);  permute_237 = view_21 = None
    permute_238: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
    view_284: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
    permute_239: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_285: "f32[1, 128, 3072]" = torch.ops.aten.reshape.default(mm_62, [1, 128, 3072]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_195: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
    mul_196: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, view_20)
    mul_197: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_196, -0.5);  mul_196 = None
    exp_15: "f32[1, 128, 3072]" = torch.ops.aten.exp.default(mul_197);  mul_197 = None
    mul_198: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_199: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_20, mul_198);  view_20 = mul_198 = None
    add_79: "f32[1, 128, 3072]" = torch.ops.aten.add.Tensor(mul_195, mul_199);  mul_195 = mul_199 = None
    mul_200: "f32[1, 128, 3072]" = torch.ops.aten.mul.Tensor(view_285, add_79);  view_285 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    view_286: "f32[128, 3072]" = torch.ops.aten.reshape.default(mul_200, [128, 3072]);  mul_200 = None
    mm_64: "f32[128, 768]" = torch.ops.aten.mm.default(view_286, permute_240);  permute_240 = None
    permute_241: "f32[3072, 128]" = torch.ops.aten.permute.default(view_286, [1, 0])
    mm_65: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_241, view_19);  permute_241 = view_19 = None
    permute_242: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_286, [0], True);  view_286 = None
    view_287: "f32[3072]" = torch.ops.aten.reshape.default(sum_96, [3072]);  sum_96 = None
    permute_243: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_288: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_64, [1, 128, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:257, code: x = self.lin1(input)
    add_80: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_190, view_288);  mul_190 = view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:310, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    mul_202: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_80, primals_13);  primals_13 = None
    mul_203: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_202, 768)
    sum_97: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
    mul_204: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_202, mul_2);  mul_202 = None
    sum_98: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
    mul_205: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_98);  sum_98 = None
    sub_64: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_203, sum_97);  mul_203 = sum_97 = None
    sub_65: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_64, mul_205);  sub_64 = mul_205 = None
    mul_206: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_65);  div_34 = sub_65 = None
    mul_207: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_80, mul_2);  mul_2 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_80, [0, 1]);  add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:235, code: context = self.out_lin(context)  # (bs, q_length, dim)
    view_289: "f32[128, 768]" = torch.ops.aten.reshape.default(mul_206, [128, 768])
    mm_66: "f32[128, 768]" = torch.ops.aten.mm.default(view_289, permute_244);  permute_244 = None
    permute_245: "f32[768, 128]" = torch.ops.aten.permute.default(view_289, [1, 0])
    mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_245, view_17);  permute_245 = view_17 = None
    permute_246: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
    view_290: "f32[768]" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
    permute_247: "f32[768, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_291: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_66, [1, 128, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:213, code: return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
    view_292: "f32[1, 128, 12, 64]" = torch.ops.aten.reshape.default(view_291, [1, 128, 12, 64]);  view_291 = None
    permute_248: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:233, code: context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    view_293: "f32[12, 128, 64]" = torch.ops.aten.reshape.default(permute_248, [12, 128, 64]);  permute_248 = None
    bmm_32: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(permute_249, view_293);  permute_249 = None
    bmm_33: "f32[12, 128, 128]" = torch.ops.aten.bmm.default(view_293, permute_250);  view_293 = permute_250 = None
    view_294: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_32, [1, 12, 128, 64]);  bmm_32 = None
    view_295: "f32[1, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_33, [1, 12, 128, 128]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:227, code: weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
    convert_element_type_14: "f32[1, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_208: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_209: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_295, mul_208);  view_295 = mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:226, code: weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
    mul_210: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_209, alias_15);  mul_209 = None
    sum_102: "f32[1, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [-1], True)
    mul_211: "f32[1, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_15, sum_102);  alias_15 = sum_102 = None
    sub_66: "f32[1, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:222, code: scores = scores.masked_fill(
    where_19: "f32[1, 12, 128, 128]" = torch.ops.aten.where.self(expand_2, full_default_7, sub_66);  expand_2 = sub_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:220, code: scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    view_296: "f32[12, 128, 128]" = torch.ops.aten.reshape.default(where_19, [12, 128, 128]);  where_19 = None
    bmm_34: "f32[12, 64, 128]" = torch.ops.aten.bmm.default(permute_251, view_296);  permute_251 = None
    bmm_35: "f32[12, 128, 64]" = torch.ops.aten.bmm.default(view_296, permute_252);  view_296 = permute_252 = None
    view_297: "f32[1, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_34, [1, 12, 64, 128]);  bmm_34 = None
    view_298: "f32[1, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_35, [1, 12, 128, 64]);  bmm_35 = None
    permute_253: "f32[1, 12, 128, 64]" = torch.ops.aten.permute.default(view_297, [0, 1, 3, 2]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:219, code: q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    div_35: "f32[1, 12, 128, 64]" = torch.ops.aten.div.Tensor(view_298, 8.0);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_254: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    clone_31: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    view_299: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_31, [1, 128, 768]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    view_300: "f32[128, 768]" = torch.ops.aten.reshape.default(view_299, [128, 768]);  view_299 = None
    mm_68: "f32[128, 768]" = torch.ops.aten.mm.default(view_300, permute_255);  permute_255 = None
    permute_256: "f32[768, 128]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_256, view);  permute_256 = None
    permute_257: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[768]" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    view_302: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_68, [1, 128, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:217, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
    add_81: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(mul_206, view_302);  mul_206 = view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_259: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(permute_253, [0, 2, 1, 3]);  permute_253 = None
    view_303: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(permute_259, [1, 128, 768]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    view_304: "f32[128, 768]" = torch.ops.aten.reshape.default(view_303, [128, 768]);  view_303 = None
    mm_70: "f32[128, 768]" = torch.ops.aten.mm.default(view_304, permute_260);  permute_260 = None
    permute_261: "f32[768, 128]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_261, view);  permute_261 = None
    permute_262: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[768]" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_306: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_70, [1, 128, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:216, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
    add_82: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_81, view_306);  add_81 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:209, code: return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
    permute_264: "f32[1, 128, 12, 64]" = torch.ops.aten.permute.default(div_35, [0, 2, 1, 3]);  div_35 = None
    clone_32: "f32[1, 128, 12, 64]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
    view_307: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(clone_32, [1, 128, 768]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    view_308: "f32[128, 768]" = torch.ops.aten.reshape.default(view_307, [128, 768]);  view_307 = None
    mm_72: "f32[128, 768]" = torch.ops.aten.mm.default(view_308, permute_265);  permute_265 = None
    permute_266: "f32[768, 128]" = torch.ops.aten.permute.default(view_308, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_266, view);  permute_266 = view = None
    permute_267: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
    view_309: "f32[768]" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
    view_310: "f32[1, 128, 768]" = torch.ops.aten.reshape.default(mm_72, [1, 128, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:215, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
    add_83: "f32[1, 128, 768]" = torch.ops.aten.add.Tensor(add_82, view_310);  add_82 = view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:137, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
    convert_element_type_15: "f32[1, 128, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_212: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_213: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(add_83, mul_212);  add_83 = mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:136, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
    mul_215: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_213, primals_3);  primals_3 = None
    mul_216: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_215, 768)
    sum_106: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True)
    mul_217: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_215, mul);  mul_215 = None
    sum_107: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True);  mul_217 = None
    mul_218: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul, sum_107);  sum_107 = None
    sub_68: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(mul_216, sum_106);  mul_216 = sum_106 = None
    sub_69: "f32[1, 128, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_218);  sub_68 = mul_218 = None
    mul_219: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_69);  div_36 = sub_69 = None
    mul_220: "f32[1, 128, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul);  mul = None
    sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1]);  mul_220 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1]);  mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:133, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
    eq_6: "b8[1, 128]" = torch.ops.aten.eq.Scalar(slice_2, -1)
    unsqueeze_6: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_6, -1);  eq_6 = None
    where_20: "f32[1, 128, 768]" = torch.ops.aten.where.self(unsqueeze_6, full_default_7, mul_219);  unsqueeze_6 = None
    full_default_23: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_23, [slice_2], where_20, True);  full_default_23 = slice_2 = where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:120, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
    eq_7: "b8[1, 128]" = torch.ops.aten.eq.Scalar(primals_104, 0)
    unsqueeze_7: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_7, -1);  eq_7 = None
    where_21: "f32[1, 128, 768]" = torch.ops.aten.where.self(unsqueeze_7, full_default_7, mul_219);  unsqueeze_7 = full_default_7 = mul_219 = None
    full_default_25: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[30522, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_25, [primals_104], where_21, True);  full_default_25 = primals_104 = where_21 = None
    return [_unsafe_index_put_1, _unsafe_index_put, sum_108, sum_109, permute_268, view_309, permute_263, view_305, permute_258, view_301, permute_247, view_290, sum_99, sum_100, permute_243, view_287, permute_239, view_284, sum_93, sum_94, permute_235, view_281, permute_230, view_277, permute_225, view_273, permute_214, view_262, sum_84, sum_85, permute_210, view_259, permute_206, view_256, sum_78, sum_79, permute_202, view_253, permute_197, view_249, permute_192, view_245, permute_181, view_234, sum_69, sum_70, permute_177, view_231, permute_173, view_228, sum_63, sum_64, permute_169, view_225, permute_164, view_221, permute_159, view_217, permute_148, view_206, sum_54, sum_55, permute_144, view_203, permute_140, view_200, sum_48, sum_49, permute_136, view_197, permute_131, view_193, permute_126, view_189, permute_115, view_178, sum_39, sum_40, permute_111, view_175, permute_107, view_172, sum_33, sum_34, permute_103, view_169, permute_98, view_165, permute_93, view_161, permute_82, view_150, sum_24, sum_25, permute_78, view_147, permute_74, view_144, sum_18, sum_19, permute_70, view_141, None, None, None, None]
    