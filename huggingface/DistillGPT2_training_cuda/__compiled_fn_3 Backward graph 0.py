from __future__ import annotations



def forward(self, primals_51: "f32[768]", primals_53: "f32[768]", primals_55: "f32[768]", primals_57: "f32[768]", primals_59: "f32[768]", primals_61: "f32[768]", primals_63: "f32[768]", primals_65: "f32[768]", primals_67: "f32[768]", primals_69: "f32[768]", primals_71: "f32[768]", primals_73: "f32[768]", primals_75: "f32[768]", primals_85: "i64[1, 512]", view: "i64[1, 512]", view_1: "i64[1, 512]", getitem_1: "b8[1, 512, 768]", mul: "f32[1, 512, 768]", slice_4: "b8[1, 1, 512, 512]", getitem_8: "b8[1, 12, 512, 512]", getitem_10: "b8[1, 512, 768]", mul_2: "f32[1, 512, 768]", addmm_2: "f32[512, 3072]", tanh: "f32[1, 512, 3072]", getitem_14: "b8[1, 512, 768]", mul_8: "f32[1, 512, 768]", slice_8: "b8[1, 1, 512, 512]", getitem_21: "b8[1, 12, 512, 512]", getitem_23: "b8[1, 512, 768]", mul_10: "f32[1, 512, 768]", addmm_6: "f32[512, 3072]", tanh_1: "f32[1, 512, 3072]", getitem_27: "b8[1, 512, 768]", mul_16: "f32[1, 512, 768]", slice_12: "b8[1, 1, 512, 512]", getitem_34: "b8[1, 12, 512, 512]", getitem_36: "b8[1, 512, 768]", mul_18: "f32[1, 512, 768]", addmm_10: "f32[512, 3072]", tanh_2: "f32[1, 512, 3072]", getitem_40: "b8[1, 512, 768]", mul_24: "f32[1, 512, 768]", slice_16: "b8[1, 1, 512, 512]", getitem_47: "b8[1, 12, 512, 512]", getitem_49: "b8[1, 512, 768]", mul_26: "f32[1, 512, 768]", addmm_14: "f32[512, 3072]", tanh_3: "f32[1, 512, 3072]", getitem_53: "b8[1, 512, 768]", mul_32: "f32[1, 512, 768]", slice_20: "b8[1, 1, 512, 512]", getitem_60: "b8[1, 12, 512, 512]", getitem_62: "b8[1, 512, 768]", mul_34: "f32[1, 512, 768]", addmm_18: "f32[512, 3072]", tanh_4: "f32[1, 512, 3072]", getitem_66: "b8[1, 512, 768]", mul_40: "f32[1, 512, 768]", slice_24: "b8[1, 1, 512, 512]", getitem_73: "b8[1, 12, 512, 512]", getitem_75: "b8[1, 512, 768]", mul_42: "f32[1, 512, 768]", addmm_22: "f32[512, 3072]", tanh_5: "f32[1, 512, 3072]", getitem_79: "b8[1, 512, 768]", mul_48: "f32[1, 512, 768]", view_111: "f32[512, 768]", sub_20: "f32[511, 50257]", convert_element_type_6: "f32[]", permute_33: "f32[50257, 768]", div_14: "f32[1, 512, 1]", permute_35: "f32[768, 3072]", permute_36: "f32[3072, 512]", permute_37: "f32[3072, 768]", permute_38: "f32[768, 512]", div_15: "f32[1, 512, 1]", permute_39: "f32[768, 768]", permute_40: "f32[768, 512]", permute_42: "f32[12, 512, 512]", permute_43: "f32[12, 64, 512]", alias_15: "f32[1, 12, 512, 512]", permute_44: "f32[12, 64, 512]", permute_45: "f32[12, 512, 64]", permute_50: "f32[2304, 768]", permute_51: "f32[768, 512]", div_17: "f32[1, 512, 1]", permute_52: "f32[768, 3072]", permute_53: "f32[3072, 512]", permute_54: "f32[3072, 768]", permute_55: "f32[768, 512]", div_18: "f32[1, 512, 1]", permute_56: "f32[768, 768]", permute_57: "f32[768, 512]", permute_59: "f32[12, 512, 512]", permute_60: "f32[12, 64, 512]", alias_17: "f32[1, 12, 512, 512]", permute_61: "f32[12, 64, 512]", permute_62: "f32[12, 512, 64]", permute_67: "f32[2304, 768]", permute_68: "f32[768, 512]", div_20: "f32[1, 512, 1]", permute_69: "f32[768, 3072]", permute_70: "f32[3072, 512]", permute_71: "f32[3072, 768]", permute_72: "f32[768, 512]", div_21: "f32[1, 512, 1]", permute_73: "f32[768, 768]", permute_74: "f32[768, 512]", permute_76: "f32[12, 512, 512]", permute_77: "f32[12, 64, 512]", alias_19: "f32[1, 12, 512, 512]", permute_78: "f32[12, 64, 512]", permute_79: "f32[12, 512, 64]", permute_84: "f32[2304, 768]", permute_85: "f32[768, 512]", div_23: "f32[1, 512, 1]", permute_86: "f32[768, 3072]", permute_87: "f32[3072, 512]", permute_88: "f32[3072, 768]", permute_89: "f32[768, 512]", div_24: "f32[1, 512, 1]", permute_90: "f32[768, 768]", permute_91: "f32[768, 512]", permute_93: "f32[12, 512, 512]", permute_94: "f32[12, 64, 512]", alias_21: "f32[1, 12, 512, 512]", permute_95: "f32[12, 64, 512]", permute_96: "f32[12, 512, 64]", permute_101: "f32[2304, 768]", permute_102: "f32[768, 512]", div_26: "f32[1, 512, 1]", permute_103: "f32[768, 3072]", permute_104: "f32[3072, 512]", permute_105: "f32[3072, 768]", permute_106: "f32[768, 512]", div_27: "f32[1, 512, 1]", permute_107: "f32[768, 768]", permute_108: "f32[768, 512]", permute_110: "f32[12, 512, 512]", permute_111: "f32[12, 64, 512]", alias_23: "f32[1, 12, 512, 512]", permute_112: "f32[12, 64, 512]", permute_113: "f32[12, 512, 64]", permute_118: "f32[2304, 768]", permute_119: "f32[768, 512]", div_29: "f32[1, 512, 1]", permute_120: "f32[768, 3072]", permute_121: "f32[3072, 512]", permute_122: "f32[3072, 768]", permute_123: "f32[768, 512]", div_30: "f32[1, 512, 1]", permute_124: "f32[768, 768]", permute_125: "f32[768, 512]", permute_127: "f32[12, 512, 512]", permute_128: "f32[12, 64, 512]", alias_25: "f32[1, 12, 512, 512]", permute_129: "f32[12, 64, 512]", permute_130: "f32[12, 512, 64]", permute_135: "f32[2304, 768]", permute_136: "f32[768, 512]", div_32: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512, 50257]", tangents_3: "f32[1, 12, 512, 64]", tangents_4: "f32[1, 12, 512, 64]", tangents_5: "f32[1, 12, 512, 64]", tangents_6: "f32[1, 12, 512, 64]", tangents_7: "f32[1, 12, 512, 64]", tangents_8: "f32[1, 12, 512, 64]", tangents_9: "f32[1, 12, 512, 64]", tangents_10: "f32[1, 12, 512, 64]", tangents_11: "f32[1, 12, 512, 64]", tangents_12: "f32[1, 12, 512, 64]", tangents_13: "f32[1, 12, 512, 64]", tangents_14: "f32[1, 12, 512, 64]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_17: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_2, [1, 512, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    alias_1: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh)
    add_7: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_35: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_6, [1, 512, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    alias_3: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_1)
    add_15: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_53: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
    alias_5: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_2)
    add_23: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_71: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_14, [1, 512, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    alias_7: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_3)
    add_31: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_89: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_18, [1, 512, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    alias_9: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_4)
    add_39: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    alias_11: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_5)
    add_47: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1106, code: shift_labels = labels[..., 1:].contiguous()
    slice_27: "i64[1, 511]" = torch.ops.aten.slice.Tensor(primals_85, 1, 1, 9223372036854775807);  primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1109, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_114: "i64[511]" = torch.ops.aten.view.default(slice_27, [-1]);  slice_27 = None
    alias_12: "f32[511, 50257]" = torch.ops.aten.alias.default(sub_20);  sub_20 = None
    full_default_12: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_13: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_6);  tangents_1 = convert_element_type_6 = None
    unsqueeze_2: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(view_114, 1);  view_114 = None
    ne_3: "b8[511, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_2, -100)
    where_8: "i64[511, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_2, full_default_12);  unsqueeze_2 = full_default_12 = None
    full_default_15: "f32[511, 50257]" = torch.ops.aten.full.default([511, 50257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[511, 50257]" = torch.ops.aten.scatter.value(full_default_15, 1, where_8, -1.0);  full_default_15 = where_8 = None
    where_9: "f32[511, 1]" = torch.ops.aten.where.self(ne_3, div_13, full_default_13);  ne_3 = div_13 = None
    mul_50: "f32[511, 50257]" = torch.ops.aten.mul.Tensor(scatter, where_9);  scatter = where_9 = None
    alias_13: "f32[511, 50257]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    exp_7: "f32[511, 50257]" = torch.ops.aten.exp.default(alias_13);  alias_13 = None
    sum_10: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(mul_50, [1], True)
    mul_51: "f32[511, 50257]" = torch.ops.aten.mul.Tensor(exp_7, sum_10);  exp_7 = sum_10 = None
    sub_21: "f32[511, 50257]" = torch.ops.aten.sub.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    view_115: "f32[1, 511, 50257]" = torch.ops.aten.view.default(sub_21, [1, 511, 50257]);  sub_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1105, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    full_default_17: "f32[1, 511, 50257]" = torch.ops.aten.full.default([1, 511, 50257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[1, 511, 50257]" = torch.ops.aten.slice_scatter.default(full_default_17, view_115, 2, 0, 9223372036854775807);  full_default_17 = view_115 = None
    full_default_18: "f32[1, 512, 50257]" = torch.ops.aten.full.default([1, 512, 50257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_1: "f32[1, 512, 50257]" = torch.ops.aten.slice_scatter.default(full_default_18, slice_scatter, 1, 0, -1);  full_default_18 = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1105, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    add_51: "f32[1, 512, 50257]" = torch.ops.aten.add.Tensor(tangents_2, slice_scatter_1);  tangents_2 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1098, code: lm_logits = self.lm_head(hidden_states)
    view_116: "f32[512, 50257]" = torch.ops.aten.view.default(add_51, [512, 50257]);  add_51 = None
    permute_31: "f32[50257, 512]" = torch.ops.aten.permute.default(view_116, [1, 0])
    mm_1: "f32[50257, 768]" = torch.ops.aten.mm.default(permute_31, view_111);  permute_31 = view_111 = None
    permute_32: "f32[768, 50257]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    mm_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_116, permute_33);  view_116 = permute_33 = None
    view_117: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_2, [1, 512, 768]);  mm_2 = None
    permute_34: "f32[50257, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_117, primals_75);  primals_75 = None
    mul_54: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_53, 768)
    sum_11: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_53, [2], True)
    mul_55: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_53, mul_48);  mul_53 = None
    sum_12: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_55, [2], True);  mul_55 = None
    mul_56: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, sum_12);  sum_12 = None
    sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_54, sum_11);  mul_54 = sum_11 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_23, mul_56);  sub_23 = mul_56 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_24);  div_14 = sub_24 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_117, mul_48);  mul_48 = None
    sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_58, [0, 1]);  mul_58 = None
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_117, [0, 1]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, mul_59);  mul_59 = None
    clone_6: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_60, memory_format = torch.contiguous_format);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_119: "f32[512, 768]" = torch.ops.aten.view.default(clone_6, [512, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_3: "f32[512, 3072]" = torch.ops.aten.mm.default(view_119, permute_35);  permute_35 = None
    mm_4: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_36, view_119);  permute_36 = None
    sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_119, [0], True);  view_119 = None
    view_120: "f32[768]" = torch.ops.aten.view.default(sum_15, [768]);  sum_15 = None
    view_121: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_3, [1, 512, 3072]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_121, mul_44);  mul_44 = None
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_121, add_47);  view_121 = add_47 = None
    alias_14: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_14, alias_14);  alias_14 = None
    sub_25: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_63);  mul_63 = None
    mul_64: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, sub_25);  mul_61 = sub_25 = None
    mul_65: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_64, 0.7978845608028654);  mul_64 = None
    mul_66: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_65, 0.044715)
    pow_7: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 2.0);  view_107 = None
    mul_67: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_7, 3.0);  pow_7 = None
    mul_68: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_52: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_65, mul_68);  mul_65 = mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_62, 0.5);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_53: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_52, mul_69);  add_52 = mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_122: "f32[512, 3072]" = torch.ops.aten.view.default(add_53, [512, 3072]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_5: "f32[512, 768]" = torch.ops.aten.mm.default(view_122, permute_37);  permute_37 = None
    mm_6: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_38, view_122);  permute_38 = None
    sum_16: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
    view_123: "f32[3072]" = torch.ops.aten.view.default(sum_16, [3072]);  sum_16 = None
    view_124: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_5, [1, 512, 768]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_71: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_124, primals_73);  primals_73 = None
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, 768)
    sum_17: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_71, [2], True)
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, mul_42);  mul_71 = None
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_73, [2], True);  mul_73 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_18);  sum_18 = None
    sub_27: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_72, sum_17);  mul_72 = sum_17 = None
    sub_28: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_27, mul_74);  sub_27 = mul_74 = None
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_28);  div_15 = sub_28 = None
    mul_76: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_124, mul_42);  mul_42 = None
    sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_76, [0, 1]);  mul_76 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_124, [0, 1]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_54: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_57, mul_75);  mul_57 = mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_8: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_77: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_78: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_54, mul_77);  mul_77 = None
    clone_7: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_78, memory_format = torch.contiguous_format);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_125: "f32[512, 768]" = torch.ops.aten.view.default(clone_7, [512, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_7: "f32[512, 768]" = torch.ops.aten.mm.default(view_125, permute_39);  permute_39 = None
    mm_8: "f32[768, 768]" = torch.ops.aten.mm.default(permute_40, view_125);  permute_40 = None
    sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
    view_126: "f32[768]" = torch.ops.aten.view.default(sum_21, [768]);  sum_21 = None
    view_127: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_7, [1, 512, 768]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_128: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_127, [1, 512, 12, 64]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_41: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_129: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_41, [12, 512, 64]);  permute_41 = None
    bmm_12: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_42, view_129);  permute_42 = None
    bmm_13: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_129, permute_43);  view_129 = permute_43 = None
    view_130: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_12, [1, 12, 512, 64]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_55: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_14, view_130);  tangents_14 = view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_131: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_13, [1, 12, 512, 512]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_9: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_79: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_80: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_131, mul_79);  view_131 = mul_79 = None
    clone_8: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_80, memory_format = torch.contiguous_format);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_81: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_8, alias_15);  clone_8 = None
    sum_22: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_81, [-1], True)
    mul_82: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_15, sum_22);  alias_15 = sum_22 = None
    sub_29: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_10: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_24, sub_29, full_default_13);  slice_24 = sub_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_16: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_10, full_default);  where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_132: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_16, [12, 512, 512]);  div_16 = None
    bmm_14: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_44, view_132);  permute_44 = None
    bmm_15: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_132, permute_45);  view_132 = permute_45 = None
    view_133: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_14, [1, 12, 64, 512]);  bmm_14 = None
    view_134: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, [1, 12, 512, 64]);  bmm_15 = None
    permute_46: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_133, [0, 1, 3, 2]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_56: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_13, permute_46);  tangents_13 = permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_55, [0, 2, 1, 3]);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_9: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    view_135: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_9, [1, 512, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_48: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_56, [0, 2, 1, 3]);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    view_136: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_10, [1, 512, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_49: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_11: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_137: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_11, [1, 512, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat: "f32[1, 512, 2304]" = torch.ops.aten.cat.default([view_137, view_136, view_135], 2);  view_137 = view_136 = view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_138: "f32[512, 2304]" = torch.ops.aten.view.default(cat, [512, 2304]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_9: "f32[512, 768]" = torch.ops.aten.mm.default(view_138, permute_50);  permute_50 = None
    mm_10: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_51, view_138);  permute_51 = None
    sum_23: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_138, [0], True);  view_138 = None
    view_139: "f32[2304]" = torch.ops.aten.view.default(sum_23, [2304]);  sum_23 = None
    view_140: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_9, [1, 512, 768]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_84: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_140, primals_71);  primals_71 = None
    mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, 768)
    sum_24: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_84, [2], True)
    mul_86: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, mul_40);  mul_84 = None
    sum_25: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_86, [2], True);  mul_86 = None
    mul_87: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_25);  sum_25 = None
    sub_31: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_85, sum_24);  mul_85 = sum_24 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_31, mul_87);  sub_31 = mul_87 = None
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_32);  div_17 = sub_32 = None
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_140, mul_40);  mul_40 = None
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_89, [0, 1]);  mul_89 = None
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_140, [0, 1]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_54, mul_88);  add_54 = mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_66, torch.float32);  getitem_66 = None
    mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_91: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_57, mul_90);  mul_90 = None
    clone_12: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_91, memory_format = torch.contiguous_format);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_141: "f32[512, 768]" = torch.ops.aten.view.default(clone_12, [512, 768]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_11: "f32[512, 3072]" = torch.ops.aten.mm.default(view_141, permute_52);  permute_52 = None
    mm_12: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_53, view_141);  permute_53 = None
    sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
    view_142: "f32[768]" = torch.ops.aten.view.default(sum_28, [768]);  sum_28 = None
    view_143: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_11, [1, 512, 3072]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_143, mul_36);  mul_36 = None
    mul_93: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_143, add_39);  view_143 = add_39 = None
    alias_16: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_94: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_16, alias_16);  alias_16 = None
    sub_33: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_94);  mul_94 = None
    mul_95: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_92, sub_33);  mul_92 = sub_33 = None
    mul_96: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_95, 0.7978845608028654);  mul_95 = None
    mul_97: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_96, 0.044715)
    pow_8: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 2.0);  view_89 = None
    mul_98: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_8, 3.0);  pow_8 = None
    mul_99: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_58: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_96, mul_99);  mul_96 = mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_100: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_93, 0.5);  mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_59: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_58, mul_100);  add_58 = mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_144: "f32[512, 3072]" = torch.ops.aten.view.default(add_59, [512, 3072]);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_13: "f32[512, 768]" = torch.ops.aten.mm.default(view_144, permute_54);  permute_54 = None
    mm_14: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_55, view_144);  permute_55 = None
    sum_29: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
    view_145: "f32[3072]" = torch.ops.aten.view.default(sum_29, [3072]);  sum_29 = None
    view_146: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_13, [1, 512, 768]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_146, primals_69);  primals_69 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, 768)
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_102, [2], True)
    mul_104: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, mul_34);  mul_102 = None
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [2], True);  mul_104 = None
    mul_105: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, sum_31);  sum_31 = None
    sub_35: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_103, sum_30);  mul_103 = sum_30 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_105);  sub_35 = mul_105 = None
    mul_106: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_36);  div_18 = sub_36 = None
    mul_107: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_146, mul_34);  mul_34 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_107, [0, 1]);  mul_107 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_146, [0, 1]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_60: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_57, mul_106);  add_57 = mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_11: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_62, torch.float32);  getitem_62 = None
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_60, mul_108);  mul_108 = None
    clone_13: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_109, memory_format = torch.contiguous_format);  mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_147: "f32[512, 768]" = torch.ops.aten.view.default(clone_13, [512, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_15: "f32[512, 768]" = torch.ops.aten.mm.default(view_147, permute_56);  permute_56 = None
    mm_16: "f32[768, 768]" = torch.ops.aten.mm.default(permute_57, view_147);  permute_57 = None
    sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
    view_148: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    view_149: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_15, [1, 512, 768]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_150: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_149, [1, 512, 12, 64]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_58: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_151: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_58, [12, 512, 64]);  permute_58 = None
    bmm_16: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_59, view_151);  permute_59 = None
    bmm_17: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_151, permute_60);  view_151 = permute_60 = None
    view_152: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_16, [1, 12, 512, 64]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_61: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_12, view_152);  tangents_12 = view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_153: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_17, [1, 12, 512, 512]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_12: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_60, torch.float32);  getitem_60 = None
    mul_110: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_111: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_153, mul_110);  view_153 = mul_110 = None
    clone_14: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_111, memory_format = torch.contiguous_format);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_112: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_14, alias_17);  clone_14 = None
    sum_35: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_112, [-1], True)
    mul_113: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_17, sum_35);  alias_17 = sum_35 = None
    sub_37: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_11: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_20, sub_37, full_default_13);  slice_20 = sub_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_19: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_11, full_default);  where_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_154: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_19, [12, 512, 512]);  div_19 = None
    bmm_18: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_61, view_154);  permute_61 = None
    bmm_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_154, permute_62);  view_154 = permute_62 = None
    view_155: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_18, [1, 12, 64, 512]);  bmm_18 = None
    view_156: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, [1, 12, 512, 64]);  bmm_19 = None
    permute_63: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_155, [0, 1, 3, 2]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_62: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_11, permute_63);  tangents_11 = permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_64: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_61, [0, 2, 1, 3]);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_15: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    view_157: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_15, [1, 512, 768]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_65: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_62, [0, 2, 1, 3]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_16: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
    view_158: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_16, [1, 512, 768]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_66: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_17: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    view_159: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_17, [1, 512, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_1: "f32[1, 512, 2304]" = torch.ops.aten.cat.default([view_159, view_158, view_157], 2);  view_159 = view_158 = view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_160: "f32[512, 2304]" = torch.ops.aten.view.default(cat_1, [512, 2304]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_17: "f32[512, 768]" = torch.ops.aten.mm.default(view_160, permute_67);  permute_67 = None
    mm_18: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_68, view_160);  permute_68 = None
    sum_36: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True);  view_160 = None
    view_161: "f32[2304]" = torch.ops.aten.view.default(sum_36, [2304]);  sum_36 = None
    view_162: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_17, [1, 512, 768]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_162, primals_67);  primals_67 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, 768)
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_115, [2], True)
    mul_117: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, mul_32);  mul_115 = None
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True);  mul_117 = None
    mul_118: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, sum_38);  sum_38 = None
    sub_39: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_116, sum_37);  mul_116 = sum_37 = None
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_39, mul_118);  sub_39 = mul_118 = None
    mul_119: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_40);  div_20 = sub_40 = None
    mul_120: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_162, mul_32);  mul_32 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_120, [0, 1]);  mul_120 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_162, [0, 1]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_60, mul_119);  add_60 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_63, mul_121);  mul_121 = None
    clone_18: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_122, memory_format = torch.contiguous_format);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_163: "f32[512, 768]" = torch.ops.aten.view.default(clone_18, [512, 768]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_19: "f32[512, 3072]" = torch.ops.aten.mm.default(view_163, permute_69);  permute_69 = None
    mm_20: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_70, view_163);  permute_70 = None
    sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_163, [0], True);  view_163 = None
    view_164: "f32[768]" = torch.ops.aten.view.default(sum_41, [768]);  sum_41 = None
    view_165: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_19, [1, 512, 3072]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_123: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_165, mul_28);  mul_28 = None
    mul_124: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_165, add_31);  view_165 = add_31 = None
    alias_18: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_125: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_18, alias_18);  alias_18 = None
    sub_41: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_125);  mul_125 = None
    mul_126: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_123, sub_41);  mul_123 = sub_41 = None
    mul_127: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_126, 0.7978845608028654);  mul_126 = None
    mul_128: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_127, 0.044715)
    pow_9: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 2.0);  view_71 = None
    mul_129: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_9, 3.0);  pow_9 = None
    mul_130: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_128, mul_129);  mul_128 = mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_64: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_127, mul_130);  mul_127 = mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_131: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_124, 0.5);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_65: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_64, mul_131);  add_64 = mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_166: "f32[512, 3072]" = torch.ops.aten.view.default(add_65, [512, 3072]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_21: "f32[512, 768]" = torch.ops.aten.mm.default(view_166, permute_71);  permute_71 = None
    mm_22: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_72, view_166);  permute_72 = None
    sum_42: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_166, [0], True);  view_166 = None
    view_167: "f32[3072]" = torch.ops.aten.view.default(sum_42, [3072]);  sum_42 = None
    view_168: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_21, [1, 512, 768]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_133: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_168, primals_65);  primals_65 = None
    mul_134: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_133, 768)
    sum_43: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [2], True)
    mul_135: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_133, mul_26);  mul_133 = None
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True);  mul_135 = None
    mul_136: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, sum_44);  sum_44 = None
    sub_43: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_134, sum_43);  mul_134 = sum_43 = None
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_43, mul_136);  sub_43 = mul_136 = None
    mul_137: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_44);  div_21 = sub_44 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_168, mul_26);  mul_26 = None
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_138, [0, 1]);  mul_138 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_168, [0, 1]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_66: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_63, mul_137);  add_63 = mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_66, mul_139);  mul_139 = None
    clone_19: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_140, memory_format = torch.contiguous_format);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_169: "f32[512, 768]" = torch.ops.aten.view.default(clone_19, [512, 768]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_23: "f32[512, 768]" = torch.ops.aten.mm.default(view_169, permute_73);  permute_73 = None
    mm_24: "f32[768, 768]" = torch.ops.aten.mm.default(permute_74, view_169);  permute_74 = None
    sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_169, [0], True);  view_169 = None
    view_170: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
    view_171: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_23, [1, 512, 768]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_172: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_171, [1, 512, 12, 64]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_75: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_173: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_75, [12, 512, 64]);  permute_75 = None
    bmm_20: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_76, view_173);  permute_76 = None
    bmm_21: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_173, permute_77);  view_173 = permute_77 = None
    view_174: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_20, [1, 12, 512, 64]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_67: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_10, view_174);  tangents_10 = view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_175: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_21, [1, 12, 512, 512]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_15: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_141: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_142: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_175, mul_141);  view_175 = mul_141 = None
    clone_20: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_142, memory_format = torch.contiguous_format);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_143: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_20, alias_19);  clone_20 = None
    sum_48: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [-1], True)
    mul_144: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_19, sum_48);  alias_19 = sum_48 = None
    sub_45: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_143, mul_144);  mul_143 = mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_12: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_16, sub_45, full_default_13);  slice_16 = sub_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_22: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_12, full_default);  where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_176: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_22, [12, 512, 512]);  div_22 = None
    bmm_22: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_78, view_176);  permute_78 = None
    bmm_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_176, permute_79);  view_176 = permute_79 = None
    view_177: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_22, [1, 12, 64, 512]);  bmm_22 = None
    view_178: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
    permute_80: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_177, [0, 1, 3, 2]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_68: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_9, permute_80);  tangents_9 = permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_81: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_67, [0, 2, 1, 3]);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_21: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    view_179: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_21, [1, 512, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_82: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_68, [0, 2, 1, 3]);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_22: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_180: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_22, [1, 512, 768]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_83: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_23: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    view_181: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_23, [1, 512, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_2: "f32[1, 512, 2304]" = torch.ops.aten.cat.default([view_181, view_180, view_179], 2);  view_181 = view_180 = view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_182: "f32[512, 2304]" = torch.ops.aten.view.default(cat_2, [512, 2304]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_25: "f32[512, 768]" = torch.ops.aten.mm.default(view_182, permute_84);  permute_84 = None
    mm_26: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_85, view_182);  permute_85 = None
    sum_49: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
    view_183: "f32[2304]" = torch.ops.aten.view.default(sum_49, [2304]);  sum_49 = None
    view_184: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_25, [1, 512, 768]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_146: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_184, primals_63);  primals_63 = None
    mul_147: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_146, 768)
    sum_50: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_146, [2], True)
    mul_148: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_146, mul_24);  mul_146 = None
    sum_51: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True);  mul_148 = None
    mul_149: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_51);  sum_51 = None
    sub_47: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_147, sum_50);  mul_147 = sum_50 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_149);  sub_47 = mul_149 = None
    mul_150: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_48);  div_23 = sub_48 = None
    mul_151: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_184, mul_24);  mul_24 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_151, [0, 1]);  mul_151 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_184, [0, 1]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_66, mul_150);  add_66 = mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_40, torch.float32);  getitem_40 = None
    mul_152: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_153: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_69, mul_152);  mul_152 = None
    clone_24: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_153, memory_format = torch.contiguous_format);  mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_185: "f32[512, 768]" = torch.ops.aten.view.default(clone_24, [512, 768]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_27: "f32[512, 3072]" = torch.ops.aten.mm.default(view_185, permute_86);  permute_86 = None
    mm_28: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_87, view_185);  permute_87 = None
    sum_54: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
    view_186: "f32[768]" = torch.ops.aten.view.default(sum_54, [768]);  sum_54 = None
    view_187: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_27, [1, 512, 3072]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_154: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_187, mul_20);  mul_20 = None
    mul_155: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_187, add_23);  view_187 = add_23 = None
    alias_20: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_156: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_20, alias_20);  alias_20 = None
    sub_49: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_156);  mul_156 = None
    mul_157: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_154, sub_49);  mul_154 = sub_49 = None
    mul_158: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_157, 0.7978845608028654);  mul_157 = None
    mul_159: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_158, 0.044715)
    pow_10: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 2.0);  view_53 = None
    mul_160: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_10, 3.0);  pow_10 = None
    mul_161: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_70: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_158, mul_161);  mul_158 = mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_162: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_155, 0.5);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_71: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_70, mul_162);  add_70 = mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_188: "f32[512, 3072]" = torch.ops.aten.view.default(add_71, [512, 3072]);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_29: "f32[512, 768]" = torch.ops.aten.mm.default(view_188, permute_88);  permute_88 = None
    mm_30: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_89, view_188);  permute_89 = None
    sum_55: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
    view_189: "f32[3072]" = torch.ops.aten.view.default(sum_55, [3072]);  sum_55 = None
    view_190: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_29, [1, 512, 768]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_164: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_190, primals_61);  primals_61 = None
    mul_165: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
    sum_56: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
    mul_166: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_18);  mul_164 = None
    sum_57: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
    mul_167: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_57);  sum_57 = None
    sub_51: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_56);  mul_165 = sum_56 = None
    sub_52: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_167);  sub_51 = mul_167 = None
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_52);  div_24 = sub_52 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_190, mul_18);  mul_18 = None
    sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_190, [0, 1]);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_72: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_69, mul_168);  add_69 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_36, torch.float32);  getitem_36 = None
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_72, mul_170);  mul_170 = None
    clone_25: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_171, memory_format = torch.contiguous_format);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_191: "f32[512, 768]" = torch.ops.aten.view.default(clone_25, [512, 768]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_31: "f32[512, 768]" = torch.ops.aten.mm.default(view_191, permute_90);  permute_90 = None
    mm_32: "f32[768, 768]" = torch.ops.aten.mm.default(permute_91, view_191);  permute_91 = None
    sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True);  view_191 = None
    view_192: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
    view_193: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_31, [1, 512, 768]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_194: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_193, [1, 512, 12, 64]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_92: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_195: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_92, [12, 512, 64]);  permute_92 = None
    bmm_24: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_93, view_195);  permute_93 = None
    bmm_25: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_195, permute_94);  view_195 = permute_94 = None
    view_196: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, [1, 12, 512, 64]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_73: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_8, view_196);  tangents_8 = view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_197: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 12, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_18: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_34, torch.float32);  getitem_34 = None
    mul_172: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_173: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_197, mul_172);  view_197 = mul_172 = None
    clone_26: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_173, memory_format = torch.contiguous_format);  mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_174: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_26, alias_21);  clone_26 = None
    sum_61: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_174, [-1], True)
    mul_175: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_21, sum_61);  alias_21 = sum_61 = None
    sub_53: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_13: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_12, sub_53, full_default_13);  slice_12 = sub_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_25: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_13, full_default);  where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_198: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_25, [12, 512, 512]);  div_25 = None
    bmm_26: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_95, view_198);  permute_95 = None
    bmm_27: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_198, permute_96);  view_198 = permute_96 = None
    view_199: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, [1, 12, 64, 512]);  bmm_26 = None
    view_200: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 12, 512, 64]);  bmm_27 = None
    permute_97: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_199, [0, 1, 3, 2]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_74: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_7, permute_97);  tangents_7 = permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_98: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_73, [0, 2, 1, 3]);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_27: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
    view_201: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_27, [1, 512, 768]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_99: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_74, [0, 2, 1, 3]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_28: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
    view_202: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_28, [1, 512, 768]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_100: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_29: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
    view_203: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_29, [1, 512, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_3: "f32[1, 512, 2304]" = torch.ops.aten.cat.default([view_203, view_202, view_201], 2);  view_203 = view_202 = view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_204: "f32[512, 2304]" = torch.ops.aten.view.default(cat_3, [512, 2304]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_33: "f32[512, 768]" = torch.ops.aten.mm.default(view_204, permute_101);  permute_101 = None
    mm_34: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_102, view_204);  permute_102 = None
    sum_62: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
    view_205: "f32[2304]" = torch.ops.aten.view.default(sum_62, [2304]);  sum_62 = None
    view_206: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_33, [1, 512, 768]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_177: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_206, primals_59);  primals_59 = None
    mul_178: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_177, 768)
    sum_63: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True)
    mul_179: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_177, mul_16);  mul_177 = None
    sum_64: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True);  mul_179 = None
    mul_180: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_64);  sum_64 = None
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_178, sum_63);  mul_178 = sum_63 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_180);  sub_55 = mul_180 = None
    mul_181: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_56);  div_26 = sub_56 = None
    mul_182: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_206, mul_16);  mul_16 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1]);  mul_182 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_206, [0, 1]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_72, mul_181);  add_72 = mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_183: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_184: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_75, mul_183);  mul_183 = None
    clone_30: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_184, memory_format = torch.contiguous_format);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_207: "f32[512, 768]" = torch.ops.aten.view.default(clone_30, [512, 768]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_35: "f32[512, 3072]" = torch.ops.aten.mm.default(view_207, permute_103);  permute_103 = None
    mm_36: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_104, view_207);  permute_104 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
    view_208: "f32[768]" = torch.ops.aten.view.default(sum_67, [768]);  sum_67 = None
    view_209: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_35, [1, 512, 3072]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_185: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_209, mul_12);  mul_12 = None
    mul_186: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_209, add_15);  view_209 = add_15 = None
    alias_22: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_187: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_22, alias_22);  alias_22 = None
    sub_57: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_187);  mul_187 = None
    mul_188: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_185, sub_57);  mul_185 = sub_57 = None
    mul_189: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_188, 0.7978845608028654);  mul_188 = None
    mul_190: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_189, 0.044715)
    pow_11: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 2.0);  view_35 = None
    mul_191: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_11, 3.0);  pow_11 = None
    mul_192: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_76: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_189, mul_192);  mul_189 = mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_193: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_186, 0.5);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_77: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_76, mul_193);  add_76 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_210: "f32[512, 3072]" = torch.ops.aten.view.default(add_77, [512, 3072]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_37: "f32[512, 768]" = torch.ops.aten.mm.default(view_210, permute_105);  permute_105 = None
    mm_38: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_106, view_210);  permute_106 = None
    sum_68: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
    view_211: "f32[3072]" = torch.ops.aten.view.default(sum_68, [3072]);  sum_68 = None
    view_212: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_37, [1, 512, 768]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_195: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_212, primals_57);  primals_57 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, 768)
    sum_69: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, mul_10);  mul_195 = None
    sum_70: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_70);  sum_70 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_196, sum_69);  mul_196 = sum_69 = None
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_198);  sub_59 = mul_198 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_60);  div_27 = sub_60 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_212, mul_10);  mul_10 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_212, [0, 1]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_78: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_75, mul_199);  add_75 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_78, mul_201);  mul_201 = None
    clone_31: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_202, memory_format = torch.contiguous_format);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_213: "f32[512, 768]" = torch.ops.aten.view.default(clone_31, [512, 768]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_39: "f32[512, 768]" = torch.ops.aten.mm.default(view_213, permute_107);  permute_107 = None
    mm_40: "f32[768, 768]" = torch.ops.aten.mm.default(permute_108, view_213);  permute_108 = None
    sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
    view_214: "f32[768]" = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_39, [1, 512, 768]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_216: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_215, [1, 512, 12, 64]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_109: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_217: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_109, [12, 512, 64]);  permute_109 = None
    bmm_28: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_110, view_217);  permute_110 = None
    bmm_29: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_217, permute_111);  view_217 = permute_111 = None
    view_218: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, [1, 12, 512, 64]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_79: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_6, view_218);  tangents_6 = view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_219: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, [1, 12, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_21: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_203: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_204: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_219, mul_203);  view_219 = mul_203 = None
    clone_32: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_204, memory_format = torch.contiguous_format);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_205: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_32, alias_23);  clone_32 = None
    sum_74: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [-1], True)
    mul_206: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_23, sum_74);  alias_23 = sum_74 = None
    sub_61: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_14: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_8, sub_61, full_default_13);  slice_8 = sub_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_28: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_14, full_default);  where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_220: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_28, [12, 512, 512]);  div_28 = None
    bmm_30: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_112, view_220);  permute_112 = None
    bmm_31: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_220, permute_113);  view_220 = permute_113 = None
    view_221: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, [1, 12, 64, 512]);  bmm_30 = None
    view_222: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 12, 512, 64]);  bmm_31 = None
    permute_114: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_221, [0, 1, 3, 2]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_80: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_5, permute_114);  tangents_5 = permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_115: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_79, [0, 2, 1, 3]);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_33: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_223: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_33, [1, 512, 768]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_116: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_80, [0, 2, 1, 3]);  add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_34: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
    view_224: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_34, [1, 512, 768]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_117: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_35: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_225: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_35, [1, 512, 768]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_4: "f32[1, 512, 2304]" = torch.ops.aten.cat.default([view_225, view_224, view_223], 2);  view_225 = view_224 = view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_226: "f32[512, 2304]" = torch.ops.aten.view.default(cat_4, [512, 2304]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_41: "f32[512, 768]" = torch.ops.aten.mm.default(view_226, permute_118);  permute_118 = None
    mm_42: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_119, view_226);  permute_119 = None
    sum_75: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[2304]" = torch.ops.aten.view.default(sum_75, [2304]);  sum_75 = None
    view_228: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_41, [1, 512, 768]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_208: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_228, primals_55);  primals_55 = None
    mul_209: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_208, 768)
    sum_76: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True)
    mul_210: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_208, mul_8);  mul_208 = None
    sum_77: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True);  mul_210 = None
    mul_211: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_77);  sum_77 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_209, sum_76);  mul_209 = sum_76 = None
    sub_64: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_211);  sub_63 = mul_211 = None
    mul_212: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_64);  div_29 = sub_64 = None
    mul_213: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_228, mul_8);  mul_8 = None
    sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1]);  mul_213 = None
    sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_228, [0, 1]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_78, mul_212);  add_78 = mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_14, torch.float32);  getitem_14 = None
    mul_214: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_215: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_81, mul_214);  mul_214 = None
    clone_36: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_215, memory_format = torch.contiguous_format);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_229: "f32[512, 768]" = torch.ops.aten.view.default(clone_36, [512, 768]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_43: "f32[512, 3072]" = torch.ops.aten.mm.default(view_229, permute_120);  permute_120 = None
    mm_44: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_121, view_229);  permute_121 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_229, [0], True);  view_229 = None
    view_230: "f32[768]" = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
    view_231: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_43, [1, 512, 3072]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_216: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_231, mul_4);  mul_4 = None
    mul_217: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_231, add_7);  view_231 = add_7 = None
    alias_24: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_218: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_24, alias_24);  alias_24 = None
    sub_65: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_218);  mul_218 = None
    mul_219: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_216, sub_65);  mul_216 = sub_65 = None
    mul_220: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_219, 0.7978845608028654);  mul_219 = None
    mul_221: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_220, 0.044715)
    pow_12: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 2.0);  view_17 = None
    mul_222: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_12, 3.0);  pow_12 = None
    mul_223: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_82: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_220, mul_223);  mul_220 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_224: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_217, 0.5);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_83: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_82, mul_224);  add_82 = mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_232: "f32[512, 3072]" = torch.ops.aten.view.default(add_83, [512, 3072]);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_45: "f32[512, 768]" = torch.ops.aten.mm.default(view_232, permute_122);  permute_122 = None
    mm_46: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_123, view_232);  permute_123 = None
    sum_81: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_232, [0], True);  view_232 = None
    view_233: "f32[3072]" = torch.ops.aten.view.default(sum_81, [3072]);  sum_81 = None
    view_234: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_45, [1, 512, 768]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_234, primals_53);  primals_53 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_226, 768)
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True)
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_226, mul_2);  mul_226 = None
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True);  mul_228 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_83);  sum_83 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_227, sum_82);  mul_227 = sum_82 = None
    sub_68: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_67, mul_229);  sub_67 = mul_229 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_68);  div_30 = sub_68 = None
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_234, mul_2);  mul_2 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_231, [0, 1]);  mul_231 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_234, [0, 1]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_84: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_81, mul_230);  add_81 = mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
    mul_232: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_233: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_84, mul_232);  mul_232 = None
    clone_37: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_233, memory_format = torch.contiguous_format);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_235: "f32[512, 768]" = torch.ops.aten.view.default(clone_37, [512, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_47: "f32[512, 768]" = torch.ops.aten.mm.default(view_235, permute_124);  permute_124 = None
    mm_48: "f32[768, 768]" = torch.ops.aten.mm.default(permute_125, view_235);  permute_125 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_235, [0], True);  view_235 = None
    view_236: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
    view_237: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_47, [1, 512, 768]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_238: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_237, [1, 512, 12, 64]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_126: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_239: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_126, [12, 512, 64]);  permute_126 = None
    bmm_32: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_127, view_239);  permute_127 = None
    bmm_33: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_239, permute_128);  view_239 = permute_128 = None
    view_240: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 12, 512, 64]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_85: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_4, view_240);  tangents_4 = view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_241: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, [1, 12, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_24: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_8, torch.float32);  getitem_8 = None
    mul_234: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_235: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_241, mul_234);  view_241 = mul_234 = None
    clone_38: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_235, memory_format = torch.contiguous_format);  mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_236: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_38, alias_25);  clone_38 = None
    sum_87: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [-1], True)
    mul_237: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_25, sum_87);  alias_25 = sum_87 = None
    sub_69: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_15: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_4, sub_69, full_default_13);  slice_4 = sub_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_31: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_15, full_default);  where_15 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_242: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_31, [12, 512, 512]);  div_31 = None
    bmm_34: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_129, view_242);  permute_129 = None
    bmm_35: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_242, permute_130);  view_242 = permute_130 = None
    view_243: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, [1, 12, 64, 512]);  bmm_34 = None
    view_244: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 12, 512, 64]);  bmm_35 = None
    permute_131: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_243, [0, 1, 3, 2]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_86: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_3, permute_131);  tangents_3 = permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_132: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_85, [0, 2, 1, 3]);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_39: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_245: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_39, [1, 512, 768]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_133: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(add_86, [0, 2, 1, 3]);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_40: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_246: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_40, [1, 512, 768]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_134: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_41: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    view_247: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_41, [1, 512, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_5: "f32[1, 512, 2304]" = torch.ops.aten.cat.default([view_247, view_246, view_245], 2);  view_247 = view_246 = view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_248: "f32[512, 2304]" = torch.ops.aten.view.default(cat_5, [512, 2304]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    mm_49: "f32[512, 768]" = torch.ops.aten.mm.default(view_248, permute_135);  permute_135 = None
    mm_50: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_136, view_248);  permute_136 = None
    sum_88: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
    view_249: "f32[2304]" = torch.ops.aten.view.default(sum_88, [2304]);  sum_88 = None
    view_250: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_49, [1, 512, 768]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    mul_239: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_250, primals_51);  primals_51 = None
    mul_240: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_239, 768)
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True)
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_239, mul);  mul_239 = None
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True);  mul_241 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul, sum_90);  sum_90 = None
    sub_71: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_240, sum_89);  mul_240 = sum_89 = None
    sub_72: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_242);  sub_71 = mul_242 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_72);  div_32 = sub_72 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_250, mul);  mul = None
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 1]);  mul_244 = None
    sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_250, [0, 1]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_87: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_84, mul_243);  add_84 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:851, code: hidden_states = self.drop(hidden_states)
    convert_element_type_25: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_245: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_246: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_87, mul_245);  add_87 = mul_245 = None
    clone_42: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_246, memory_format = torch.contiguous_format);  mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    full_default_25: "b8[1, 512, 1]" = torch.ops.aten.full.default([1, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_16: "f32[1, 512, 768]" = torch.ops.aten.where.self(full_default_25, full_default_13, clone_42);  full_default_25 = None
    full_default_27: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_27, [view_1], where_16, True);  full_default_27 = view_1 = where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_17: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_4, full_default_13, clone_42);  unsqueeze_4 = full_default_13 = clone_42 = None
    full_default_29: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[50257, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_29, [view], where_17, True);  full_default_29 = view = where_17 = None
    return [view_249, mm_50, view_236, mm_48, view_233, mm_46, view_230, mm_44, view_227, mm_42, view_214, mm_40, view_211, mm_38, view_208, mm_36, view_205, mm_34, view_192, mm_32, view_189, mm_30, view_186, mm_28, view_183, mm_26, view_170, mm_24, view_167, mm_22, view_164, mm_20, view_161, mm_18, view_148, mm_16, view_145, mm_14, view_142, mm_12, view_139, mm_10, view_126, mm_8, view_123, mm_6, view_120, mm_4, _unsafe_index_put_1, _unsafe_index_put, sum_91, sum_92, sum_84, sum_85, sum_78, sum_79, sum_71, sum_72, sum_65, sum_66, sum_58, sum_59, sum_52, sum_53, sum_45, sum_46, sum_39, sum_40, sum_32, sum_33, sum_26, sum_27, sum_19, sum_20, sum_13, sum_14, permute_34, None, None, None, None, None, None, None, None]
    