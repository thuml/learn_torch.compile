from __future__ import annotations



def forward(self, primals_4: "f32[768]", primals_8: "f32[768]", primals_14: "f32[768]", primals_16: "f32[768]", primals_22: "f32[768]", primals_24: "f32[768]", primals_30: "f32[768]", primals_32: "f32[768]", primals_38: "f32[768]", primals_40: "f32[768]", primals_46: "f32[768]", primals_48: "f32[768]", primals_54: "f32[768]", primals_56: "f32[768]", primals_62: "f32[768]", primals_64: "f32[768]", primals_70: "f32[768]", primals_72: "f32[768]", primals_78: "f32[768]", primals_80: "f32[768]", primals_86: "f32[768]", primals_88: "f32[768]", primals_94: "f32[768]", primals_96: "f32[768]", primals_102: "f32[768]", primals_108: "f32[768]", primals_114: "i64[1, 512]", primals_115: "i64[1, 512]", expand: "i64[1, 512]", slice_2: "i64[1, 512]", mul: "f32[1, 512, 768]", view: "f32[512, 768]", getitem_3: "b8[1, 512, 768]", mul_2: "f32[1, 512, 768]", view_2: "f32[512, 768]", addmm_1: "f32[512, 3072]", tanh: "f32[1, 512, 3072]", view_4: "f32[512, 3072]", getitem_7: "b8[1, 512, 768]", mul_8: "f32[1, 512, 768]", mul_10: "f32[1, 512, 768]", view_6: "f32[512, 768]", addmm_3: "f32[512, 3072]", tanh_1: "f32[1, 512, 3072]", view_8: "f32[512, 3072]", getitem_13: "b8[1, 512, 768]", mul_16: "f32[1, 512, 768]", mul_18: "f32[1, 512, 768]", view_10: "f32[512, 768]", addmm_5: "f32[512, 3072]", tanh_2: "f32[1, 512, 3072]", view_12: "f32[512, 3072]", getitem_19: "b8[1, 512, 768]", mul_24: "f32[1, 512, 768]", mul_26: "f32[1, 512, 768]", view_14: "f32[512, 768]", addmm_7: "f32[512, 3072]", tanh_3: "f32[1, 512, 3072]", view_16: "f32[512, 3072]", getitem_25: "b8[1, 512, 768]", mul_32: "f32[1, 512, 768]", mul_34: "f32[1, 512, 768]", view_18: "f32[512, 768]", addmm_9: "f32[512, 3072]", tanh_4: "f32[1, 512, 3072]", view_20: "f32[512, 3072]", getitem_31: "b8[1, 512, 768]", mul_40: "f32[1, 512, 768]", mul_42: "f32[1, 512, 768]", view_22: "f32[512, 768]", addmm_11: "f32[512, 3072]", tanh_5: "f32[1, 512, 3072]", view_24: "f32[512, 3072]", getitem_37: "b8[1, 512, 768]", mul_48: "f32[1, 512, 768]", mul_50: "f32[1, 512, 768]", view_26: "f32[512, 768]", addmm_13: "f32[512, 3072]", tanh_6: "f32[1, 512, 3072]", view_28: "f32[512, 3072]", getitem_43: "b8[1, 512, 768]", mul_56: "f32[1, 512, 768]", mul_58: "f32[1, 512, 768]", view_30: "f32[512, 768]", addmm_15: "f32[512, 3072]", tanh_7: "f32[1, 512, 3072]", view_32: "f32[512, 3072]", getitem_49: "b8[1, 512, 768]", mul_64: "f32[1, 512, 768]", mul_66: "f32[1, 512, 768]", view_34: "f32[512, 768]", addmm_17: "f32[512, 3072]", tanh_8: "f32[1, 512, 3072]", view_36: "f32[512, 3072]", getitem_55: "b8[1, 512, 768]", mul_72: "f32[1, 512, 768]", mul_74: "f32[1, 512, 768]", view_38: "f32[512, 768]", addmm_19: "f32[512, 3072]", tanh_9: "f32[1, 512, 3072]", view_40: "f32[512, 3072]", getitem_61: "b8[1, 512, 768]", mul_80: "f32[1, 512, 768]", mul_82: "f32[1, 512, 768]", view_42: "f32[512, 768]", addmm_21: "f32[512, 3072]", tanh_10: "f32[1, 512, 3072]", view_44: "f32[512, 3072]", getitem_67: "b8[1, 512, 768]", mul_88: "f32[1, 512, 768]", mul_90: "f32[1, 512, 768]", view_46: "f32[512, 768]", addmm_23: "f32[512, 3072]", tanh_11: "f32[1, 512, 3072]", view_48: "f32[512, 3072]", getitem_73: "b8[1, 512, 768]", mul_96: "f32[1, 512, 768]", view_50: "f32[512, 768]", addmm_26: "f32[512, 768]", tanh_13: "f32[1, 512, 768]", getitem_77: "f32[1, 512, 1]", rsqrt_25: "f32[1, 512, 1]", view_52: "f32[512, 768]", sub_27: "f32[512, 32000]", convert_element_type_12: "f32[]", permute_28: "f32[32000, 768]", permute_32: "f32[768, 768]", div_3: "f32[1, 512, 1]", permute_36: "f32[768, 3072]", permute_40: "f32[3072, 768]", div_4: "f32[1, 512, 1]", div_5: "f32[1, 512, 1]", permute_44: "f32[768, 3072]", permute_48: "f32[3072, 768]", div_6: "f32[1, 512, 1]", div_7: "f32[1, 512, 1]", permute_52: "f32[768, 3072]", permute_56: "f32[3072, 768]", div_8: "f32[1, 512, 1]", div_9: "f32[1, 512, 1]", permute_60: "f32[768, 3072]", permute_64: "f32[3072, 768]", div_10: "f32[1, 512, 1]", div_11: "f32[1, 512, 1]", permute_68: "f32[768, 3072]", permute_72: "f32[3072, 768]", div_12: "f32[1, 512, 1]", div_13: "f32[1, 512, 1]", permute_76: "f32[768, 3072]", permute_80: "f32[3072, 768]", div_14: "f32[1, 512, 1]", div_15: "f32[1, 512, 1]", permute_84: "f32[768, 3072]", permute_88: "f32[3072, 768]", div_16: "f32[1, 512, 1]", div_17: "f32[1, 512, 1]", permute_92: "f32[768, 3072]", permute_96: "f32[3072, 768]", div_18: "f32[1, 512, 1]", div_19: "f32[1, 512, 1]", permute_100: "f32[768, 3072]", permute_104: "f32[3072, 768]", div_20: "f32[1, 512, 1]", div_21: "f32[1, 512, 1]", permute_108: "f32[768, 3072]", permute_112: "f32[3072, 768]", div_22: "f32[1, 512, 1]", div_23: "f32[1, 512, 1]", permute_116: "f32[768, 3072]", permute_120: "f32[3072, 768]", div_24: "f32[1, 512, 1]", div_25: "f32[1, 512, 1]", permute_124: "f32[768, 3072]", permute_128: "f32[3072, 768]", div_26: "f32[1, 512, 1]", permute_132: "f32[768, 768]", div_27: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512, 32000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_3: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_3, 0.5)
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_7: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 3072]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_7, 0.5)
    add_16: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_11: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 3072]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_15: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    add_32: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 3072]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    add_40: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_23: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 3072]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_27: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
    add_56: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_31: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 3072]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    add_64: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_35: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 3072]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_39: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_39, 0.5)
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_43: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 3072]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    add_88: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_47: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 3072]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:345, code: hidden_states = self.dense(hidden_states)
    view_51: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_98, add_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:347, code: hidden_states = self.LayerNorm(hidden_states)
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_77);  mul_101 = getitem_77 = None
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:775, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_55: "i64[512]" = torch.ops.aten.reshape.default(primals_115, [-1]);  primals_115 = None
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_12);  tangents_1 = convert_element_type_12 = None
    unsqueeze_1: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_55, 1);  view_55 = None
    ne_3: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    where_2: "i64[512, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
    full_default_3: "f32[512, 32000]" = torch.ops.aten.full.default([512, 32000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[512, 32000]" = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
    where_3: "f32[512, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = None
    mul_104: "f32[512, 32000]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    exp_1: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_4: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [1], True)
    mul_105: "f32[512, 32000]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
    sub_28: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    view_56: "f32[1, 512, 32000]" = torch.ops.aten.reshape.default(sub_28, [1, 512, 32000]);  sub_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:775, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    add_104: "f32[1, 512, 32000]" = torch.ops.aten.add.Tensor(tangents_2, view_56);  tangents_2 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:365, code: hidden_states = self.decoder(hidden_states)
    view_57: "f32[512, 32000]" = torch.ops.aten.reshape.default(add_104, [512, 32000]);  add_104 = None
    mm: "f32[512, 768]" = torch.ops.aten.mm.default(view_57, permute_28);  permute_28 = None
    permute_29: "f32[32000, 512]" = torch.ops.aten.permute.default(view_57, [1, 0])
    mm_1: "f32[32000, 768]" = torch.ops.aten.mm.default(permute_29, view_52);  permute_29 = view_52 = None
    permute_30: "f32[768, 32000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_5: "f32[1, 32000]" = torch.ops.aten.sum.dim_IntList(view_57, [0], True);  view_57 = None
    view_58: "f32[32000]" = torch.ops.aten.reshape.default(sum_5, [32000]);  sum_5 = None
    permute_31: "f32[32000, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    view_59: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm, [1, 512, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:347, code: hidden_states = self.LayerNorm(hidden_states)
    mul_107: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_59, primals_108);  primals_108 = None
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, 768)
    sum_6: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_107, [2], True)
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, mul_102);  mul_107 = None
    sum_7: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True);  mul_109 = None
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, sum_7);  sum_7 = None
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_108, sum_6);  mul_108 = sum_6 = None
    sub_31: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_30, mul_110);  sub_30 = mul_110 = None
    div_2: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_31);  div_2 = sub_31 = None
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_59, mul_102);  mul_102 = None
    sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_112, [0, 1]);  mul_112 = None
    sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_59, [0, 1]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_111, mul_98);  mul_98 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_111, add_101);  mul_111 = add_101 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, mul_115);  mul_115 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, sub_32);  mul_113 = sub_32 = None
    mul_117: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_116, 0.7978845608028654);  mul_116 = None
    mul_118: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_117, 0.044715)
    pow_14: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_51, 2.0);  view_51 = None
    mul_119: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
    mul_120: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_118, mul_119);  mul_118 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_105: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_117, mul_120);  mul_117 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_114, 0.5);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_106: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_105, mul_121);  add_105 = mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:345, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 768]" = torch.ops.aten.reshape.default(add_106, [512, 768]);  add_106 = None
    mm_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_60, permute_32);  permute_32 = None
    permute_33: "f32[768, 512]" = torch.ops.aten.permute.default(view_60, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_33, view_50);  permute_33 = view_50 = None
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_60, [0], True);  view_60 = None
    view_61: "f32[768]" = torch.ops.aten.reshape.default(sum_10, [768]);  sum_10 = None
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    view_62: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_123: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, primals_102);  primals_102 = None
    mul_124: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_123, 768)
    sum_11: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True)
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_123, mul_96);  mul_123 = None
    sum_12: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True);  mul_125 = None
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, sum_12);  sum_12 = None
    sub_34: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_124, sum_11);  mul_124 = sum_11 = None
    sub_35: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_34, mul_126);  sub_34 = mul_126 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_35);  div_3 = sub_35 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, mul_96);  mul_96 = None
    sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1]);  mul_128 = None
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_62, [0, 1]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_129: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_130: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_127, mul_129);  mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_130, [512, 768]);  mul_130 = None
    mm_4: "f32[512, 3072]" = torch.ops.aten.mm.default(view_63, permute_36);  permute_36 = None
    permute_37: "f32[768, 512]" = torch.ops.aten.permute.default(view_63, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_37, view_48);  permute_37 = view_48 = None
    permute_38: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_63, [0], True);  view_63 = None
    view_64: "f32[768]" = torch.ops.aten.reshape.default(sum_15, [768]);  sum_15 = None
    permute_39: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_65: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [1, 512, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_131: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, mul_92);  mul_92 = None
    mul_132: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, add_96);  view_65 = add_96 = None
    mul_133: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
    sub_36: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_133);  mul_133 = None
    mul_134: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_131, sub_36);  mul_131 = sub_36 = None
    mul_135: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_134, 0.7978845608028654);  mul_134 = None
    mul_136: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_135, 0.044715)
    pow_15: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 2.0);  view_47 = None
    mul_137: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
    mul_138: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_136, mul_137);  mul_136 = mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_107: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_135, mul_138);  mul_135 = mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_139: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_132, 0.5);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_108: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_107, mul_139);  add_107 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_108, [512, 3072]);  add_108 = None
    mm_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_66, permute_40);  permute_40 = None
    permute_41: "f32[3072, 512]" = torch.ops.aten.permute.default(view_66, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_41, view_46);  permute_41 = view_46 = None
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_16: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_66, [0], True);  view_66 = None
    view_67: "f32[3072]" = torch.ops.aten.reshape.default(sum_16, [3072]);  sum_16 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_68: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_6, [1, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_109: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_127, view_68);  mul_127 = view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_109, primals_96);  primals_96 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_141, 768)
    sum_17: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [2], True)
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_141, mul_90);  mul_141 = None
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True);  mul_143 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_18);  sum_18 = None
    sub_38: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_142, sum_17);  mul_142 = sum_17 = None
    sub_39: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_144);  sub_38 = mul_144 = None
    mul_145: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
    mul_146: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_109, mul_90);  mul_90 = None
    sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_146, [0, 1]);  mul_146 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_109, [0, 1]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_default_5: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_145, 3, 0)
    view_as_complex: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter);  select_scatter = None
    _fft_c2c_12: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex, [1, 2], 0, False);  view_as_complex = None
    view_as_real_12: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_12);  _fft_c2c_12 = None
    select_13: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_12, 3, 0);  view_as_real_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_110: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_145, select_13);  mul_145 = select_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_148: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, primals_94);  primals_94 = None
    mul_149: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_148, 768)
    sum_21: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
    mul_150: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_148, mul_88);  mul_148 = None
    sum_22: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
    mul_151: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_88, sum_22);  sum_22 = None
    sub_41: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_149, sum_21);  mul_149 = sum_21 = None
    sub_42: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_151);  sub_41 = mul_151 = None
    mul_152: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_42);  div_5 = sub_42 = None
    mul_153: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_88);  mul_88 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_152, mul_154);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_69: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_155, [512, 768]);  mul_155 = None
    mm_8: "f32[512, 3072]" = torch.ops.aten.mm.default(view_69, permute_44);  permute_44 = None
    permute_45: "f32[768, 512]" = torch.ops.aten.permute.default(view_69, [1, 0])
    mm_9: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_45, view_44);  permute_45 = view_44 = None
    permute_46: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_25: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_69, [0], True);  view_69 = None
    view_70: "f32[768]" = torch.ops.aten.reshape.default(sum_25, [768]);  sum_25 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    view_71: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_8, [1, 512, 3072]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, mul_84);  mul_84 = None
    mul_157: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, add_88);  view_71 = add_88 = None
    mul_158: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
    sub_43: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_158);  mul_158 = None
    mul_159: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_156, sub_43);  mul_156 = sub_43 = None
    mul_160: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_159, 0.7978845608028654);  mul_159 = None
    mul_161: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_160, 0.044715)
    pow_16: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2.0);  view_43 = None
    mul_162: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
    mul_163: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_111: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_160, mul_163);  mul_160 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_157, 0.5);  mul_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_112: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_111, mul_164);  add_111 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_72: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_112, [512, 3072]);  add_112 = None
    mm_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_72, permute_48);  permute_48 = None
    permute_49: "f32[3072, 512]" = torch.ops.aten.permute.default(view_72, [1, 0])
    mm_11: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_49, view_42);  permute_49 = view_42 = None
    permute_50: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_26: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_72, [0], True);  view_72 = None
    view_73: "f32[3072]" = torch.ops.aten.reshape.default(sum_26, [3072]);  sum_26 = None
    permute_51: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    view_74: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_10, [1, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_152, view_74);  mul_152 = view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_166: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_88);  primals_88 = None
    mul_167: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, 768)
    sum_27: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True)
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, mul_82);  mul_166 = None
    sum_28: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True);  mul_168 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_82, sum_28);  sum_28 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_167, sum_27);  mul_167 = sum_27 = None
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_169);  sub_45 = mul_169 = None
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_46);  div_6 = sub_46 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_82);  mul_82 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1]);  mul_171 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_1: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_170, 3, 0)
    view_as_complex_1: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_1);  select_scatter_1 = None
    _fft_c2c_13: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_1, [1, 2], 0, False);  view_as_complex_1 = None
    view_as_real_13: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_13);  _fft_c2c_13 = None
    select_14: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_13, 3, 0);  view_as_real_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_114: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_170, select_14);  mul_170 = select_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, primals_86);  primals_86 = None
    mul_174: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_173, 768)
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True)
    mul_175: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_173, mul_80);  mul_173 = None
    sum_32: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True);  mul_175 = None
    mul_176: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_32);  sum_32 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_174, sum_31);  mul_174 = sum_31 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_176);  sub_48 = mul_176 = None
    mul_177: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_49);  div_7 = sub_49 = None
    mul_178: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_80);  mul_80 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_178, [0, 1]);  mul_178 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_15: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_179: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_180: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_177, mul_179);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_75: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_180, [512, 768]);  mul_180 = None
    mm_12: "f32[512, 3072]" = torch.ops.aten.mm.default(view_75, permute_52);  permute_52 = None
    permute_53: "f32[768, 512]" = torch.ops.aten.permute.default(view_75, [1, 0])
    mm_13: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_53, view_40);  permute_53 = view_40 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
    view_76: "f32[768]" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    view_77: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_12, [1, 512, 3072]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_181: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, mul_76);  mul_76 = None
    mul_182: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, add_80);  view_77 = add_80 = None
    mul_183: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
    sub_50: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_183);  mul_183 = None
    mul_184: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_181, sub_50);  mul_181 = sub_50 = None
    mul_185: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_184, 0.7978845608028654);  mul_184 = None
    mul_186: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_185, 0.044715)
    pow_17: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_39, 2.0);  view_39 = None
    mul_187: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
    mul_188: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_115: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_185, mul_188);  mul_185 = mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_189: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_182, 0.5);  mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_116: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_115, mul_189);  add_115 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_78: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_116, [512, 3072]);  add_116 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_78, permute_56);  permute_56 = None
    permute_57: "f32[3072, 512]" = torch.ops.aten.permute.default(view_78, [1, 0])
    mm_15: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_57, view_38);  permute_57 = view_38 = None
    permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_36: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_78, [0], True);  view_78 = None
    view_79: "f32[3072]" = torch.ops.aten.reshape.default(sum_36, [3072]);  sum_36 = None
    permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    view_80: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_177, view_80);  mul_177 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_191: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, primals_80);  primals_80 = None
    mul_192: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_191, 768)
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_191, [2], True)
    mul_193: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_191, mul_74);  mul_191 = None
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True);  mul_193 = None
    mul_194: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_38);  sum_38 = None
    sub_52: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_192, sum_37);  mul_192 = sum_37 = None
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_194);  sub_52 = mul_194 = None
    mul_195: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_53);  div_8 = sub_53 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_74);  mul_74 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 1]);  mul_196 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_2: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_195, 3, 0)
    view_as_complex_2: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_2);  select_scatter_2 = None
    _fft_c2c_14: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_2, [1, 2], 0, False);  view_as_complex_2 = None
    view_as_real_14: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_14);  _fft_c2c_14 = None
    select_15: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_14, 3, 0);  view_as_real_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_118: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_195, select_15);  mul_195 = select_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_118, primals_78);  primals_78 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_198, 768)
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [2], True)
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_198, mul_72);  mul_198 = None
    sum_42: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True);  mul_200 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_42);  sum_42 = None
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_199, sum_41);  mul_199 = sum_41 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_201);  sub_55 = mul_201 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_56);  div_9 = sub_56 = None
    mul_203: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_118, mul_72);  mul_72 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 1]);  mul_203 = None
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_118, [0, 1]);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_204: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_205: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_202, mul_204);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_81: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_205, [512, 768]);  mul_205 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_81, permute_60);  permute_60 = None
    permute_61: "f32[768, 512]" = torch.ops.aten.permute.default(view_81, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_61, view_36);  permute_61 = view_36 = None
    permute_62: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_81, [0], True);  view_81 = None
    view_82: "f32[768]" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
    permute_63: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    view_83: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_206: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_83, mul_68);  mul_68 = None
    mul_207: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_83, add_72);  view_83 = add_72 = None
    mul_208: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
    sub_57: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_208);  mul_208 = None
    mul_209: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_206, sub_57);  mul_206 = sub_57 = None
    mul_210: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_209, 0.7978845608028654);  mul_209 = None
    mul_211: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_210, 0.044715)
    pow_18: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 2.0);  view_35 = None
    mul_212: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
    mul_213: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_119: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_210, mul_213);  mul_210 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_214: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_207, 0.5);  mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_120: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_119, mul_214);  add_119 = mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_120, [512, 3072]);  add_120 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_84, permute_64);  permute_64 = None
    permute_65: "f32[3072, 512]" = torch.ops.aten.permute.default(view_84, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_65, view_34);  permute_65 = view_34 = None
    permute_66: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_46: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_84, [0], True);  view_84 = None
    view_85: "f32[3072]" = torch.ops.aten.reshape.default(sum_46, [3072]);  sum_46 = None
    permute_67: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    view_86: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_121: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_202, view_86);  mul_202 = view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_216: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_121, primals_72);  primals_72 = None
    mul_217: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_216, 768)
    sum_47: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_216, [2], True)
    mul_218: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_216, mul_66);  mul_216 = None
    sum_48: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_218, [2], True);  mul_218 = None
    mul_219: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_48);  sum_48 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_217, sum_47);  mul_217 = sum_47 = None
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_219);  sub_59 = mul_219 = None
    mul_220: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_60);  div_10 = sub_60 = None
    mul_221: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_121, mul_66);  mul_66 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 1]);  mul_221 = None
    sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_121, [0, 1]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_3: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_220, 3, 0)
    view_as_complex_3: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_3);  select_scatter_3 = None
    _fft_c2c_15: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_3, [1, 2], 0, False);  view_as_complex_3 = None
    view_as_real_15: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_15);  _fft_c2c_15 = None
    select_16: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_15, 3, 0);  view_as_real_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_220, select_16);  mul_220 = select_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_223: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_70);  primals_70 = None
    mul_224: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_223, 768)
    sum_51: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True)
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_223, mul_64);  mul_223 = None
    sum_52: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_52);  sum_52 = None
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_224, sum_51);  mul_224 = sum_51 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_226);  sub_62 = mul_226 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_63);  div_11 = sub_63 = None
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_64);  mul_64 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 1]);  mul_228 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_227, mul_229);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_87: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_230, [512, 768]);  mul_230 = None
    mm_20: "f32[512, 3072]" = torch.ops.aten.mm.default(view_87, permute_68);  permute_68 = None
    permute_69: "f32[768, 512]" = torch.ops.aten.permute.default(view_87, [1, 0])
    mm_21: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_69, view_32);  permute_69 = view_32 = None
    permute_70: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_87, [0], True);  view_87 = None
    view_88: "f32[768]" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    view_89: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_20, [1, 512, 3072]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_231: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, mul_60);  mul_60 = None
    mul_232: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, add_64);  view_89 = add_64 = None
    mul_233: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
    sub_64: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_233);  mul_233 = None
    mul_234: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_231, sub_64);  mul_231 = sub_64 = None
    mul_235: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_234, 0.7978845608028654);  mul_234 = None
    mul_236: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_235, 0.044715)
    pow_19: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_31, 2.0);  view_31 = None
    mul_237: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
    mul_238: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_123: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_235, mul_238);  mul_235 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_239: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_232, 0.5);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_124: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_123, mul_239);  add_123 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_90: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_124, [512, 3072]);  add_124 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_90, permute_72);  permute_72 = None
    permute_73: "f32[3072, 512]" = torch.ops.aten.permute.default(view_90, [1, 0])
    mm_23: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_73, view_30);  permute_73 = view_30 = None
    permute_74: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_56: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_90, [0], True);  view_90 = None
    view_91: "f32[3072]" = torch.ops.aten.reshape.default(sum_56, [3072]);  sum_56 = None
    permute_75: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    view_92: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_125: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_227, view_92);  mul_227 = view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_64);  primals_64 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_241, 768)
    sum_57: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_241, mul_58);  mul_241 = None
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_58);  sum_58 = None
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_242, sum_57);  mul_242 = sum_57 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_244);  sub_66 = mul_244 = None
    mul_245: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_67);  div_12 = sub_67 = None
    mul_246: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_58);  mul_58 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_4: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_245, 3, 0)
    view_as_complex_4: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_4);  select_scatter_4 = None
    _fft_c2c_16: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_4, [1, 2], 0, False);  view_as_complex_4 = None
    view_as_real_16: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_16);  _fft_c2c_16 = None
    select_17: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_16, 3, 0);  view_as_real_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_126: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_245, select_17);  mul_245 = select_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_248: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, primals_62);  primals_62 = None
    mul_249: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_248, 768)
    sum_61: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
    mul_250: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_248, mul_56);  mul_248 = None
    sum_62: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
    mul_251: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_56, sum_62);  sum_62 = None
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_249, sum_61);  mul_249 = sum_61 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_251);  sub_69 = mul_251 = None
    mul_252: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_70);  div_13 = sub_70 = None
    mul_253: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_56);  mul_56 = None
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_18: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_252, mul_254);  mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_93: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_255, [512, 768]);  mul_255 = None
    mm_24: "f32[512, 3072]" = torch.ops.aten.mm.default(view_93, permute_76);  permute_76 = None
    permute_77: "f32[768, 512]" = torch.ops.aten.permute.default(view_93, [1, 0])
    mm_25: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_77, view_28);  permute_77 = view_28 = None
    permute_78: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_93, [0], True);  view_93 = None
    view_94: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
    permute_79: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    view_95: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_24, [1, 512, 3072]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_256: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_95, mul_52);  mul_52 = None
    mul_257: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_95, add_56);  view_95 = add_56 = None
    mul_258: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
    sub_71: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_258);  mul_258 = None
    mul_259: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_256, sub_71);  mul_256 = sub_71 = None
    mul_260: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_259, 0.7978845608028654);  mul_259 = None
    mul_261: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_260, 0.044715)
    pow_20: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 2.0);  view_27 = None
    mul_262: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
    mul_263: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_127: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_260, mul_263);  mul_260 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_264: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_257, 0.5);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_128: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_127, mul_264);  add_127 = mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_96: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_128, [512, 3072]);  add_128 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_96, permute_80);  permute_80 = None
    permute_81: "f32[3072, 512]" = torch.ops.aten.permute.default(view_96, [1, 0])
    mm_27: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_81, view_26);  permute_81 = view_26 = None
    permute_82: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_66: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_96, [0], True);  view_96 = None
    view_97: "f32[3072]" = torch.ops.aten.reshape.default(sum_66, [3072]);  sum_66 = None
    permute_83: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    view_98: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_252, view_98);  mul_252 = view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_266: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_56);  primals_56 = None
    mul_267: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, 768)
    sum_67: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
    mul_268: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, mul_50);  mul_266 = None
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
    mul_269: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_68);  sum_68 = None
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_267, sum_67);  mul_267 = sum_67 = None
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_269);  sub_73 = mul_269 = None
    mul_270: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_74);  div_14 = sub_74 = None
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_50);  mul_50 = None
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_5: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_270, 3, 0)
    view_as_complex_5: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_5);  select_scatter_5 = None
    _fft_c2c_17: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_5, [1, 2], 0, False);  view_as_complex_5 = None
    view_as_real_17: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_17);  _fft_c2c_17 = None
    select_18: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_17, 3, 0);  view_as_real_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_130: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_270, select_18);  mul_270 = select_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, primals_54);  primals_54 = None
    mul_274: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_273, 768)
    sum_71: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True)
    mul_275: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_273, mul_48);  mul_273 = None
    sum_72: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [2], True);  mul_275 = None
    mul_276: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, sum_72);  sum_72 = None
    sub_76: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_274, sum_71);  mul_274 = sum_71 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_276);  sub_76 = mul_276 = None
    mul_277: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_77);  div_15 = sub_77 = None
    mul_278: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, mul_48);  mul_48 = None
    sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1]);  mul_278 = None
    sum_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_279: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_280: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_277, mul_279);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_99: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_280, [512, 768]);  mul_280 = None
    mm_28: "f32[512, 3072]" = torch.ops.aten.mm.default(view_99, permute_84);  permute_84 = None
    permute_85: "f32[768, 512]" = torch.ops.aten.permute.default(view_99, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_85, view_24);  permute_85 = view_24 = None
    permute_86: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_99, [0], True);  view_99 = None
    view_100: "f32[768]" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    view_101: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_28, [1, 512, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_281: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_101, mul_44);  mul_44 = None
    mul_282: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_101, add_48);  view_101 = add_48 = None
    mul_283: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
    sub_78: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_283);  mul_283 = None
    mul_284: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_281, sub_78);  mul_281 = sub_78 = None
    mul_285: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_284, 0.7978845608028654);  mul_284 = None
    mul_286: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_285, 0.044715)
    pow_21: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 2.0);  view_23 = None
    mul_287: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
    mul_288: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_131: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_285, mul_288);  mul_285 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_289: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_282, 0.5);  mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_132: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_131, mul_289);  add_131 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_102: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_132, [512, 3072]);  add_132 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_102, permute_88);  permute_88 = None
    permute_89: "f32[3072, 512]" = torch.ops.aten.permute.default(view_102, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_89, view_22);  permute_89 = view_22 = None
    permute_90: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_76: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_102, [0], True);  view_102 = None
    view_103: "f32[3072]" = torch.ops.aten.reshape.default(sum_76, [3072]);  sum_76 = None
    permute_91: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    view_104: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_277, view_104);  mul_277 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_291: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, primals_48);  primals_48 = None
    mul_292: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, 768)
    sum_77: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True)
    mul_293: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, mul_42);  mul_291 = None
    sum_78: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
    mul_294: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_78);  sum_78 = None
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_292, sum_77);  mul_292 = sum_77 = None
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_294);  sub_80 = mul_294 = None
    mul_295: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_81);  div_16 = sub_81 = None
    mul_296: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, mul_42);  mul_42 = None
    sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1]);  mul_296 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1]);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_6: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_295, 3, 0)
    view_as_complex_6: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_6);  select_scatter_6 = None
    _fft_c2c_18: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_6, [1, 2], 0, False);  view_as_complex_6 = None
    view_as_real_18: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_18);  _fft_c2c_18 = None
    select_19: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_18, 3, 0);  view_as_real_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_134: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_295, select_19);  mul_295 = select_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_298: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_46);  primals_46 = None
    mul_299: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_298, 768)
    sum_81: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True)
    mul_300: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_298, mul_40);  mul_298 = None
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_82);  sum_82 = None
    sub_83: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_299, sum_81);  mul_299 = sum_81 = None
    sub_84: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_301);  sub_83 = mul_301 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_84);  div_17 = sub_84 = None
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_40);  mul_40 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_304: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_305: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_302, mul_304);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_105: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_305, [512, 768]);  mul_305 = None
    mm_32: "f32[512, 3072]" = torch.ops.aten.mm.default(view_105, permute_92);  permute_92 = None
    permute_93: "f32[768, 512]" = torch.ops.aten.permute.default(view_105, [1, 0])
    mm_33: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_93, view_20);  permute_93 = view_20 = None
    permute_94: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True);  view_105 = None
    view_106: "f32[768]" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
    permute_95: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_32, [1, 512, 3072]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_306: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_36);  mul_36 = None
    mul_307: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, add_40);  view_107 = add_40 = None
    mul_308: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
    sub_85: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_308);  mul_308 = None
    mul_309: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_306, sub_85);  mul_306 = sub_85 = None
    mul_310: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_309, 0.7978845608028654);  mul_309 = None
    mul_311: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_310, 0.044715)
    pow_22: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 2.0);  view_19 = None
    mul_312: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
    mul_313: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_135: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_310, mul_313);  mul_310 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_314: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_307, 0.5);  mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_136: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_135, mul_314);  add_135 = mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_136, [512, 3072]);  add_136 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_108, permute_96);  permute_96 = None
    permute_97: "f32[3072, 512]" = torch.ops.aten.permute.default(view_108, [1, 0])
    mm_35: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_97, view_18);  permute_97 = view_18 = None
    permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_86: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True);  view_108 = None
    view_109: "f32[3072]" = torch.ops.aten.reshape.default(sum_86, [3072]);  sum_86 = None
    permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    view_110: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_137: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_302, view_110);  mul_302 = view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_40);  primals_40 = None
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_316, 768)
    sum_87: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True)
    mul_318: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_316, mul_34);  mul_316 = None
    sum_88: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True);  mul_318 = None
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, sum_88);  sum_88 = None
    sub_87: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_317, sum_87);  mul_317 = sum_87 = None
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_319);  sub_87 = mul_319 = None
    mul_320: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_88);  div_18 = sub_88 = None
    mul_321: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_34);  mul_34 = None
    sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 1]);  mul_321 = None
    sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_7: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_320, 3, 0)
    view_as_complex_7: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_7);  select_scatter_7 = None
    _fft_c2c_19: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_7, [1, 2], 0, False);  view_as_complex_7 = None
    view_as_real_19: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_19);  _fft_c2c_19 = None
    select_20: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_19, 3, 0);  view_as_real_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_138: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_320, select_20);  mul_320 = select_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_323: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, primals_38);  primals_38 = None
    mul_324: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_323, 768)
    sum_91: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True)
    mul_325: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_323, mul_32);  mul_323 = None
    sum_92: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True);  mul_325 = None
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, sum_92);  sum_92 = None
    sub_90: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_324, sum_91);  mul_324 = sum_91 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_326);  sub_90 = mul_326 = None
    mul_327: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_91);  div_19 = sub_91 = None
    mul_328: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_32);  mul_32 = None
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1]);  mul_328 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_21: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_327, mul_329);  mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_111: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_330, [512, 768]);  mul_330 = None
    mm_36: "f32[512, 3072]" = torch.ops.aten.mm.default(view_111, permute_100);  permute_100 = None
    permute_101: "f32[768, 512]" = torch.ops.aten.permute.default(view_111, [1, 0])
    mm_37: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_101, view_16);  permute_101 = view_16 = None
    permute_102: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_111, [0], True);  view_111 = None
    view_112: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
    permute_103: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    view_113: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_36, [1, 512, 3072]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_331: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_113, mul_28);  mul_28 = None
    mul_332: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_113, add_32);  view_113 = add_32 = None
    mul_333: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
    sub_92: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_333);  mul_333 = None
    mul_334: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_331, sub_92);  mul_331 = sub_92 = None
    mul_335: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_334, 0.7978845608028654);  mul_334 = None
    mul_336: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_335, 0.044715)
    pow_23: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_15, 2.0);  view_15 = None
    mul_337: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
    mul_338: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_139: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_335, mul_338);  mul_335 = mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_339: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_332, 0.5);  mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_140: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_139, mul_339);  add_139 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_114: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_140, [512, 3072]);  add_140 = None
    mm_38: "f32[512, 768]" = torch.ops.aten.mm.default(view_114, permute_104);  permute_104 = None
    permute_105: "f32[3072, 512]" = torch.ops.aten.permute.default(view_114, [1, 0])
    mm_39: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_105, view_14);  permute_105 = view_14 = None
    permute_106: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
    view_115: "f32[3072]" = torch.ops.aten.reshape.default(sum_96, [3072]);  sum_96 = None
    permute_107: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    view_116: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_38, [1, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_327, view_116);  mul_327 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, primals_32);  primals_32 = None
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, 768)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, mul_26);  mul_341 = None
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, sum_98);  sum_98 = None
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_342, sum_97);  mul_342 = sum_97 = None
    sub_95: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_94, mul_344);  sub_94 = mul_344 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_95);  div_20 = sub_95 = None
    mul_346: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_26);  mul_26 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_8: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_345, 3, 0)
    view_as_complex_8: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_8);  select_scatter_8 = None
    _fft_c2c_20: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_8, [1, 2], 0, False);  view_as_complex_8 = None
    view_as_real_20: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_20);  _fft_c2c_20 = None
    select_21: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_20, 3, 0);  view_as_real_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_142: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_345, select_21);  mul_345 = select_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_348: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, primals_30);  primals_30 = None
    mul_349: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_348, 768)
    sum_101: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True)
    mul_350: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_348, mul_24);  mul_348 = None
    sum_102: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True);  mul_350 = None
    mul_351: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_102);  sum_102 = None
    sub_97: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_349, sum_101);  mul_349 = sum_101 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_351);  sub_97 = mul_351 = None
    mul_352: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_98);  div_21 = sub_98 = None
    mul_353: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, mul_24);  mul_24 = None
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 1]);  mul_353 = None
    sum_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
    mul_354: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_355: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_352, mul_354);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_117: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_355, [512, 768]);  mul_355 = None
    mm_40: "f32[512, 3072]" = torch.ops.aten.mm.default(view_117, permute_108);  permute_108 = None
    permute_109: "f32[768, 512]" = torch.ops.aten.permute.default(view_117, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_109, view_12);  permute_109 = view_12 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
    view_118: "f32[768]" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
    permute_111: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    view_119: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_40, [1, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_356: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_119, mul_20);  mul_20 = None
    mul_357: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_119, add_24);  view_119 = add_24 = None
    mul_358: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
    sub_99: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_358);  mul_358 = None
    mul_359: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_356, sub_99);  mul_356 = sub_99 = None
    mul_360: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_359, 0.7978845608028654);  mul_359 = None
    mul_361: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_360, 0.044715)
    pow_24: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 2.0);  view_11 = None
    mul_362: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
    mul_363: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_143: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_360, mul_363);  mul_360 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_364: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_357, 0.5);  mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_144: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_143, mul_364);  add_143 = mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_120: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_144, [512, 3072]);  add_144 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_120, permute_112);  permute_112 = None
    permute_113: "f32[3072, 512]" = torch.ops.aten.permute.default(view_120, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_113, view_10);  permute_113 = view_10 = None
    permute_114: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_106: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True);  view_120 = None
    view_121: "f32[3072]" = torch.ops.aten.reshape.default(sum_106, [3072]);  sum_106 = None
    permute_115: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    view_122: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_145: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_352, view_122);  mul_352 = view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_366: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, primals_24);  primals_24 = None
    mul_367: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_366, 768)
    sum_107: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True)
    mul_368: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_366, mul_18);  mul_366 = None
    sum_108: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [2], True);  mul_368 = None
    mul_369: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_108);  sum_108 = None
    sub_101: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_367, sum_107);  mul_367 = sum_107 = None
    sub_102: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_369);  sub_101 = mul_369 = None
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_102);  div_22 = sub_102 = None
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_18);  mul_18 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 1]);  mul_371 = None
    sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_9: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_370, 3, 0)
    view_as_complex_9: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_9);  select_scatter_9 = None
    _fft_c2c_21: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_9, [1, 2], 0, False);  view_as_complex_9 = None
    view_as_real_21: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_21);  _fft_c2c_21 = None
    select_22: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_21, 3, 0);  view_as_real_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_370, select_22);  mul_370 = select_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_22);  primals_22 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_373, 768)
    sum_111: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True)
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_373, mul_16);  mul_373 = None
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True);  mul_375 = None
    mul_376: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_112);  sum_112 = None
    sub_104: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_374, sum_111);  mul_374 = sum_111 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_376);  sub_104 = mul_376 = None
    mul_377: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_105);  div_23 = sub_105 = None
    mul_378: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_16);  mul_16 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_378, [0, 1]);  mul_378 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
    mul_379: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_380: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_377, mul_379);  mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_123: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_380, [512, 768]);  mul_380 = None
    mm_44: "f32[512, 3072]" = torch.ops.aten.mm.default(view_123, permute_116);  permute_116 = None
    permute_117: "f32[768, 512]" = torch.ops.aten.permute.default(view_123, [1, 0])
    mm_45: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_117, view_8);  permute_117 = view_8 = None
    permute_118: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True);  view_123 = None
    view_124: "f32[768]" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    view_125: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_44, [1, 512, 3072]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_381: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, mul_12);  mul_12 = None
    mul_382: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, add_16);  view_125 = add_16 = None
    mul_383: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
    sub_106: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_383);  mul_383 = None
    mul_384: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_381, sub_106);  mul_381 = sub_106 = None
    mul_385: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_384, 0.7978845608028654);  mul_384 = None
    mul_386: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_385, 0.044715)
    pow_25: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 2.0);  view_7 = None
    mul_387: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
    mul_388: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_147: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_385, mul_388);  mul_385 = mul_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_389: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_382, 0.5);  mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_148: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_147, mul_389);  add_147 = mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_148, [512, 3072]);  add_148 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_126, permute_120);  permute_120 = None
    permute_121: "f32[3072, 512]" = torch.ops.aten.permute.default(view_126, [1, 0])
    mm_47: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_121, view_6);  permute_121 = view_6 = None
    permute_122: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_116: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
    view_127: "f32[3072]" = torch.ops.aten.reshape.default(sum_116, [3072]);  sum_116 = None
    permute_123: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    view_128: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_377, view_128);  mul_377 = view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_391: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_16);  primals_16 = None
    mul_392: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_391, 768)
    sum_117: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [2], True)
    mul_393: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_391, mul_10);  mul_391 = None
    sum_118: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True);  mul_393 = None
    mul_394: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_118);  sum_118 = None
    sub_108: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_392, sum_117);  mul_392 = sum_117 = None
    sub_109: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_394);  sub_108 = mul_394 = None
    mul_395: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_109);  div_24 = sub_109 = None
    mul_396: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_10);  mul_10 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 1]);  mul_396 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_10: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_395, 3, 0)
    view_as_complex_10: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_10);  select_scatter_10 = None
    _fft_c2c_22: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_10, [1, 2], 0, False);  view_as_complex_10 = None
    view_as_real_22: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_22);  _fft_c2c_22 = None
    select_23: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_22, 3, 0);  view_as_real_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_150: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_395, select_23);  mul_395 = select_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_398: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, primals_14);  primals_14 = None
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, 768)
    sum_121: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True)
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, mul_8);  mul_398 = None
    sum_122: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_122);  sum_122 = None
    sub_111: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_399, sum_121);  mul_399 = sum_121 = None
    sub_112: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_401);  sub_111 = mul_401 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_112);  div_25 = sub_112 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_8);  mul_8 = None
    sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_24: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_402, mul_404);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_405, [512, 768]);  mul_405 = None
    mm_48: "f32[512, 3072]" = torch.ops.aten.mm.default(view_129, permute_124);  permute_124 = None
    permute_125: "f32[768, 512]" = torch.ops.aten.permute.default(view_129, [1, 0])
    mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_125, view_4);  permute_125 = view_4 = None
    permute_126: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
    view_130: "f32[768]" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
    permute_127: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    view_131: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_48, [1, 512, 3072]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_406: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, mul_4);  mul_4 = None
    mul_407: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, add_8);  view_131 = add_8 = None
    mul_408: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
    sub_113: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_408);  mul_408 = None
    mul_409: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_406, sub_113);  mul_406 = sub_113 = None
    mul_410: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_409, 0.7978845608028654);  mul_409 = None
    mul_411: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_410, 0.044715)
    pow_26: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 2.0);  view_3 = None
    mul_412: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
    mul_413: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_151: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_410, mul_413);  mul_410 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_414: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_407, 0.5);  mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_152: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_151, mul_414);  add_151 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_132: "f32[512, 3072]" = torch.ops.aten.reshape.default(add_152, [512, 3072]);  add_152 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_132, permute_128);  permute_128 = None
    permute_129: "f32[3072, 512]" = torch.ops.aten.permute.default(view_132, [1, 0])
    mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_129, view_2);  permute_129 = view_2 = None
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_126: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_132, [0], True);  view_132 = None
    view_133: "f32[3072]" = torch.ops.aten.reshape.default(sum_126, [3072]);  sum_126 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    view_134: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_402, view_134);  mul_402 = view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, primals_8);  primals_8 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, 768)
    sum_127: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, mul_2);  mul_416 = None
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
    mul_419: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_128);  sum_128 = None
    sub_115: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_417, sum_127);  mul_417 = sum_127 = None
    sub_116: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_419);  sub_115 = mul_419 = None
    mul_420: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_116);  div_26 = sub_116 = None
    mul_421: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_2);  mul_2 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    select_scatter_11: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default_5, mul_420, 3, 0);  full_default_5 = None
    view_as_complex_11: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_11);  select_scatter_11 = None
    _fft_c2c_23: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_11, [1, 2], 0, False);  view_as_complex_11 = None
    view_as_real_23: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_23);  _fft_c2c_23 = None
    select_24: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_23, 3, 0);  view_as_real_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_154: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_420, select_24);  mul_420 = select_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:157, code: embeddings = self.dropout(embeddings)
    convert_element_type_25: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_422: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_423: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_154, mul_422);  add_154 = mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:156, code: embeddings = self.projection(embeddings)
    view_135: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_423, [512, 768]);  mul_423 = None
    mm_52: "f32[512, 768]" = torch.ops.aten.mm.default(view_135, permute_132);  permute_132 = None
    permute_133: "f32[768, 512]" = torch.ops.aten.permute.default(view_135, [1, 0])
    mm_53: "f32[768, 768]" = torch.ops.aten.mm.default(permute_133, view);  permute_133 = view = None
    permute_134: "f32[768, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
    view_136: "f32[768]" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None
    permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    view_137: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_52, [1, 512, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:155, code: embeddings = self.LayerNorm(embeddings)
    mul_425: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_137, primals_4);  primals_4 = None
    mul_426: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_425, 768)
    sum_132: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
    mul_427: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_425, mul);  mul_425 = None
    sum_133: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
    mul_428: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul, sum_133);  sum_133 = None
    sub_118: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_426, sum_132);  mul_426 = sum_132 = None
    sub_119: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_428);  sub_118 = mul_428 = None
    mul_429: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_119);  div_27 = sub_119 = None
    mul_430: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_137, mul);  mul = None
    sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:153, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_2, -1)
    unsqueeze_2: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_2, full_default_1, mul_429);  unsqueeze_2 = None
    full_default_18: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_18, [slice_2], where_4, True);  full_default_18 = slice_2 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:149, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_3: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_5: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_3, full_default_1, mul_429);  unsqueeze_3 = None
    full_default_20: "f32[4, 768]" = torch.ops.aten.full.default([4, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[4, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_20, [expand], where_5, True);  full_default_20 = expand = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:148, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_114, 3)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_6: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_4, full_default_1, mul_429);  unsqueeze_4 = full_default_1 = mul_429 = None
    full_default_22: "f32[32000, 768]" = torch.ops.aten.full.default([32000, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[32000, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_22, [primals_114], where_6, True);  full_default_22 = primals_114 = where_6 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_134, sum_135, permute_135, view_136, sum_129, sum_130, permute_131, view_133, permute_127, view_130, sum_123, sum_124, sum_119, sum_120, permute_123, view_127, permute_119, view_124, sum_113, sum_114, sum_109, sum_110, permute_115, view_121, permute_111, view_118, sum_103, sum_104, sum_99, sum_100, permute_107, view_115, permute_103, view_112, sum_93, sum_94, sum_89, sum_90, permute_99, view_109, permute_95, view_106, sum_83, sum_84, sum_79, sum_80, permute_91, view_103, permute_87, view_100, sum_73, sum_74, sum_69, sum_70, permute_83, view_97, permute_79, view_94, sum_63, sum_64, sum_59, sum_60, permute_75, view_91, permute_71, view_88, sum_53, sum_54, sum_49, sum_50, permute_67, view_85, permute_63, view_82, sum_43, sum_44, sum_39, sum_40, permute_59, view_79, permute_55, view_76, sum_33, sum_34, sum_29, sum_30, permute_51, view_73, permute_47, view_70, sum_23, sum_24, sum_19, sum_20, permute_43, view_67, permute_39, view_64, sum_13, sum_14, None, None, permute_35, view_61, sum_8, sum_9, permute_31, view_58, None, None, None, None]
    