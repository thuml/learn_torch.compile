from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[2304]"; primals_2: "f32[768, 2304]"; primals_3: "f32[768]"; primals_4: "f32[768, 768]"; primals_5: "f32[3072]"; primals_6: "f32[768, 3072]"; primals_7: "f32[768]"; primals_8: "f32[3072, 768]"; primals_9: "f32[2304]"; primals_10: "f32[768, 2304]"; primals_11: "f32[768]"; primals_12: "f32[768, 768]"; primals_13: "f32[3072]"; primals_14: "f32[768, 3072]"; primals_15: "f32[768]"; primals_16: "f32[3072, 768]"; primals_17: "f32[2304]"; primals_18: "f32[768, 2304]"; primals_19: "f32[768]"; primals_20: "f32[768, 768]"; primals_21: "f32[3072]"; primals_22: "f32[768, 3072]"; primals_23: "f32[768]"; primals_24: "f32[3072, 768]"; primals_25: "f32[2304]"; primals_26: "f32[768, 2304]"; primals_27: "f32[768]"; primals_28: "f32[768, 768]"; primals_29: "f32[3072]"; primals_30: "f32[768, 3072]"; primals_31: "f32[768]"; primals_32: "f32[3072, 768]"; primals_33: "f32[2304]"; primals_34: "f32[768, 2304]"; primals_35: "f32[768]"; primals_36: "f32[768, 768]"; primals_37: "f32[3072]"; primals_38: "f32[768, 3072]"; primals_39: "f32[768]"; primals_40: "f32[3072, 768]"; primals_41: "f32[2304]"; primals_42: "f32[768, 2304]"; primals_43: "f32[768]"; primals_44: "f32[768, 768]"; primals_45: "f32[3072]"; primals_46: "f32[768, 3072]"; primals_47: "f32[768]"; primals_48: "f32[3072, 768]"; primals_49: "f32[50257, 768]"; primals_50: "f32[1024, 768]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[768]"; primals_56: "f32[768]"; primals_57: "f32[768]"; primals_58: "f32[768]"; primals_59: "f32[768]"; primals_60: "f32[768]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[768]"; primals_64: "f32[768]"; primals_65: "f32[768]"; primals_66: "f32[768]"; primals_67: "f32[768]"; primals_68: "f32[768]"; primals_69: "f32[768]"; primals_70: "f32[768]"; primals_71: "f32[768]"; primals_72: "f32[768]"; primals_73: "f32[768]"; primals_74: "f32[768]"; primals_75: "f32[768]"; primals_76: "f32[768]"; primals_77: "f32[50257, 768]"; primals_78: "b8[1, 1, 1024, 1024]"; primals_79: "b8[1, 1, 1024, 1024]"; primals_80: "b8[1, 1, 1024, 1024]"; primals_81: "b8[1, 1, 1024, 1024]"; primals_82: "b8[1, 1, 1024, 1024]"; primals_83: "b8[1, 1, 1024, 1024]"; primals_84: "i64[1, 512]"; primals_85: "i64[1, 512]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512, 50257]"; tangents_3: "f32[1, 12, 512, 64]"; tangents_4: "f32[1, 12, 512, 64]"; tangents_5: "f32[1, 12, 512, 64]"; tangents_6: "f32[1, 12, 512, 64]"; tangents_7: "f32[1, 12, 512, 64]"; tangents_8: "f32[1, 12, 512, 64]"; tangents_9: "f32[1, 12, 512, 64]"; tangents_10: "f32[1, 12, 512, 64]"; tangents_11: "f32[1, 12, 512, 64]"; tangents_12: "f32[1, 12, 512, 64]"; tangents_13: "f32[1, 12, 512, 64]"; tangents_14: "f32[1, 12, 512, 64]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:781, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 512]" = torch.ops.aten.view.default(primals_84, [-1, 512]);  primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:802, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:803, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 512]" = torch.ops.aten.view.default(unsqueeze, [-1, 512]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_49, view);  primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_50, view_1);  primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:845, code: hidden_states = inputs_embeds + position_embeds
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:851, code: hidden_states = self.drop(hidden_states)
    native_dropout = torch.ops.aten.native_dropout.default(add, 0.1, True);  add = None
    getitem: "f32[1, 512, 768]" = native_dropout[0]
    getitem_1: "b8[1, 512, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(getitem, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean[0]
    getitem_3: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(getitem, getitem_3)
    mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul, primals_51);  mul = None
    add_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_52);  mul_1 = primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_2: "f32[512, 768]" = torch.ops.aten.view.default(add_2, [-1, 768]);  add_2 = None
    addmm: "f32[512, 2304]" = torch.ops.aten.addmm.default(primals_1, view_2, primals_2);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_3: "f32[1, 512, 2304]" = torch.ops.aten.view.default(addmm, [1, 512, 2304]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [768, 768, 768], 2);  view_3 = None
    getitem_4: "f32[1, 512, 768]" = split_with_sizes[0]
    getitem_5: "f32[1, 512, 768]" = split_with_sizes[1]
    getitem_6: "f32[1, 512, 768]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_4: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_4, [1, 512, 12, 64]);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_5: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_5, [1, 512, 12, 64]);  getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_6: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_6, [1, 512, 12, 64]);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_2: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_3: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
    expand: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute, [1, 12, 512, 64]);  permute = None
    view_7: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand, [12, 512, 64]);  expand = None
    expand_1: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_3, [1, 12, 64, 512]);  permute_3 = None
    view_8: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_1, [12, 64, 512]);  expand_1 = None
    bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_7, view_8)
    view_9: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm, [1, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    div: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_9, full);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_78, 0, 0, 9223372036854775807);  primals_78 = None
    slice_2: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    slice_3: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 512);  slice_2 = None
    slice_4: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 512);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    device_put: "f32[]" = torch.ops.prims.device_put.default(full_1, device(type='cuda', index=0));  full_1 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(device_put, torch.float32);  device_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_4, div, convert_element_type);  div = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_1, 0.1, True);  div_1 = None
    getitem_7: "f32[1, 12, 512, 512]" = native_dropout_1[0]
    getitem_8: "b8[1, 12, 512, 512]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_2: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_7, [1, 12, 512, 512]);  getitem_7 = None
    view_10: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_2, [12, 512, 512]);  expand_2 = None
    expand_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_2, [1, 12, 512, 64])
    view_11: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_3, [12, 512, 64]);  expand_3 = None
    bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_10, view_11)
    view_12: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_4: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    clone: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_13: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone, [1, 512, 768]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_14: "f32[512, 768]" = torch.ops.aten.view.default(view_13, [-1, 768]);  view_13 = None
    addmm_1: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_3, view_14, primals_4);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_15: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_1, [1, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_15, 0.1, True);  view_15 = None
    getitem_9: "f32[1, 512, 768]" = native_dropout_2[0]
    getitem_10: "b8[1, 512, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_9, getitem);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_11: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_12: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_12)
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_53);  mul_2 = None
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_54);  mul_3 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_16: "f32[512, 768]" = torch.ops.aten.view.default(add_5, [-1, 768]);  add_5 = None
    addmm_2: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_5, view_16, primals_6);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_17: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_2, [1, 512, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    pow_1: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_17, mul_5);  mul_5 = None
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    alias_1: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh)
    add_7: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_18: "f32[512, 3072]" = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_7, view_18, primals_8);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_19: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_3, [1, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_19, 0.1, True);  view_19 = None
    getitem_13: "f32[1, 512, 768]" = native_dropout_3[0]
    getitem_14: "b8[1, 512, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_8: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_3, getitem_13);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_15: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_16: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_15, 1e-05);  getitem_15 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_16)
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_55);  mul_8 = None
    add_10: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_56);  mul_9 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_20: "f32[512, 768]" = torch.ops.aten.view.default(add_10, [-1, 768]);  add_10 = None
    addmm_4: "f32[512, 2304]" = torch.ops.aten.addmm.default(primals_9, view_20, primals_10);  primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_21: "f32[1, 512, 2304]" = torch.ops.aten.view.default(addmm_4, [1, 512, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_21, [768, 768, 768], 2);  view_21 = None
    getitem_17: "f32[1, 512, 768]" = split_with_sizes_1[0]
    getitem_18: "f32[1, 512, 768]" = split_with_sizes_1[1]
    getitem_19: "f32[1, 512, 768]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_22: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_17, [1, 512, 12, 64]);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_5: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_23: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_18, [1, 512, 12, 64]);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_6: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_24: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_19, [1, 512, 12, 64]);  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_7: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_8: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2])
    expand_4: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 512, 64]);  permute_5 = None
    view_25: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_4, [12, 512, 64]);  expand_4 = None
    expand_5: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_8, [1, 12, 64, 512]);  permute_8 = None
    view_26: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_5, [12, 64, 512]);  expand_5 = None
    bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_25, view_26)
    view_27: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_2: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_27, full_2);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_5: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_79, 0, 0, 9223372036854775807);  primals_79 = None
    slice_6: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    slice_7: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 512);  slice_6 = None
    slice_8: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 512);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    device_put_1: "f32[]" = torch.ops.prims.device_put.default(full_3, device(type='cuda', index=0));  full_3 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(device_put_1, torch.float32);  device_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_1: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_8, div_2, convert_element_type_1);  div_2 = convert_element_type_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_2: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_4 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_20: "f32[1, 12, 512, 512]" = native_dropout_4[0]
    getitem_21: "b8[1, 12, 512, 512]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_20, [1, 12, 512, 512]);  getitem_20 = None
    view_28: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_6, [12, 512, 512]);  expand_6 = None
    expand_7: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_7, [1, 12, 512, 64])
    view_29: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_7, [12, 512, 64]);  expand_7 = None
    bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_28, view_29)
    view_30: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_9: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_1: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_31: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_1, [1, 512, 768]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_32: "f32[512, 768]" = torch.ops.aten.view.default(view_31, [-1, 768]);  view_31 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_11, view_32, primals_12);  primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_33: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_5, [1, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_33, 0.1, True);  view_33 = None
    getitem_22: "f32[1, 512, 768]" = native_dropout_5[0]
    getitem_23: "b8[1, 512, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_22, add_8);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_25: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_25)
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_57);  mul_10 = None
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_58);  mul_11 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_34: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [-1, 768]);  add_13 = None
    addmm_6: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_13, view_34, primals_14);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_35: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_6, [1, 512, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_2: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_13);  mul_13 = None
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    alias_3: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_1)
    add_15: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_36: "f32[512, 3072]" = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
    addmm_7: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_15, view_36, primals_16);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_37: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_7, [1, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_37, 0.1, True);  view_37 = None
    getitem_26: "f32[1, 512, 768]" = native_dropout_6[0]
    getitem_27: "b8[1, 512, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_16: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_11, getitem_26);  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_17: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_29)
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_59);  mul_16 = None
    add_18: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_60);  mul_17 = primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_38: "f32[512, 768]" = torch.ops.aten.view.default(add_18, [-1, 768]);  add_18 = None
    addmm_8: "f32[512, 2304]" = torch.ops.aten.addmm.default(primals_17, view_38, primals_18);  primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_39: "f32[1, 512, 2304]" = torch.ops.aten.view.default(addmm_8, [1, 512, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_39, [768, 768, 768], 2);  view_39 = None
    getitem_30: "f32[1, 512, 768]" = split_with_sizes_2[0]
    getitem_31: "f32[1, 512, 768]" = split_with_sizes_2[1]
    getitem_32: "f32[1, 512, 768]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_40: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_30, [1, 512, 12, 64]);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_10: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_41: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_31, [1, 512, 12, 64]);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_11: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_42: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_32, [1, 512, 12, 64]);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_12: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_13: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_11, [0, 1, 3, 2])
    expand_8: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_10, [1, 12, 512, 64]);  permute_10 = None
    view_43: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_8, [12, 512, 64]);  expand_8 = None
    expand_9: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_13, [1, 12, 64, 512]);  permute_13 = None
    view_44: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_9, [12, 64, 512]);  expand_9 = None
    bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_43, view_44)
    view_45: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_4: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    div_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_45, full_4);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_9: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_80, 0, 0, 9223372036854775807);  primals_80 = None
    slice_10: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    slice_11: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_10, 2, 0, 512);  slice_10 = None
    slice_12: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 512);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    device_put_2: "f32[]" = torch.ops.prims.device_put.default(full_5, device(type='cuda', index=0));  full_5 = None
    convert_element_type_2: "f32[]" = torch.ops.prims.convert_element_type.default(device_put_2, torch.float32);  device_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_2: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_12, div_4, convert_element_type_2);  div_4 = convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_4: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_7 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_33: "f32[1, 12, 512, 512]" = native_dropout_7[0]
    getitem_34: "b8[1, 12, 512, 512]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_10: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_33, [1, 12, 512, 512]);  getitem_33 = None
    view_46: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_10, [12, 512, 512]);  expand_10 = None
    expand_11: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_12, [1, 12, 512, 64])
    view_47: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_11, [12, 512, 64]);  expand_11 = None
    bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_46, view_47)
    view_48: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_14: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_49: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_2, [1, 512, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_50: "f32[512, 768]" = torch.ops.aten.view.default(view_49, [-1, 768]);  view_49 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_19, view_50, primals_20);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_51: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_9, [1, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_51, 0.1, True);  view_51 = None
    getitem_35: "f32[1, 512, 768]" = native_dropout_8[0]
    getitem_36: "b8[1, 512, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_35, add_16);  getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_37: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_38: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_38)
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_61);  mul_18 = None
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_62);  mul_19 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_52: "f32[512, 768]" = torch.ops.aten.view.default(add_21, [-1, 768]);  add_21 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_21, view_52, primals_22);  primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_53: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
    pow_3: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_53, mul_21);  mul_21 = None
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    alias_5: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_2)
    add_23: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_54: "f32[512, 3072]" = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_23, view_54, primals_24);  primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_55: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_55, 0.1, True);  view_55 = None
    getitem_39: "f32[1, 512, 768]" = native_dropout_9[0]
    getitem_40: "b8[1, 512, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_24: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_19, getitem_39);  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_41: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_42: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_25: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_42)
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_63);  mul_24 = None
    add_26: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_64);  mul_25 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_56: "f32[512, 768]" = torch.ops.aten.view.default(add_26, [-1, 768]);  add_26 = None
    addmm_12: "f32[512, 2304]" = torch.ops.aten.addmm.default(primals_25, view_56, primals_26);  primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_57: "f32[1, 512, 2304]" = torch.ops.aten.view.default(addmm_12, [1, 512, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_57, [768, 768, 768], 2);  view_57 = None
    getitem_43: "f32[1, 512, 768]" = split_with_sizes_3[0]
    getitem_44: "f32[1, 512, 768]" = split_with_sizes_3[1]
    getitem_45: "f32[1, 512, 768]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_58: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_43, [1, 512, 12, 64]);  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_15: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_59: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_44, [1, 512, 12, 64]);  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_16: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_60: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_45, [1, 512, 12, 64]);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_17: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_18: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2])
    expand_12: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_15, [1, 12, 512, 64]);  permute_15 = None
    view_61: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_12, [12, 512, 64]);  expand_12 = None
    expand_13: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_18, [1, 12, 64, 512]);  permute_18 = None
    view_62: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_13, [12, 64, 512]);  expand_13 = None
    bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_61, view_62)
    view_63: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_6: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_63, full_6);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_13: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_81, 0, 0, 9223372036854775807);  primals_81 = None
    slice_14: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    slice_15: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 512);  slice_14 = None
    slice_16: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 512);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    device_put_3: "f32[]" = torch.ops.prims.device_put.default(full_7, device(type='cuda', index=0));  full_7 = None
    convert_element_type_3: "f32[]" = torch.ops.prims.convert_element_type.default(device_put_3, torch.float32);  device_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_3: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_16, div_6, convert_element_type_3);  div_6 = convert_element_type_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_10 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_46: "f32[1, 12, 512, 512]" = native_dropout_10[0]
    getitem_47: "b8[1, 12, 512, 512]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_46, [1, 12, 512, 512]);  getitem_46 = None
    view_64: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_14, [12, 512, 512]);  expand_14 = None
    expand_15: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_17, [1, 12, 512, 64])
    view_65: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_15, [12, 512, 64]);  expand_15 = None
    bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_64, view_65)
    view_66: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    clone_3: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_67: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_3, [1, 512, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_68: "f32[512, 768]" = torch.ops.aten.view.default(view_67, [-1, 768]);  view_67 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_27, view_68, primals_28);  primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_69: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_13, [1, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_69, 0.1, True);  view_69 = None
    getitem_48: "f32[1, 512, 768]" = native_dropout_11[0]
    getitem_49: "b8[1, 512, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_48, add_24);  getitem_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_51: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_51)
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_27: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_65);  mul_26 = None
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_66);  mul_27 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_70: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [-1, 768]);  add_29 = None
    addmm_14: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_29, view_70, primals_30);  primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_71: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_14, [1, 512, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    pow_4: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
    mul_29: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_71, mul_29);  mul_29 = None
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    alias_7: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_3)
    add_31: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_72: "f32[512, 3072]" = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_31, view_72, primals_32);  primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_73: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_15, [1, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_73, 0.1, True);  view_73 = None
    getitem_52: "f32[1, 512, 768]" = native_dropout_12[0]
    getitem_53: "b8[1, 512, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_32: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_27, getitem_52);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_55: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_33: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_55)
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_67);  mul_32 = None
    add_34: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_68);  mul_33 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_74: "f32[512, 768]" = torch.ops.aten.view.default(add_34, [-1, 768]);  add_34 = None
    addmm_16: "f32[512, 2304]" = torch.ops.aten.addmm.default(primals_33, view_74, primals_34);  primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_75: "f32[1, 512, 2304]" = torch.ops.aten.view.default(addmm_16, [1, 512, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_75, [768, 768, 768], 2);  view_75 = None
    getitem_56: "f32[1, 512, 768]" = split_with_sizes_4[0]
    getitem_57: "f32[1, 512, 768]" = split_with_sizes_4[1]
    getitem_58: "f32[1, 512, 768]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_76: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_56, [1, 512, 12, 64]);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_20: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_77: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_57, [1, 512, 12, 64]);  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_21: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_78: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_58, [1, 512, 12, 64]);  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_22: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_23: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_21, [0, 1, 3, 2])
    expand_16: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_20, [1, 12, 512, 64]);  permute_20 = None
    view_79: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_16, [12, 512, 64]);  expand_16 = None
    expand_17: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_23, [1, 12, 64, 512]);  permute_23 = None
    view_80: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_17, [12, 64, 512]);  expand_17 = None
    bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_79, view_80)
    view_81: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_8: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    div_8: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_81, full_8);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_17: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_82, 0, 0, 9223372036854775807);  primals_82 = None
    slice_18: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    slice_19: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_18, 2, 0, 512);  slice_18 = None
    slice_20: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 512);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_9: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    device_put_4: "f32[]" = torch.ops.prims.device_put.default(full_9, device(type='cuda', index=0));  full_9 = None
    convert_element_type_4: "f32[]" = torch.ops.prims.convert_element_type.default(device_put_4, torch.float32);  device_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_4: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_20, div_8, convert_element_type_4);  div_8 = convert_element_type_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_8: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_59: "f32[1, 12, 512, 512]" = native_dropout_13[0]
    getitem_60: "b8[1, 12, 512, 512]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_18: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_59, [1, 12, 512, 512]);  getitem_59 = None
    view_82: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_18, [12, 512, 512]);  expand_18 = None
    expand_19: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_22, [1, 12, 512, 64])
    view_83: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_19, [12, 512, 64]);  expand_19 = None
    bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_24: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    clone_4: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_85: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_4, [1, 512, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_86: "f32[512, 768]" = torch.ops.aten.view.default(view_85, [-1, 768]);  view_85 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_35, view_86, primals_36);  primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_87: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_17, [1, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_87, 0.1, True);  view_87 = None
    getitem_61: "f32[1, 512, 768]" = native_dropout_14[0]
    getitem_62: "b8[1, 512, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_61, add_32);  getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_63: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_64: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_64)
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_35: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_69);  mul_34 = None
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_70);  mul_35 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_88: "f32[512, 768]" = torch.ops.aten.view.default(add_37, [-1, 768]);  add_37 = None
    addmm_18: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_37, view_88, primals_38);  primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_89: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_18, [1, 512, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    pow_5: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
    mul_37: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_37);  mul_37 = None
    mul_38: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    alias_9: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_4)
    add_39: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_90: "f32[512, 3072]" = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
    addmm_19: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_39, view_90, primals_40);  primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_91: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_19, [1, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_91, 0.1, True);  view_91 = None
    getitem_65: "f32[1, 512, 768]" = native_dropout_15[0]
    getitem_66: "b8[1, 512, 768]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_40: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_35, getitem_65);  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_67: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_68: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_41: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_68)
    mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_71);  mul_40 = None
    add_42: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_72);  mul_41 = primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_92: "f32[512, 768]" = torch.ops.aten.view.default(add_42, [-1, 768]);  add_42 = None
    addmm_20: "f32[512, 2304]" = torch.ops.aten.addmm.default(primals_41, view_92, primals_42);  primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_93: "f32[1, 512, 2304]" = torch.ops.aten.view.default(addmm_20, [1, 512, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_93, [768, 768, 768], 2);  view_93 = None
    getitem_69: "f32[1, 512, 768]" = split_with_sizes_5[0]
    getitem_70: "f32[1, 512, 768]" = split_with_sizes_5[1]
    getitem_71: "f32[1, 512, 768]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_94: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_69, [1, 512, 12, 64]);  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_95: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_70, [1, 512, 12, 64]);  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_26: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_96: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_71, [1, 512, 12, 64]);  getitem_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_27: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_20: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_25, [1, 12, 512, 64]);  permute_25 = None
    view_97: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_20, [12, 512, 64]);  expand_20 = None
    expand_21: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 512]);  permute_28 = None
    view_98: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_21, [12, 64, 512]);  expand_21 = None
    bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_10: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_99, full_10);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_21: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_83, 0, 0, 9223372036854775807);  primals_83 = None
    slice_22: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    slice_23: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 512);  slice_22 = None
    slice_24: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 512);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    device_put_5: "f32[]" = torch.ops.prims.device_put.default(full_11, device(type='cuda', index=0));  full_11 = None
    convert_element_type_5: "f32[]" = torch.ops.prims.convert_element_type.default(device_put_5, torch.float32);  device_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_5: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_24, div_10, convert_element_type_5);  div_10 = convert_element_type_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_10: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_16 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_72: "f32[1, 12, 512, 512]" = native_dropout_16[0]
    getitem_73: "b8[1, 12, 512, 512]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_72, [1, 12, 512, 512]);  getitem_72 = None
    view_100: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_22, [12, 512, 512]);  expand_22 = None
    expand_23: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 512, 64])
    view_101: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_23, [12, 512, 64]);  expand_23 = None
    bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
    view_102: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_5: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_103: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_5, [1, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_104: "f32[512, 768]" = torch.ops.aten.view.default(view_103, [-1, 768]);  view_103 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_43, view_104, primals_44);  primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_105: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_21, [1, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_105, 0.1, True);  view_105 = None
    getitem_74: "f32[1, 512, 768]" = native_dropout_17[0]
    getitem_75: "b8[1, 512, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_74, add_40);  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_77: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_77)
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_73);  mul_42 = None
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_74);  mul_43 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_106: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [-1, 768]);  add_45 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_45, view_106, primals_46);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    pow_6: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
    mul_45: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_107, mul_45);  mul_45 = None
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    alias_11: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_5)
    add_47: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_108: "f32[512, 3072]" = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_47, view_108, primals_48);  primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_109: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_23, [1, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_78: "f32[1, 512, 768]" = native_dropout_18[0]
    getitem_79: "b8[1, 512, 768]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_48: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_43, getitem_78);  getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_81: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_49: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_81)
    mul_48: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_75);  mul_48 = None
    add_50: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_49, primals_76);  mul_49 = primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    view_110: "f32[1, 512, 768]" = torch.ops.aten.view.default(add_50, [-1, 512, 768]);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1098, code: lm_logits = self.lm_head(hidden_states)
    permute_30: "f32[768, 50257]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    view_111: "f32[512, 768]" = torch.ops.aten.view.default(view_110, [512, 768]);  view_110 = None
    mm: "f32[512, 50257]" = torch.ops.aten.mm.default(view_111, permute_30)
    view_112: "f32[1, 512, 50257]" = torch.ops.aten.view.default(mm, [1, 512, 50257]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1105, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    slice_25: "f32[1, 511, 50257]" = torch.ops.aten.slice.Tensor(view_112, 1, 0, -1)
    slice_26: "f32[1, 511, 50257]" = torch.ops.aten.slice.Tensor(slice_25, 2, 0, 9223372036854775807);  slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1106, code: shift_labels = labels[..., 1:].contiguous()
    slice_27: "i64[1, 511]" = torch.ops.aten.slice.Tensor(primals_85, 1, 1, 9223372036854775807);  primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1109, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_113: "f32[511, 50257]" = torch.ops.aten.view.default(slice_26, [-1, 50257]);  slice_26 = None
    view_114: "i64[511]" = torch.ops.aten.view.default(slice_27, [-1]);  slice_27 = None
    amax_6: "f32[511, 1]" = torch.ops.aten.amax.default(view_113, [1], True)
    sub_19: "f32[511, 50257]" = torch.ops.aten.sub.Tensor(view_113, amax_6);  view_113 = amax_6 = None
    exp_6: "f32[511, 50257]" = torch.ops.aten.exp.default(sub_19)
    sum_7: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
    log: "f32[511, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
    sub_20: "f32[511, 50257]" = torch.ops.aten.sub.Tensor(sub_19, log);  sub_19 = log = None
    alias_12: "f32[511, 50257]" = torch.ops.aten.alias.default(sub_20)
    ne: "b8[511]" = torch.ops.aten.ne.Scalar(view_114, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "i64[511]" = torch.ops.aten.where.self(ne, view_114, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_1: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
    gather: "f32[511, 1]" = torch.ops.aten.gather.default(sub_20, 1, unsqueeze_1);  sub_20 = unsqueeze_1 = None
    squeeze: "f32[511]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[511]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[511]" = torch.ops.aten.ne.Scalar(view_114, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[511]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[511]" = torch.ops.aten.ne.Scalar(view_114, -100)
    sum_8: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_6: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
    sum_9: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
    div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type_6);  sum_9 = None
    div_13: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_6);  tangents_1 = convert_element_type_6 = None
    unsqueeze_2: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(view_114, 1);  view_114 = None
    ne_3: "b8[511, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_2, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "i64[511, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_2, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full_12: "f32[511, 50257]" = torch.ops.aten.full.default([511, 50257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[511, 50257]" = torch.ops.aten.scatter.value(full_12, 1, where_8, -1.0);  full_12 = where_8 = None
    ne_4: "b8[511, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_2, -100);  unsqueeze_2 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[511, 1]" = torch.ops.aten.where.self(ne_4, div_13, scalar_tensor_3);  ne_4 = div_13 = scalar_tensor_3 = None
    mul_50: "f32[511, 50257]" = torch.ops.aten.mul.Tensor(scatter, where_9);  scatter = where_9 = None
    alias_13: "f32[511, 50257]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    exp_7: "f32[511, 50257]" = torch.ops.aten.exp.default(alias_13);  alias_13 = None
    sum_10: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(mul_50, [1], True)
    mul_51: "f32[511, 50257]" = torch.ops.aten.mul.Tensor(exp_7, sum_10);  exp_7 = sum_10 = None
    sub_21: "f32[511, 50257]" = torch.ops.aten.sub.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    view_115: "f32[1, 511, 50257]" = torch.ops.aten.view.default(sub_21, [1, 511, 50257]);  sub_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1105, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    full_13: "f32[1, 511, 50257]" = torch.ops.aten.full.default([1, 511, 50257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[1, 511, 50257]" = torch.ops.aten.slice_scatter.default(full_13, view_115, 2, 0, 9223372036854775807);  full_13 = view_115 = None
    full_14: "f32[1, 512, 50257]" = torch.ops.aten.full.default([1, 512, 50257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_1: "f32[1, 512, 50257]" = torch.ops.aten.slice_scatter.default(full_14, slice_scatter, 1, 0, -1);  full_14 = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1105, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    add_51: "f32[1, 512, 50257]" = torch.ops.aten.add.Tensor(tangents_2, slice_scatter_1);  tangents_2 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1098, code: lm_logits = self.lm_head(hidden_states)
    view_116: "f32[512, 50257]" = torch.ops.aten.view.default(add_51, [512, 50257]);  add_51 = None
    permute_31: "f32[50257, 512]" = torch.ops.aten.permute.default(view_116, [1, 0])
    mm_1: "f32[50257, 768]" = torch.ops.aten.mm.default(permute_31, view_111);  permute_31 = view_111 = None
    permute_32: "f32[768, 50257]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    permute_33: "f32[50257, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_116, permute_33);  view_116 = permute_33 = None
    view_117: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_2, [1, 512, 768]);  mm_2 = None
    permute_34: "f32[50257, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    view_118: "f32[1, 512, 768]" = torch.ops.aten.view.default(view_117, [1, 512, 768]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    sub_22: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_81);  add_48 = getitem_81 = None
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_12);  sub_22 = None
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_118, primals_75);  primals_75 = None
    mul_54: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_53, 768)
    sum_11: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_53, [2], True)
    mul_55: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_53, mul_52);  mul_53 = None
    sum_12: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_55, [2], True);  mul_55 = None
    mul_56: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, sum_12);  sum_12 = None
    sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_54, sum_11);  mul_54 = sum_11 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_23, mul_56);  sub_23 = mul_56 = None
    div_14: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_24);  div_14 = sub_24 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_118, mul_52);  mul_52 = None
    sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_58, [0, 1]);  mul_58 = None
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_118, [0, 1]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, mul_59);  mul_59 = None
    clone_6: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_60, memory_format = torch.contiguous_format);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_119: "f32[512, 768]" = torch.ops.aten.view.default(clone_6, [512, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_35: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    mm_3: "f32[512, 3072]" = torch.ops.aten.mm.default(view_119, permute_35);  permute_35 = None
    permute_36: "f32[3072, 512]" = torch.ops.aten.permute.default(view_108, [1, 0]);  view_108 = None
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
    permute_37: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    mm_5: "f32[512, 768]" = torch.ops.aten.mm.default(view_122, permute_37);  permute_37 = None
    permute_38: "f32[768, 512]" = torch.ops.aten.permute.default(view_106, [1, 0]);  view_106 = None
    mm_6: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_38, view_122);  permute_38 = None
    sum_16: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
    view_123: "f32[3072]" = torch.ops.aten.view.default(sum_16, [3072]);  sum_16 = None
    view_124: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_5, [1, 512, 768]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_26: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_77);  add_43 = getitem_77 = None
    mul_70: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_11);  sub_26 = None
    mul_71: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_124, primals_73);  primals_73 = None
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, 768)
    sum_17: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_71, [2], True)
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, mul_70);  mul_71 = None
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_73, [2], True);  mul_73 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_18);  sum_18 = None
    sub_27: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_72, sum_17);  mul_72 = sum_17 = None
    sub_28: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_27, mul_74);  sub_27 = mul_74 = None
    div_15: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_28);  div_15 = sub_28 = None
    mul_76: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_124, mul_70);  mul_70 = None
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
    permute_39: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    mm_7: "f32[512, 768]" = torch.ops.aten.mm.default(view_125, permute_39);  permute_39 = None
    permute_40: "f32[768, 512]" = torch.ops.aten.permute.default(view_104, [1, 0]);  view_104 = None
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
    permute_42: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_12: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_42, view_129);  permute_42 = None
    permute_43: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
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
    alias_15: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_81: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_8, alias_15);  clone_8 = None
    sum_22: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_81, [-1], True)
    mul_82: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_15, sum_22);  alias_15 = sum_22 = None
    sub_29: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_24, sub_29, scalar_tensor_4);  slice_24 = sub_29 = scalar_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_16: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_10, full_10);  where_10 = full_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_132: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_16, [12, 512, 512]);  div_16 = None
    permute_44: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    bmm_14: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_44, view_132);  permute_44 = None
    permute_45: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
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
    permute_50: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    mm_9: "f32[512, 768]" = torch.ops.aten.mm.default(view_138, permute_50);  permute_50 = None
    permute_51: "f32[768, 512]" = torch.ops.aten.permute.default(view_92, [1, 0]);  view_92 = None
    mm_10: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_51, view_138);  permute_51 = None
    sum_23: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_138, [0], True);  view_138 = None
    view_139: "f32[2304]" = torch.ops.aten.view.default(sum_23, [2304]);  sum_23 = None
    view_140: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_9, [1, 512, 768]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_68);  add_40 = getitem_68 = None
    mul_83: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_10);  sub_30 = None
    mul_84: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_140, primals_71);  primals_71 = None
    mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, 768)
    sum_24: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_84, [2], True)
    mul_86: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, mul_83);  mul_84 = None
    sum_25: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_86, [2], True);  mul_86 = None
    mul_87: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_83, sum_25);  sum_25 = None
    sub_31: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_85, sum_24);  mul_85 = sum_24 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_31, mul_87);  sub_31 = mul_87 = None
    div_17: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_32);  div_17 = sub_32 = None
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_140, mul_83);  mul_83 = None
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
    permute_52: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    mm_11: "f32[512, 3072]" = torch.ops.aten.mm.default(view_141, permute_52);  permute_52 = None
    permute_53: "f32[3072, 512]" = torch.ops.aten.permute.default(view_90, [1, 0]);  view_90 = None
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
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    mm_13: "f32[512, 768]" = torch.ops.aten.mm.default(view_144, permute_54);  permute_54 = None
    permute_55: "f32[768, 512]" = torch.ops.aten.permute.default(view_88, [1, 0]);  view_88 = None
    mm_14: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_55, view_144);  permute_55 = None
    sum_29: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
    view_145: "f32[3072]" = torch.ops.aten.view.default(sum_29, [3072]);  sum_29 = None
    view_146: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_13, [1, 512, 768]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_34: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_64);  add_35 = getitem_64 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_9);  sub_34 = None
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_146, primals_69);  primals_69 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, 768)
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_102, [2], True)
    mul_104: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, mul_101);  mul_102 = None
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [2], True);  mul_104 = None
    mul_105: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_101, sum_31);  sum_31 = None
    sub_35: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_103, sum_30);  mul_103 = sum_30 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_105);  sub_35 = mul_105 = None
    div_18: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_106: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_36);  div_18 = sub_36 = None
    mul_107: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_146, mul_101);  mul_101 = None
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
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    mm_15: "f32[512, 768]" = torch.ops.aten.mm.default(view_147, permute_56);  permute_56 = None
    permute_57: "f32[768, 512]" = torch.ops.aten.permute.default(view_86, [1, 0]);  view_86 = None
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
    permute_59: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    bmm_16: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_59, view_151);  permute_59 = None
    permute_60: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
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
    alias_17: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_112: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_14, alias_17);  clone_14 = None
    sum_35: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_112, [-1], True)
    mul_113: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_17, sum_35);  alias_17 = sum_35 = None
    sub_37: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_20, sub_37, scalar_tensor_5);  slice_20 = sub_37 = scalar_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_19: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_11, full_8);  where_11 = full_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_154: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_19, [12, 512, 512]);  div_19 = None
    permute_61: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_18: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_61, view_154);  permute_61 = None
    permute_62: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
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
    permute_67: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    mm_17: "f32[512, 768]" = torch.ops.aten.mm.default(view_160, permute_67);  permute_67 = None
    permute_68: "f32[768, 512]" = torch.ops.aten.permute.default(view_74, [1, 0]);  view_74 = None
    mm_18: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_68, view_160);  permute_68 = None
    sum_36: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True);  view_160 = None
    view_161: "f32[2304]" = torch.ops.aten.view.default(sum_36, [2304]);  sum_36 = None
    view_162: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_17, [1, 512, 768]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_38: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_55);  add_32 = getitem_55 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_8);  sub_38 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_162, primals_67);  primals_67 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, 768)
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_115, [2], True)
    mul_117: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, mul_114);  mul_115 = None
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True);  mul_117 = None
    mul_118: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_114, sum_38);  sum_38 = None
    sub_39: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_116, sum_37);  mul_116 = sum_37 = None
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_39, mul_118);  sub_39 = mul_118 = None
    div_20: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_119: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_40);  div_20 = sub_40 = None
    mul_120: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_162, mul_114);  mul_114 = None
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
    permute_69: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    mm_19: "f32[512, 3072]" = torch.ops.aten.mm.default(view_163, permute_69);  permute_69 = None
    permute_70: "f32[3072, 512]" = torch.ops.aten.permute.default(view_72, [1, 0]);  view_72 = None
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
    permute_71: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    mm_21: "f32[512, 768]" = torch.ops.aten.mm.default(view_166, permute_71);  permute_71 = None
    permute_72: "f32[768, 512]" = torch.ops.aten.permute.default(view_70, [1, 0]);  view_70 = None
    mm_22: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_72, view_166);  permute_72 = None
    sum_42: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_166, [0], True);  view_166 = None
    view_167: "f32[3072]" = torch.ops.aten.view.default(sum_42, [3072]);  sum_42 = None
    view_168: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_21, [1, 512, 768]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_42: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_51);  add_27 = getitem_51 = None
    mul_132: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_7);  sub_42 = None
    mul_133: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_168, primals_65);  primals_65 = None
    mul_134: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_133, 768)
    sum_43: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [2], True)
    mul_135: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_133, mul_132);  mul_133 = None
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True);  mul_135 = None
    mul_136: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_132, sum_44);  sum_44 = None
    sub_43: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_134, sum_43);  mul_134 = sum_43 = None
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_43, mul_136);  sub_43 = mul_136 = None
    div_21: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_137: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_44);  div_21 = sub_44 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_168, mul_132);  mul_132 = None
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
    permute_73: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    mm_23: "f32[512, 768]" = torch.ops.aten.mm.default(view_169, permute_73);  permute_73 = None
    permute_74: "f32[768, 512]" = torch.ops.aten.permute.default(view_68, [1, 0]);  view_68 = None
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
    permute_76: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    bmm_20: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_76, view_173);  permute_76 = None
    permute_77: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None
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
    alias_19: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_143: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_20, alias_19);  clone_20 = None
    sum_48: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [-1], True)
    mul_144: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_19, sum_48);  alias_19 = sum_48 = None
    sub_45: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_143, mul_144);  mul_143 = mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_16, sub_45, scalar_tensor_6);  slice_16 = sub_45 = scalar_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_22: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_12, full_6);  where_12 = full_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_176: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_22, [12, 512, 512]);  div_22 = None
    permute_78: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    bmm_22: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_78, view_176);  permute_78 = None
    permute_79: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
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
    permute_84: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    mm_25: "f32[512, 768]" = torch.ops.aten.mm.default(view_182, permute_84);  permute_84 = None
    permute_85: "f32[768, 512]" = torch.ops.aten.permute.default(view_56, [1, 0]);  view_56 = None
    mm_26: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_85, view_182);  permute_85 = None
    sum_49: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
    view_183: "f32[2304]" = torch.ops.aten.view.default(sum_49, [2304]);  sum_49 = None
    view_184: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_25, [1, 512, 768]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_42);  add_24 = getitem_42 = None
    mul_145: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_6);  sub_46 = None
    mul_146: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_184, primals_63);  primals_63 = None
    mul_147: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_146, 768)
    sum_50: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_146, [2], True)
    mul_148: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_146, mul_145);  mul_146 = None
    sum_51: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True);  mul_148 = None
    mul_149: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_145, sum_51);  sum_51 = None
    sub_47: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_147, sum_50);  mul_147 = sum_50 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_149);  sub_47 = mul_149 = None
    div_23: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_150: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_48);  div_23 = sub_48 = None
    mul_151: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_184, mul_145);  mul_145 = None
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
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    mm_27: "f32[512, 3072]" = torch.ops.aten.mm.default(view_185, permute_86);  permute_86 = None
    permute_87: "f32[3072, 512]" = torch.ops.aten.permute.default(view_54, [1, 0]);  view_54 = None
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
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    mm_29: "f32[512, 768]" = torch.ops.aten.mm.default(view_188, permute_88);  permute_88 = None
    permute_89: "f32[768, 512]" = torch.ops.aten.permute.default(view_52, [1, 0]);  view_52 = None
    mm_30: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_89, view_188);  permute_89 = None
    sum_55: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
    view_189: "f32[3072]" = torch.ops.aten.view.default(sum_55, [3072]);  sum_55 = None
    view_190: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_29, [1, 512, 768]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_50: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_38);  add_19 = getitem_38 = None
    mul_163: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_5);  sub_50 = None
    mul_164: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_190, primals_61);  primals_61 = None
    mul_165: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
    sum_56: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
    mul_166: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_163);  mul_164 = None
    sum_57: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
    mul_167: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_163, sum_57);  sum_57 = None
    sub_51: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_56);  mul_165 = sum_56 = None
    sub_52: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_167);  sub_51 = mul_167 = None
    div_24: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_52);  div_24 = sub_52 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_190, mul_163);  mul_163 = None
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
    permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    mm_31: "f32[512, 768]" = torch.ops.aten.mm.default(view_191, permute_90);  permute_90 = None
    permute_91: "f32[768, 512]" = torch.ops.aten.permute.default(view_50, [1, 0]);  view_50 = None
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
    permute_93: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    bmm_24: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_93, view_195);  permute_93 = None
    permute_94: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
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
    alias_21: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_174: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_26, alias_21);  clone_26 = None
    sum_61: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_174, [-1], True)
    mul_175: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_21, sum_61);  alias_21 = sum_61 = None
    sub_53: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_12, sub_53, scalar_tensor_7);  slice_12 = sub_53 = scalar_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_25: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_13, full_4);  where_13 = full_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_198: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_25, [12, 512, 512]);  div_25 = None
    permute_95: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    bmm_26: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_95, view_198);  permute_95 = None
    permute_96: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
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
    permute_101: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    mm_33: "f32[512, 768]" = torch.ops.aten.mm.default(view_204, permute_101);  permute_101 = None
    permute_102: "f32[768, 512]" = torch.ops.aten.permute.default(view_38, [1, 0]);  view_38 = None
    mm_34: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_102, view_204);  permute_102 = None
    sum_62: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
    view_205: "f32[2304]" = torch.ops.aten.view.default(sum_62, [2304]);  sum_62 = None
    view_206: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_33, [1, 512, 768]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_54: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_29);  add_16 = getitem_29 = None
    mul_176: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_4);  sub_54 = None
    mul_177: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_206, primals_59);  primals_59 = None
    mul_178: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_177, 768)
    sum_63: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True)
    mul_179: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_177, mul_176);  mul_177 = None
    sum_64: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True);  mul_179 = None
    mul_180: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_176, sum_64);  sum_64 = None
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_178, sum_63);  mul_178 = sum_63 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_180);  sub_55 = mul_180 = None
    div_26: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_181: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_56);  div_26 = sub_56 = None
    mul_182: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_206, mul_176);  mul_176 = None
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
    permute_103: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    mm_35: "f32[512, 3072]" = torch.ops.aten.mm.default(view_207, permute_103);  permute_103 = None
    permute_104: "f32[3072, 512]" = torch.ops.aten.permute.default(view_36, [1, 0]);  view_36 = None
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
    permute_105: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    mm_37: "f32[512, 768]" = torch.ops.aten.mm.default(view_210, permute_105);  permute_105 = None
    permute_106: "f32[768, 512]" = torch.ops.aten.permute.default(view_34, [1, 0]);  view_34 = None
    mm_38: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_106, view_210);  permute_106 = None
    sum_68: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
    view_211: "f32[3072]" = torch.ops.aten.view.default(sum_68, [3072]);  sum_68 = None
    view_212: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_37, [1, 512, 768]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_58: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_25);  add_11 = getitem_25 = None
    mul_194: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_3);  sub_58 = None
    mul_195: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_212, primals_57);  primals_57 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, 768)
    sum_69: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, mul_194);  mul_195 = None
    sum_70: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_194, sum_70);  sum_70 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_196, sum_69);  mul_196 = sum_69 = None
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_198);  sub_59 = mul_198 = None
    div_27: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_60);  div_27 = sub_60 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_212, mul_194);  mul_194 = None
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
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    mm_39: "f32[512, 768]" = torch.ops.aten.mm.default(view_213, permute_107);  permute_107 = None
    permute_108: "f32[768, 512]" = torch.ops.aten.permute.default(view_32, [1, 0]);  view_32 = None
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
    permute_110: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    bmm_28: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_110, view_217);  permute_110 = None
    permute_111: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
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
    alias_23: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_205: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_32, alias_23);  clone_32 = None
    sum_74: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [-1], True)
    mul_206: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_23, sum_74);  alias_23 = sum_74 = None
    sub_61: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_8, sub_61, scalar_tensor_8);  slice_8 = sub_61 = scalar_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_28: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_14, full_2);  where_14 = full_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_220: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_28, [12, 512, 512]);  div_28 = None
    permute_112: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    bmm_30: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_112, view_220);  permute_112 = None
    permute_113: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
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
    permute_118: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    mm_41: "f32[512, 768]" = torch.ops.aten.mm.default(view_226, permute_118);  permute_118 = None
    permute_119: "f32[768, 512]" = torch.ops.aten.permute.default(view_20, [1, 0]);  view_20 = None
    mm_42: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_119, view_226);  permute_119 = None
    sum_75: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[2304]" = torch.ops.aten.view.default(sum_75, [2304]);  sum_75 = None
    view_228: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_41, [1, 512, 768]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_16);  add_8 = getitem_16 = None
    mul_207: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_2);  sub_62 = None
    mul_208: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_228, primals_55);  primals_55 = None
    mul_209: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_208, 768)
    sum_76: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True)
    mul_210: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_208, mul_207);  mul_208 = None
    sum_77: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True);  mul_210 = None
    mul_211: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_207, sum_77);  sum_77 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_209, sum_76);  mul_209 = sum_76 = None
    sub_64: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_211);  sub_63 = mul_211 = None
    div_29: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_212: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_64);  div_29 = sub_64 = None
    mul_213: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_228, mul_207);  mul_207 = None
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
    permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    mm_43: "f32[512, 3072]" = torch.ops.aten.mm.default(view_229, permute_120);  permute_120 = None
    permute_121: "f32[3072, 512]" = torch.ops.aten.permute.default(view_18, [1, 0]);  view_18 = None
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
    permute_122: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    mm_45: "f32[512, 768]" = torch.ops.aten.mm.default(view_232, permute_122);  permute_122 = None
    permute_123: "f32[768, 512]" = torch.ops.aten.permute.default(view_16, [1, 0]);  view_16 = None
    mm_46: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_123, view_232);  permute_123 = None
    sum_81: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_232, [0], True);  view_232 = None
    view_233: "f32[3072]" = torch.ops.aten.view.default(sum_81, [3072]);  sum_81 = None
    view_234: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_45, [1, 512, 768]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_12);  add_3 = getitem_12 = None
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_1);  sub_66 = None
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_234, primals_53);  primals_53 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_226, 768)
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True)
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_226, mul_225);  mul_226 = None
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True);  mul_228 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_225, sum_83);  sum_83 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_227, sum_82);  mul_227 = sum_82 = None
    sub_68: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_67, mul_229);  sub_67 = mul_229 = None
    div_30: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_68);  div_30 = sub_68 = None
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_234, mul_225);  mul_225 = None
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
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
    mm_47: "f32[512, 768]" = torch.ops.aten.mm.default(view_235, permute_124);  permute_124 = None
    permute_125: "f32[768, 512]" = torch.ops.aten.permute.default(view_14, [1, 0]);  view_14 = None
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
    permute_127: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_32: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_127, view_239);  permute_127 = None
    permute_128: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
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
    alias_25: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_236: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_38, alias_25);  clone_38 = None
    sum_87: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [-1], True)
    mul_237: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_25, sum_87);  alias_25 = sum_87 = None
    sub_69: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_4, sub_69, scalar_tensor_9);  slice_4 = sub_69 = scalar_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_31: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(where_15, full);  where_15 = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_242: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_31, [12, 512, 512]);  div_31 = None
    permute_129: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    bmm_34: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_129, view_242);  permute_129 = None
    permute_130: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
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
    permute_135: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    mm_49: "f32[512, 768]" = torch.ops.aten.mm.default(view_248, permute_135);  permute_135 = None
    permute_136: "f32[768, 512]" = torch.ops.aten.permute.default(view_2, [1, 0]);  view_2 = None
    mm_50: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_136, view_248);  permute_136 = None
    sum_88: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
    view_249: "f32[2304]" = torch.ops.aten.view.default(sum_88, [2304]);  sum_88 = None
    view_250: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_49, [1, 512, 768]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(getitem, getitem_3);  getitem = getitem_3 = None
    mul_238: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt);  sub_70 = None
    mul_239: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_250, primals_51);  primals_51 = None
    mul_240: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_239, 768)
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True)
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_239, mul_238);  mul_239 = None
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True);  mul_241 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_238, sum_90);  sum_90 = None
    sub_71: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_240, sum_89);  mul_240 = sum_89 = None
    sub_72: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_242);  sub_71 = mul_242 = None
    div_32: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_72);  div_32 = sub_72 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_250, mul_238);  mul_238 = None
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
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(view_1, -1)
    unsqueeze_3: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_3, scalar_tensor_10, clone_42);  unsqueeze_3 = scalar_tensor_10 = None
    full_15: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_15, [view_1], where_16, True);  full_15 = view_1 = where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_4, scalar_tensor_11, clone_42);  unsqueeze_4 = scalar_tensor_11 = clone_42 = None
    full_16: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[50257, 768]" = torch.ops.aten._unsafe_index_put.default(full_16, [view], where_17, True);  full_16 = view = where_17 = None
    return pytree.tree_unflatten([div_12, view_112, permute_1, permute_2, permute_6, permute_7, permute_11, permute_12, permute_16, permute_17, permute_21, permute_22, permute_26, permute_27, view_249, mm_50, view_236, mm_48, view_233, mm_46, view_230, mm_44, view_227, mm_42, view_214, mm_40, view_211, mm_38, view_208, mm_36, view_205, mm_34, view_192, mm_32, view_189, mm_30, view_186, mm_28, view_183, mm_26, view_170, mm_24, view_167, mm_22, view_164, mm_20, view_161, mm_18, view_148, mm_16, view_145, mm_14, view_142, mm_12, view_139, mm_10, view_126, mm_8, view_123, mm_6, view_120, mm_4, _unsafe_index_put_1, _unsafe_index_put, sum_91, sum_92, sum_84, sum_85, sum_78, sum_79, sum_71, sum_72, sum_65, sum_66, sum_58, sum_59, sum_52, sum_53, sum_45, sum_46, sum_39, sum_40, sum_32, sum_33, sum_26, sum_27, sum_19, sum_20, sum_13, sum_14, permute_34, None, None, None, None, None, None, None, None], self._out_spec)
    