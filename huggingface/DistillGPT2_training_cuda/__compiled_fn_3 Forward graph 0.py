from __future__ import annotations



def forward(self, primals_1: "f32[2304]", primals_2: "f32[768, 2304]", primals_3: "f32[768]", primals_4: "f32[768, 768]", primals_5: "f32[3072]", primals_6: "f32[768, 3072]", primals_7: "f32[768]", primals_8: "f32[3072, 768]", primals_9: "f32[2304]", primals_10: "f32[768, 2304]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[3072]", primals_14: "f32[768, 3072]", primals_15: "f32[768]", primals_16: "f32[3072, 768]", primals_17: "f32[2304]", primals_18: "f32[768, 2304]", primals_19: "f32[768]", primals_20: "f32[768, 768]", primals_21: "f32[3072]", primals_22: "f32[768, 3072]", primals_23: "f32[768]", primals_24: "f32[3072, 768]", primals_25: "f32[2304]", primals_26: "f32[768, 2304]", primals_27: "f32[768]", primals_28: "f32[768, 768]", primals_29: "f32[3072]", primals_30: "f32[768, 3072]", primals_31: "f32[768]", primals_32: "f32[3072, 768]", primals_33: "f32[2304]", primals_34: "f32[768, 2304]", primals_35: "f32[768]", primals_36: "f32[768, 768]", primals_37: "f32[3072]", primals_38: "f32[768, 3072]", primals_39: "f32[768]", primals_40: "f32[3072, 768]", primals_41: "f32[2304]", primals_42: "f32[768, 2304]", primals_43: "f32[768]", primals_44: "f32[768, 768]", primals_45: "f32[3072]", primals_46: "f32[768, 3072]", primals_47: "f32[768]", primals_48: "f32[3072, 768]", primals_49: "f32[50257, 768]", primals_50: "f32[1024, 768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[50257, 768]", primals_78: "b8[1, 1, 1024, 1024]", primals_79: "b8[1, 1, 1024, 1024]", primals_80: "b8[1, 1, 1024, 1024]", primals_81: "b8[1, 1, 1024, 1024]", primals_82: "b8[1, 1, 1024, 1024]", primals_83: "b8[1, 1, 1024, 1024]", primals_84: "i64[1, 512]", primals_85: "i64[1, 512]"):
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
    sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(getitem, getitem_3);  getitem_3 = None
    mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul, primals_51)
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
    full_default: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_9, full_default);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_78, 0, 0, 9223372036854775807);  primals_78 = None
    slice_2: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    slice_3: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 512);  slice_2 = None
    slice_4: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 512);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_4, div, full_default_1);  div = None
    
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
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_9, getitem);  getitem_9 = getitem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_11: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_12: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_12);  getitem_12 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_53)
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_54);  mul_3 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_16: "f32[512, 768]" = torch.ops.aten.view.default(add_5, [-1, 768]);  add_5 = None
    addmm_2: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_5, view_16, primals_6);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_17: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_2, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    pow_1: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_17, mul_5);  view_17 = mul_5 = None
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    add_7: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
    
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
    add_8: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_3, getitem_13);  add_3 = getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_15: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_16: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_15, 1e-05);  getitem_15 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_16);  getitem_16 = None
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_55)
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
    div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_27, full_default);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_5: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_79, 0, 0, 9223372036854775807);  primals_79 = None
    slice_6: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    slice_7: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 512);  slice_6 = None
    slice_8: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 512);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_1: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_8, div_2, full_default_1);  div_2 = None
    
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
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_22, add_8);  getitem_22 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_25: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_25);  getitem_25 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_57)
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_58);  mul_11 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_34: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [-1, 768]);  add_13 = None
    addmm_6: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_13, view_34, primals_14);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_35: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_6, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_2: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_13);  view_35 = mul_13 = None
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    add_15: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
    
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
    add_16: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_11, getitem_26);  add_11 = getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_17: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_29);  getitem_29 = None
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_59)
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
    div_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_45, full_default);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_9: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_80, 0, 0, 9223372036854775807);  primals_80 = None
    slice_10: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    slice_11: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_10, 2, 0, 512);  slice_10 = None
    slice_12: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 512);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_2: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_12, div_4, full_default_1);  div_4 = None
    
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
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_35, add_16);  getitem_35 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_37: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_38: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_38);  getitem_38 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_61)
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_62);  mul_19 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_52: "f32[512, 768]" = torch.ops.aten.view.default(add_21, [-1, 768]);  add_21 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_21, view_52, primals_22);  primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_53: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
    pow_3: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_53, mul_21);  view_53 = mul_21 = None
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    add_23: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
    
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
    add_24: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_19, getitem_39);  add_19 = getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_41: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_42: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_25: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_42);  getitem_42 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_63)
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
    div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_63, full_default);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_13: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_81, 0, 0, 9223372036854775807);  primals_81 = None
    slice_14: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    slice_15: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 512);  slice_14 = None
    slice_16: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 512);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_3: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_16, div_6, full_default_1);  div_6 = None
    
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
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_48, add_24);  getitem_48 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_51: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_51);  getitem_51 = None
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_27: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_65)
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_66);  mul_27 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_70: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [-1, 768]);  add_29 = None
    addmm_14: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_29, view_70, primals_30);  primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_71: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_14, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    pow_4: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
    mul_29: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_71, mul_29);  view_71 = mul_29 = None
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    add_31: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
    
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
    add_32: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_27, getitem_52);  add_27 = getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_55: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_33: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_55);  getitem_55 = None
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_67)
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
    div_8: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_81, full_default);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_17: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_82, 0, 0, 9223372036854775807);  primals_82 = None
    slice_18: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    slice_19: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_18, 2, 0, 512);  slice_18 = None
    slice_20: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 512);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_4: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_20, div_8, full_default_1);  div_8 = None
    
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
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_61, add_32);  getitem_61 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_63: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_64: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_64);  getitem_64 = None
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_35: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_69)
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_70);  mul_35 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_88: "f32[512, 768]" = torch.ops.aten.view.default(add_37, [-1, 768]);  add_37 = None
    addmm_18: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_37, view_88, primals_38);  primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_89: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_18, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    pow_5: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
    mul_37: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_37);  view_89 = mul_37 = None
    mul_38: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_39: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
    
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
    add_40: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_35, getitem_65);  add_35 = getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_67: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_68: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_41: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_68);  getitem_68 = None
    mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_71)
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
    div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_99, full_default);  view_99 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_21: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_83, 0, 0, 9223372036854775807);  primals_83 = None
    slice_22: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    slice_23: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 512);  slice_22 = None
    slice_24: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 512);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_5: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_24, div_10, full_default_1);  div_10 = full_default_1 = None
    
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
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_74, add_40);  getitem_74 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_77: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_77);  getitem_77 = None
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_73)
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_74);  mul_43 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_106: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [-1, 768]);  add_45 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_45, view_106, primals_46);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    pow_6: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
    mul_45: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_107, mul_45);  view_107 = mul_45 = None
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    add_47: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
    
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
    add_48: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_43, getitem_78);  add_43 = getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_81: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_49: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_81);  add_48 = getitem_81 = None
    mul_48: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_75)
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
    slice_27: "i64[1, 511]" = torch.ops.aten.slice.Tensor(primals_85, 1, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1109, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_113: "f32[511, 50257]" = torch.ops.aten.view.default(slice_26, [-1, 50257]);  slice_26 = None
    view_114: "i64[511]" = torch.ops.aten.view.default(slice_27, [-1]);  slice_27 = None
    amax_6: "f32[511, 1]" = torch.ops.aten.amax.default(view_113, [1], True)
    sub_19: "f32[511, 50257]" = torch.ops.aten.sub.Tensor(view_113, amax_6);  view_113 = amax_6 = None
    exp_6: "f32[511, 50257]" = torch.ops.aten.exp.default(sub_19)
    sum_7: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
    log: "f32[511, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
    sub_20: "f32[511, 50257]" = torch.ops.aten.sub.Tensor(sub_19, log);  sub_19 = log = None
    ne: "b8[511]" = torch.ops.aten.ne.Scalar(view_114, -100)
    full_default_12: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_6: "i64[511]" = torch.ops.aten.where.self(ne, view_114, full_default_12);  view_114 = full_default_12 = None
    unsqueeze_1: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
    gather: "f32[511, 1]" = torch.ops.aten.gather.default(sub_20, 1, unsqueeze_1);  unsqueeze_1 = None
    squeeze: "f32[511]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[511]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_7: "f32[511]" = torch.ops.aten.where.self(ne, neg, full_default_13);  neg = full_default_13 = None
    sum_8: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type_6: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
    sum_9: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
    div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type_6);  sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1098, code: lm_logits = self.lm_head(hidden_states)
    permute_33: "f32[50257, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    div_14: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_35: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    permute_36: "f32[3072, 512]" = torch.ops.aten.permute.default(view_108, [1, 0]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_37: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    permute_38: "f32[768, 512]" = torch.ops.aten.permute.default(view_106, [1, 0]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_15: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_39: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    permute_40: "f32[768, 512]" = torch.ops.aten.permute.default(view_104, [1, 0]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_42: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    permute_43: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_15: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_44: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    permute_45: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_50: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    permute_51: "f32[768, 512]" = torch.ops.aten.permute.default(view_92, [1, 0]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_17: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_52: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    permute_53: "f32[3072, 512]" = torch.ops.aten.permute.default(view_90, [1, 0]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    permute_55: "f32[768, 512]" = torch.ops.aten.permute.default(view_88, [1, 0]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_18: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    permute_57: "f32[768, 512]" = torch.ops.aten.permute.default(view_86, [1, 0]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_59: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    permute_60: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_17: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_61: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    permute_62: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_67: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    permute_68: "f32[768, 512]" = torch.ops.aten.permute.default(view_74, [1, 0]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_20: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_69: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    permute_70: "f32[3072, 512]" = torch.ops.aten.permute.default(view_72, [1, 0]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_71: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    permute_72: "f32[768, 512]" = torch.ops.aten.permute.default(view_70, [1, 0]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_21: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_73: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    permute_74: "f32[768, 512]" = torch.ops.aten.permute.default(view_68, [1, 0]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_76: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    permute_77: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_19: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_78: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    permute_79: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_84: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    permute_85: "f32[768, 512]" = torch.ops.aten.permute.default(view_56, [1, 0]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_23: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    permute_87: "f32[3072, 512]" = torch.ops.aten.permute.default(view_54, [1, 0]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    permute_89: "f32[768, 512]" = torch.ops.aten.permute.default(view_52, [1, 0]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_24: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    permute_91: "f32[768, 512]" = torch.ops.aten.permute.default(view_50, [1, 0]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_93: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    permute_94: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_21: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_95: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    permute_96: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_101: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    permute_102: "f32[768, 512]" = torch.ops.aten.permute.default(view_38, [1, 0]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_26: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_103: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    permute_104: "f32[3072, 512]" = torch.ops.aten.permute.default(view_36, [1, 0]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_105: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    permute_106: "f32[768, 512]" = torch.ops.aten.permute.default(view_34, [1, 0]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_27: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    permute_108: "f32[768, 512]" = torch.ops.aten.permute.default(view_32, [1, 0]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_110: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    permute_111: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_23: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_112: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    permute_113: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_118: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    permute_119: "f32[768, 512]" = torch.ops.aten.permute.default(view_20, [1, 0]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_29: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    permute_121: "f32[3072, 512]" = torch.ops.aten.permute.default(view_18, [1, 0]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_122: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    permute_123: "f32[768, 512]" = torch.ops.aten.permute.default(view_16, [1, 0]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    div_30: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
    permute_125: "f32[768, 512]" = torch.ops.aten.permute.default(view_14, [1, 0]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    permute_127: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    permute_128: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_25: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_129: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    permute_130: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_135: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    permute_136: "f32[768, 512]" = torch.ops.aten.permute.default(view_2, [1, 0]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    div_32: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [div_12, view_112, permute_1, permute_2, permute_6, permute_7, permute_11, permute_12, permute_16, permute_17, permute_21, permute_22, permute_26, permute_27, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_85, view, view_1, getitem_1, mul, slice_4, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, slice_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, slice_12, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, slice_16, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, slice_20, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, slice_24, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, view_111, sub_20, convert_element_type_6, permute_33, div_14, permute_35, permute_36, permute_37, permute_38, div_15, permute_39, permute_40, permute_42, permute_43, alias_15, permute_44, permute_45, permute_50, permute_51, div_17, permute_52, permute_53, permute_54, permute_55, div_18, permute_56, permute_57, permute_59, permute_60, alias_17, permute_61, permute_62, permute_67, permute_68, div_20, permute_69, permute_70, permute_71, permute_72, div_21, permute_73, permute_74, permute_76, permute_77, alias_19, permute_78, permute_79, permute_84, permute_85, div_23, permute_86, permute_87, permute_88, permute_89, div_24, permute_90, permute_91, permute_93, permute_94, alias_21, permute_95, permute_96, permute_101, permute_102, div_26, permute_103, permute_104, permute_105, permute_106, div_27, permute_107, permute_108, permute_110, permute_111, alias_23, permute_112, permute_113, permute_118, permute_119, div_29, permute_120, permute_121, permute_122, permute_123, div_30, permute_124, permute_125, permute_127, permute_128, alias_25, permute_129, permute_130, permute_135, permute_136, div_32]
    