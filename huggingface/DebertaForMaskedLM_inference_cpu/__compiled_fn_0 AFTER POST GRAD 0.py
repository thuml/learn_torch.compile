from __future__ import annotations



def forward(self, arg0_1: "f32[768]", arg1_1: "f32[768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768]", arg74_1: "f32[50265, 768]", arg75_1: "f32[512, 768]", arg76_1: "f32[2304, 768]", arg77_1: "f32[768, 768]", arg78_1: "f32[768]", arg79_1: "f32[3072, 768]", arg80_1: "f32[3072]", arg81_1: "f32[768, 3072]", arg82_1: "f32[768]", arg83_1: "f32[2304, 768]", arg84_1: "f32[768, 768]", arg85_1: "f32[768]", arg86_1: "f32[3072, 768]", arg87_1: "f32[3072]", arg88_1: "f32[768, 3072]", arg89_1: "f32[768]", arg90_1: "f32[2304, 768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[3072, 768]", arg94_1: "f32[3072]", arg95_1: "f32[768, 3072]", arg96_1: "f32[768]", arg97_1: "f32[2304, 768]", arg98_1: "f32[768, 768]", arg99_1: "f32[768]", arg100_1: "f32[3072, 768]", arg101_1: "f32[3072]", arg102_1: "f32[768, 3072]", arg103_1: "f32[768]", arg104_1: "f32[2304, 768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[3072, 768]", arg108_1: "f32[3072]", arg109_1: "f32[768, 3072]", arg110_1: "f32[768]", arg111_1: "f32[2304, 768]", arg112_1: "f32[768, 768]", arg113_1: "f32[768]", arg114_1: "f32[3072, 768]", arg115_1: "f32[3072]", arg116_1: "f32[768, 3072]", arg117_1: "f32[768]", arg118_1: "f32[2304, 768]", arg119_1: "f32[768, 768]", arg120_1: "f32[768]", arg121_1: "f32[3072, 768]", arg122_1: "f32[3072]", arg123_1: "f32[768, 3072]", arg124_1: "f32[768]", arg125_1: "f32[2304, 768]", arg126_1: "f32[768, 768]", arg127_1: "f32[768]", arg128_1: "f32[3072, 768]", arg129_1: "f32[3072]", arg130_1: "f32[768, 3072]", arg131_1: "f32[768]", arg132_1: "f32[2304, 768]", arg133_1: "f32[768, 768]", arg134_1: "f32[768]", arg135_1: "f32[3072, 768]", arg136_1: "f32[3072]", arg137_1: "f32[768, 3072]", arg138_1: "f32[768]", arg139_1: "f32[2304, 768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[3072, 768]", arg143_1: "f32[3072]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[2304, 768]", arg147_1: "f32[768, 768]", arg148_1: "f32[768]", arg149_1: "f32[3072, 768]", arg150_1: "f32[3072]", arg151_1: "f32[768, 3072]", arg152_1: "f32[768]", arg153_1: "f32[2304, 768]", arg154_1: "f32[768, 768]", arg155_1: "f32[768]", arg156_1: "f32[3072, 768]", arg157_1: "f32[3072]", arg158_1: "f32[768, 3072]", arg159_1: "f32[768]", arg160_1: "f32[768, 768]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[768]", arg164_1: "f32[50265, 768]", arg165_1: "f32[50265]", arg166_1: "i64[1, 512]", arg167_1: "i64[1, 512]", arg168_1: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:809, code: mask = mask.unsqueeze(2)
    full_default: "f32[1, 512, 1]" = torch.ops.aten.full.default([1, 512, 1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:786, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg74_1, arg167_1, 0);  arg74_1 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:789, code: position_embeddings = self.position_embeddings(position_ids.long())
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg75_1, arg166_1);  arg75_1 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:795, code: embeddings += position_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add, mean)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add, mean);  add = mean = None
    pow_1: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
    mean_1: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-07);  mean_1 = None
    sqrt: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_1);  add_1 = None
    div: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_1, sqrt);  sub_1 = sqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg0_1, div);  arg0_1 = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:812, code: embeddings = embeddings * mask
    add_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul, arg1_1);  mul = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view: "f32[512, 768]" = torch.ops.aten.reshape.default(add_2, [512, 768])
    permute: "f32[768, 2304]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    mm: "f32[512, 2304]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
    view_1: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm, [1, 512, 2304]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_2: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_1, [1, 512, 12, -1]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_1: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split = torch.ops.aten.split.Tensor(permute_1, 64, -1);  permute_1 = None
    getitem: "f32[1, 12, 512, 64]" = split[0]
    getitem_1: "f32[1, 12, 512, 64]" = split[1]
    getitem_2: "f32[1, 12, 512, 64]" = split[2];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant0: "f32[]" = self._tensor_constant0
    lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    mul_3: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy, 1);  lift_fresh_copy = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:972, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:421, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    unsqueeze_1: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_2: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 1, 512]" = torch.ops.aten.squeeze.dim(unsqueeze_2, -2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:422, code: attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    unsqueeze_3: "f32[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
    mul_2: "f32[1, 1, 512, 512]" = torch.ops.aten.mul.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant1: "f32[]" = self._tensor_constant1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_2: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_4: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
    unsqueeze_5: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_3: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_5, [1, 1, 12, -1]);  unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_3: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem, permute_2);  getitem = permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_1: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_3, full_default_1);  add_3 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_1, [1, 12, 512, 64]);  div_1 = None
    view_5: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand, [12, 512, 64]);  expand = None
    permute_4: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
    expand_1: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_4, [1, 12, 64, 512]);  permute_4 = None
    view_6: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_1, [12, 64, 512]);  expand_1 = None
    bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
    view_7: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [1, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_7);  full_default_3 = view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_2: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_1: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_4, div_2);  full_default_2 = full_default_4 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_4: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_1, [1, 12, 512, 512]);  where_1 = None
    view_10: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_4, [12, 512, 512]);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_6: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg3_1, 0);  arg3_1 = None
    unsqueeze_7: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_4: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_7, [1, 1, 12, -1]);  unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_3: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_4: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_2, permute_3);  getitem_2 = permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_4, [1, 12, 512, 64]);  add_4 = None
    view_9: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_3, [12, 512, 64]);  expand_3 = None
    bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_10, view_9);  view_10 = view_9 = None
    view_11: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_5: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_12: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone, [1, 512, -1]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_13: "f32[512, 768]" = torch.ops.aten.reshape.default(view_12, [512, 768]);  view_12 = None
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(arg78_1, view_13, permute_6);  arg78_1 = view_13 = permute_6 = None
    view_14: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm, [1, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_14, add_2);  view_14 = add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_2: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_5, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, mean_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, mean_2);  add_5 = mean_2 = None
    pow_2: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_3, 2);  sub_3 = None
    mean_3: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-07);  mean_3 = None
    sqrt_2: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    div_3: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_4, sqrt_2);  sub_4 = sqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg4_1, div_3);  arg4_1 = div_3 = None
    add_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, arg5_1);  mul_4 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_15: "f32[512, 768]" = torch.ops.aten.reshape.default(add_7, [512, 768])
    permute_7: "f32[768, 3072]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    addmm_1: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg80_1, view_15, permute_7);  arg80_1 = view_15 = permute_7 = None
    view_16: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476);  view_16 = None
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_17: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_7, [512, 3072]);  mul_7 = None
    permute_8: "f32[3072, 768]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(arg82_1, view_17, permute_8);  arg82_1 = view_17 = permute_8 = None
    view_18: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_18, add_7);  view_18 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_4: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_9, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, mean_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, mean_4);  add_9 = mean_4 = None
    pow_3: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_5, 2);  sub_5 = None
    mean_5: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-07);  mean_5 = None
    sqrt_3: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    div_4: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_6, sqrt_3);  sub_6 = sqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg6_1, div_4);  arg6_1 = div_4 = None
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_8, arg7_1);  mul_8 = arg7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_19: "f32[512, 768]" = torch.ops.aten.reshape.default(add_11, [512, 768])
    permute_9: "f32[768, 2304]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    mm_1: "f32[512, 2304]" = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
    view_20: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_1, [1, 512, 2304]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_21: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_20, [1, 512, 12, -1]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_10: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_1 = torch.ops.aten.split.Tensor(permute_10, 64, -1);  permute_10 = None
    getitem_3: "f32[1, 12, 512, 64]" = split_1[0]
    getitem_4: "f32[1, 12, 512, 64]" = split_1[1]
    getitem_5: "f32[1, 12, 512, 64]" = split_1[2];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant2: "f32[]" = self._tensor_constant2
    lift_fresh_copy_2: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
    mul_9: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_2, 1);  lift_fresh_copy_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_1: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant3: "f32[]" = self._tensor_constant3
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_8: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_6: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_8: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg8_1, 0);  arg8_1 = None
    unsqueeze_9: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_22: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_9, [1, 1, 12, -1]);  unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_11: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_12: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_3, permute_11);  getitem_3 = permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_5: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_5: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_12, full_default_5);  add_12 = full_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_5: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_5, [1, 12, 512, 64]);  div_5 = None
    view_24: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_5, [12, 512, 64]);  expand_5 = None
    permute_13: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_4, [0, 1, 3, 2]);  getitem_4 = None
    expand_6: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_13, [1, 12, 64, 512]);  permute_13 = None
    view_25: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_6, [12, 64, 512]);  expand_6 = None
    bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = view_25 = None
    view_26: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_2: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_6, full_default_7, view_26);  full_default_7 = view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_2, amax_1);  where_2 = amax_1 = None
    exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_3: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_6, full_default_8, div_6);  full_default_6 = full_default_8 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_9: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_3, [1, 12, 512, 512]);  where_3 = None
    view_29: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_9, [12, 512, 512]);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_10: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg9_1, 0);  arg9_1 = None
    unsqueeze_11: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_23: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_11, [1, 1, 12, -1]);  unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_12: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_13: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_5, permute_12);  getitem_5 = permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_8: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_13, [1, 12, 512, 64]);  add_13 = None
    view_28: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_8, [12, 512, 64]);  expand_8 = None
    bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_29, view_28);  view_29 = view_28 = None
    view_30: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_14: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_1: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_31: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_1, [1, 512, -1]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[512, 768]" = torch.ops.aten.reshape.default(view_31, [512, 768]);  view_31 = None
    permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(arg85_1, view_32, permute_15);  arg85_1 = view_32 = permute_15 = None
    view_33: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_14: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_33, add_11);  view_33 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_6: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_14, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, mean_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, mean_6);  add_14 = mean_6 = None
    pow_4: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_8, 2);  sub_8 = None
    mean_7: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_15: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-07);  mean_7 = None
    sqrt_5: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    div_7: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_9, sqrt_5);  sub_9 = sqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg10_1, div_7);  arg10_1 = div_7 = None
    add_16: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, arg11_1);  mul_10 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_34: "f32[512, 768]" = torch.ops.aten.reshape.default(add_16, [512, 768])
    permute_16: "f32[768, 3072]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_4: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg87_1, view_34, permute_16);  arg87_1 = view_34 = permute_16 = None
    view_35: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_11: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476);  view_35 = None
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_17: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_17);  mul_11 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_36: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_13, [512, 3072]);  mul_13 = None
    permute_17: "f32[3072, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(arg89_1, view_36, permute_17);  arg89_1 = view_36 = permute_17 = None
    view_37: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_18: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_37, add_16);  view_37 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_8: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_18, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_10: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_8);  add_18 = mean_8 = None
    pow_5: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_10, 2);  sub_10 = None
    mean_9: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_19: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-07);  mean_9 = None
    sqrt_6: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    div_8: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_11, sqrt_6);  sub_11 = sqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_14: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg12_1, div_8);  arg12_1 = div_8 = None
    add_20: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_14, arg13_1);  mul_14 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_38: "f32[512, 768]" = torch.ops.aten.reshape.default(add_20, [512, 768])
    permute_18: "f32[768, 2304]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    mm_2: "f32[512, 2304]" = torch.ops.aten.mm.default(view_38, permute_18);  view_38 = permute_18 = None
    view_39: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 2304]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_40: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_39, [1, 512, 12, -1]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_19: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_2 = torch.ops.aten.split.Tensor(permute_19, 64, -1);  permute_19 = None
    getitem_6: "f32[1, 12, 512, 64]" = split_2[0]
    getitem_7: "f32[1, 12, 512, 64]" = split_2[1]
    getitem_8: "f32[1, 12, 512, 64]" = split_2[2];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant4: "f32[]" = self._tensor_constant4
    lift_fresh_copy_4: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
    mul_15: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_4, 1);  lift_fresh_copy_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_2: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant5: "f32[]" = self._tensor_constant5
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_12: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_10: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_12: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
    unsqueeze_13: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_41: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_13, [1, 1, 12, -1]);  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_20: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_21: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_6, permute_20);  getitem_6 = permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_9: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_9: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_21, full_default_9);  add_21 = full_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_10: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_9, [1, 12, 512, 64]);  div_9 = None
    view_43: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_10, [12, 512, 64]);  expand_10 = None
    permute_22: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_7, [0, 1, 3, 2]);  getitem_7 = None
    expand_11: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_22, [1, 12, 64, 512]);  permute_22 = None
    view_44: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_11, [12, 64, 512]);  expand_11 = None
    bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
    view_45: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_4: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_10, full_default_11, view_45);  full_default_11 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_12: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_4, amax_2);  where_4 = amax_2 = None
    exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_5: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_10, full_default_12, div_10);  full_default_10 = full_default_12 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_14: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_5, [1, 12, 512, 512]);  where_5 = None
    view_48: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_14, [12, 512, 512]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_14: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg15_1, 0);  arg15_1 = None
    unsqueeze_15: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 1);  unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_42: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_15, [1, 1, 12, -1]);  unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_21: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_22: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_8, permute_21);  getitem_8 = permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_13: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_22, [1, 12, 512, 64]);  add_22 = None
    view_47: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_13, [12, 512, 64]);  expand_13 = None
    bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_48, view_47);  view_48 = view_47 = None
    view_49: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_23: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_50: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_2, [1, 512, -1]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_51: "f32[512, 768]" = torch.ops.aten.reshape.default(view_50, [512, 768]);  view_50 = None
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(arg92_1, view_51, permute_24);  arg92_1 = view_51 = permute_24 = None
    view_52: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_52, add_20);  view_52 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_10: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_23, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, mean_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, mean_10);  add_23 = mean_10 = None
    pow_6: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_13, 2);  sub_13 = None
    mean_11: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-07);  mean_11 = None
    sqrt_8: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    div_11: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_14, sqrt_8);  sub_14 = sqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg16_1, div_11);  arg16_1 = div_11 = None
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, arg17_1);  mul_16 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_53: "f32[512, 768]" = torch.ops.aten.reshape.default(add_25, [512, 768])
    permute_25: "f32[768, 3072]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    addmm_7: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg94_1, view_53, permute_25);  arg94_1 = view_53 = permute_25 = None
    view_54: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_17: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_54, 0.5)
    mul_18: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_54, 0.7071067811865476);  view_54 = None
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_26: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_19: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_26);  mul_17 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_55: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_19, [512, 3072]);  mul_19 = None
    permute_26: "f32[3072, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(arg96_1, view_55, permute_26);  arg96_1 = view_55 = permute_26 = None
    view_56: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [1, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_56, add_25);  view_56 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_12: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_27, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, mean_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, mean_12);  add_27 = mean_12 = None
    pow_7: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_15, 2);  sub_15 = None
    mean_13: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-07);  mean_13 = None
    sqrt_9: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    div_12: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_16, sqrt_9);  sub_16 = sqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_20: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg18_1, div_12);  arg18_1 = div_12 = None
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_20, arg19_1);  mul_20 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_57: "f32[512, 768]" = torch.ops.aten.reshape.default(add_29, [512, 768])
    permute_27: "f32[768, 2304]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    mm_3: "f32[512, 2304]" = torch.ops.aten.mm.default(view_57, permute_27);  view_57 = permute_27 = None
    view_58: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_3, [1, 512, 2304]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_59: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_58, [1, 512, 12, -1]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_3 = torch.ops.aten.split.Tensor(permute_28, 64, -1);  permute_28 = None
    getitem_9: "f32[1, 12, 512, 64]" = split_3[0]
    getitem_10: "f32[1, 12, 512, 64]" = split_3[1]
    getitem_11: "f32[1, 12, 512, 64]" = split_3[2];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant6: "f32[]" = self._tensor_constant6
    lift_fresh_copy_6: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
    mul_21: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_6, 1);  lift_fresh_copy_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_3: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant7: "f32[]" = self._tensor_constant7
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_16: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_14: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_15: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_16: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg20_1, 0);  arg20_1 = None
    unsqueeze_17: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_60: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_17, [1, 1, 12, -1]);  unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_29: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_30: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_9, permute_29);  getitem_9 = permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_13: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_13: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_30, full_default_13);  add_30 = full_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_15: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_13, [1, 12, 512, 64]);  div_13 = None
    view_62: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_15, [12, 512, 64]);  expand_15 = None
    permute_31: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
    expand_16: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_31, [1, 12, 64, 512]);  permute_31 = None
    view_63: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_16, [12, 64, 512]);  expand_16 = None
    bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_62, view_63);  view_62 = view_63 = None
    view_64: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_6: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_14, full_default_15, view_64);  full_default_15 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_17: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_6, amax_3);  where_6 = amax_3 = None
    exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_14: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_7: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_14, full_default_16, div_14);  full_default_14 = full_default_16 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_19: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_7, [1, 12, 512, 512]);  where_7 = None
    view_67: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_19, [12, 512, 512]);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_18: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg21_1, 0);  arg21_1 = None
    unsqueeze_19: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 1);  unsqueeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_61: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_19, [1, 1, 12, -1]);  unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_30: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_31: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_11, permute_30);  getitem_11 = permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_18: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_31, [1, 12, 512, 64]);  add_31 = None
    view_66: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_18, [12, 512, 64]);  expand_18 = None
    bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_67, view_66);  view_67 = view_66 = None
    view_68: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_32: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    clone_3: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_69: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_3, [1, 512, -1]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_70: "f32[512, 768]" = torch.ops.aten.reshape.default(view_69, [512, 768]);  view_69 = None
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(arg99_1, view_70, permute_33);  arg99_1 = view_70 = permute_33 = None
    view_71: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_32: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_71, add_29);  view_71 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_14: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_32, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_19: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_14);  add_32 = mean_14 = None
    pow_8: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_18, 2);  sub_18 = None
    mean_15: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_33: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-07);  mean_15 = None
    sqrt_11: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    div_15: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_19, sqrt_11);  sub_19 = sqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_22: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg22_1, div_15);  arg22_1 = div_15 = None
    add_34: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_22, arg23_1);  mul_22 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_72: "f32[512, 768]" = torch.ops.aten.reshape.default(add_34, [512, 768])
    permute_34: "f32[768, 3072]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg101_1, view_72, permute_34);  arg101_1 = view_72 = permute_34 = None
    view_73: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.5)
    mul_24: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476);  view_73 = None
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_35: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_25: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_23, add_35);  mul_23 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_74: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_25, [512, 3072]);  mul_25 = None
    permute_35: "f32[3072, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(arg103_1, view_74, permute_35);  arg103_1 = view_74 = permute_35 = None
    view_75: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_36: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_75, add_34);  view_75 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_16: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_36, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, mean_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, mean_16);  add_36 = mean_16 = None
    pow_9: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_20, 2);  sub_20 = None
    mean_17: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_37: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-07);  mean_17 = None
    sqrt_12: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
    div_16: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_21, sqrt_12);  sub_21 = sqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg24_1, div_16);  arg24_1 = div_16 = None
    add_38: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, arg25_1);  mul_26 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_76: "f32[512, 768]" = torch.ops.aten.reshape.default(add_38, [512, 768])
    permute_36: "f32[768, 2304]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    mm_4: "f32[512, 2304]" = torch.ops.aten.mm.default(view_76, permute_36);  view_76 = permute_36 = None
    view_77: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_4, [1, 512, 2304]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_78: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_77, [1, 512, 12, -1]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_4 = torch.ops.aten.split.Tensor(permute_37, 64, -1);  permute_37 = None
    getitem_12: "f32[1, 12, 512, 64]" = split_4[0]
    getitem_13: "f32[1, 12, 512, 64]" = split_4[1]
    getitem_14: "f32[1, 12, 512, 64]" = split_4[2];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant8: "f32[]" = self._tensor_constant8
    lift_fresh_copy_8: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
    mul_27: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_8, 1);  lift_fresh_copy_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_4: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant9: "f32[]" = self._tensor_constant9
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_20: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_18: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_19: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_20: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg26_1, 0);  arg26_1 = None
    unsqueeze_21: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 1);  unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_79: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_21, [1, 1, 12, -1]);  unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_39: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_12, permute_38);  getitem_12 = permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_17: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_17: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_39, full_default_17);  add_39 = full_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_20: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_17, [1, 12, 512, 64]);  div_17 = None
    view_81: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_20, [12, 512, 64]);  expand_20 = None
    permute_40: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_13, [0, 1, 3, 2]);  getitem_13 = None
    expand_21: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_40, [1, 12, 64, 512]);  permute_40 = None
    view_82: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_21, [12, 64, 512]);  expand_21 = None
    bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_81, view_82);  view_81 = view_82 = None
    view_83: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_8: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_18, full_default_19, view_83);  full_default_19 = view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_22: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_8, amax_4);  where_8 = amax_4 = None
    exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_18: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_9: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_18, full_default_20, div_18);  full_default_18 = full_default_20 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_24: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_9, [1, 12, 512, 512]);  where_9 = None
    view_86: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_24, [12, 512, 512]);  expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_22: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg27_1, 0);  arg27_1 = None
    unsqueeze_23: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 1);  unsqueeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_80: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_23, [1, 1, 12, -1]);  unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_39: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_40: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_14, permute_39);  getitem_14 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_40, [1, 12, 512, 64]);  add_40 = None
    view_85: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_23, [12, 512, 64]);  expand_23 = None
    bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_86, view_85);  view_86 = view_85 = None
    view_87: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_41: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
    clone_4: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_88: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_4, [1, 512, -1]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_89: "f32[512, 768]" = torch.ops.aten.reshape.default(view_88, [512, 768]);  view_88 = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(arg106_1, view_89, permute_42);  arg106_1 = view_89 = permute_42 = None
    view_90: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_90, add_38);  view_90 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_18: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_41, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, mean_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, mean_18);  add_41 = mean_18 = None
    pow_10: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_23, 2);  sub_23 = None
    mean_19: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-07);  mean_19 = None
    sqrt_14: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    div_19: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_24, sqrt_14);  sub_24 = sqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_28: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg28_1, div_19);  arg28_1 = div_19 = None
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_28, arg29_1);  mul_28 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_91: "f32[512, 768]" = torch.ops.aten.reshape.default(add_43, [512, 768])
    permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    addmm_13: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg108_1, view_91, permute_43);  arg108_1 = view_91 = permute_43 = None
    view_92: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_29: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_92, 0.5)
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476);  view_92 = None
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
    add_44: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_29, add_44);  mul_29 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_93: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_31, [512, 3072]);  mul_31 = None
    permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(arg110_1, view_93, permute_44);  arg110_1 = view_93 = permute_44 = None
    view_94: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_94, add_43);  view_94 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_20: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_45, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_26: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, mean_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, mean_20);  add_45 = mean_20 = None
    pow_11: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_25, 2);  sub_25 = None
    mean_21: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-07);  mean_21 = None
    sqrt_15: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    div_20: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_26, sqrt_15);  sub_26 = sqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg30_1, div_20);  arg30_1 = div_20 = None
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, arg31_1);  mul_32 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_95: "f32[512, 768]" = torch.ops.aten.reshape.default(add_47, [512, 768])
    permute_45: "f32[768, 2304]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    mm_5: "f32[512, 2304]" = torch.ops.aten.mm.default(view_95, permute_45);  view_95 = permute_45 = None
    view_96: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_5, [1, 512, 2304]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_97: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_96, [1, 512, 12, -1]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_5 = torch.ops.aten.split.Tensor(permute_46, 64, -1);  permute_46 = None
    getitem_15: "f32[1, 12, 512, 64]" = split_5[0]
    getitem_16: "f32[1, 12, 512, 64]" = split_5[1]
    getitem_17: "f32[1, 12, 512, 64]" = split_5[2];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant10: "f32[]" = self._tensor_constant10
    lift_fresh_copy_10: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
    mul_33: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_10, 1);  lift_fresh_copy_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_5: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant11: "f32[]" = self._tensor_constant11
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_24: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_22: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_23: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_24: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg32_1, 0);  arg32_1 = None
    unsqueeze_25: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 1);  unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_98: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_25, [1, 1, 12, -1]);  unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_47: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_48: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_15, permute_47);  getitem_15 = permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_21: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_21: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_48, full_default_21);  add_48 = full_default_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_25: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_21, [1, 12, 512, 64]);  div_21 = None
    view_100: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_25, [12, 512, 64]);  expand_25 = None
    permute_49: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_16, [0, 1, 3, 2]);  getitem_16 = None
    expand_26: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_49, [1, 12, 64, 512]);  permute_49 = None
    view_101: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_26, [12, 64, 512]);  expand_26 = None
    bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
    view_102: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_10: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_22, full_default_23, view_102);  full_default_23 = view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_27: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_10, amax_5);  where_10 = amax_5 = None
    exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_22: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_11: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_22, full_default_24, div_22);  full_default_22 = full_default_24 = div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_29: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_11, [1, 12, 512, 512]);  where_11 = None
    view_105: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_29, [12, 512, 512]);  expand_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_26: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg33_1, 0);  arg33_1 = None
    unsqueeze_27: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 1);  unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_99: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_27, [1, 1, 12, -1]);  unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_49: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_17, permute_48);  getitem_17 = permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_28: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_49, [1, 12, 512, 64]);  add_49 = None
    view_104: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_28, [12, 512, 64]);  expand_28 = None
    bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_105, view_104);  view_105 = view_104 = None
    view_106: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_50: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    clone_5: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_107: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_5, [1, 512, -1]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 768]" = torch.ops.aten.reshape.default(view_107, [512, 768]);  view_107 = None
    permute_51: "f32[768, 768]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(arg113_1, view_108, permute_51);  arg113_1 = view_108 = permute_51 = None
    view_109: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_50: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_109, add_47);  view_109 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_22: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_50, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, mean_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_28: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, mean_22);  add_50 = mean_22 = None
    pow_12: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_28, 2);  sub_28 = None
    mean_23: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_51: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-07);  mean_23 = None
    sqrt_17: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_51);  add_51 = None
    div_23: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_29, sqrt_17);  sub_29 = sqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg34_1, div_23);  arg34_1 = div_23 = None
    add_52: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, arg35_1);  mul_34 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_110: "f32[512, 768]" = torch.ops.aten.reshape.default(add_52, [512, 768])
    permute_52: "f32[768, 3072]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_16: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg115_1, view_110, permute_52);  arg115_1 = view_110 = permute_52 = None
    view_111: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_35: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_53: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_37: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_35, add_53);  mul_35 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_112: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_37, [512, 3072]);  mul_37 = None
    permute_53: "f32[3072, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(arg117_1, view_112, permute_53);  arg117_1 = view_112 = permute_53 = None
    view_113: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_54: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_113, add_52);  view_113 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_24: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_54, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_31: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_54, mean_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_54, mean_24);  add_54 = mean_24 = None
    pow_13: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_30, 2);  sub_30 = None
    mean_25: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_55: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-07);  mean_25 = None
    sqrt_18: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    div_24: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_31, sqrt_18);  sub_31 = sqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_38: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg36_1, div_24);  arg36_1 = div_24 = None
    add_56: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_38, arg37_1);  mul_38 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_114: "f32[512, 768]" = torch.ops.aten.reshape.default(add_56, [512, 768])
    permute_54: "f32[768, 2304]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    mm_6: "f32[512, 2304]" = torch.ops.aten.mm.default(view_114, permute_54);  view_114 = permute_54 = None
    view_115: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_6, [1, 512, 2304]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_116: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_115, [1, 512, 12, -1]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_55: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_6 = torch.ops.aten.split.Tensor(permute_55, 64, -1);  permute_55 = None
    getitem_18: "f32[1, 12, 512, 64]" = split_6[0]
    getitem_19: "f32[1, 12, 512, 64]" = split_6[1]
    getitem_20: "f32[1, 12, 512, 64]" = split_6[2];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant12: "f32[]" = self._tensor_constant12
    lift_fresh_copy_12: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
    mul_39: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_12, 1);  lift_fresh_copy_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_6: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant13: "f32[]" = self._tensor_constant13
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_28: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_26: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_27: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_28: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg38_1, 0);  arg38_1 = None
    unsqueeze_29: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 1);  unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_117: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_29, [1, 1, 12, -1]);  unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_56: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_57: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_18, permute_56);  getitem_18 = permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_25: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_25: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_57, full_default_25);  add_57 = full_default_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_30: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_25, [1, 12, 512, 64]);  div_25 = None
    view_119: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_30, [12, 512, 64]);  expand_30 = None
    permute_58: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_19, [0, 1, 3, 2]);  getitem_19 = None
    expand_31: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_58, [1, 12, 64, 512]);  permute_58 = None
    view_120: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_31, [12, 64, 512]);  expand_31 = None
    bmm_12: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
    view_121: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [1, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_12: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_26, full_default_27, view_121);  full_default_27 = view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_6: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_12, [-1], True)
    sub_32: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_12, amax_6);  where_12 = amax_6 = None
    exp_6: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_7: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_26: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_13: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_26, full_default_28, div_26);  full_default_26 = full_default_28 = div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_34: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_13, [1, 12, 512, 512]);  where_13 = None
    view_124: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_34, [12, 512, 512]);  expand_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_30: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg39_1, 0);  arg39_1 = None
    unsqueeze_31: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 1);  unsqueeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_118: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_31, [1, 1, 12, -1]);  unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_57: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_58: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_20, permute_57);  getitem_20 = permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_33: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_58, [1, 12, 512, 64]);  add_58 = None
    view_123: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_33, [12, 512, 64]);  expand_33 = None
    bmm_13: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_123);  view_124 = view_123 = None
    view_125: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_59: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
    clone_6: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_126: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_6, [1, 512, -1]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_127: "f32[512, 768]" = torch.ops.aten.reshape.default(view_126, [512, 768]);  view_126 = None
    permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(arg120_1, view_127, permute_60);  arg120_1 = view_127 = permute_60 = None
    view_128: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_128, add_56);  view_128 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_26: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_59, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_34: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, mean_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, mean_26);  add_59 = mean_26 = None
    pow_14: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_33, 2);  sub_33 = None
    mean_27: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-07);  mean_27 = None
    sqrt_20: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    div_27: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_34, sqrt_20);  sub_34 = sqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg40_1, div_27);  arg40_1 = div_27 = None
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_40, arg41_1);  mul_40 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[512, 768]" = torch.ops.aten.reshape.default(add_61, [512, 768])
    permute_61: "f32[768, 3072]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_19: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg122_1, view_129, permute_61);  arg122_1 = view_129 = permute_61 = None
    view_130: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_41: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
    mul_42: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.7071067811865476);  view_130 = None
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_62: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_43: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_41, add_62);  mul_41 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_131: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_43, [512, 3072]);  mul_43 = None
    permute_62: "f32[3072, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(arg124_1, view_131, permute_62);  arg124_1 = view_131 = permute_62 = None
    view_132: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_132, add_61);  view_132 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_28: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_63, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, mean_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_35: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, mean_28);  add_63 = mean_28 = None
    pow_15: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_35, 2);  sub_35 = None
    mean_29: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-07);  mean_29 = None
    sqrt_21: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    div_28: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_36, sqrt_21);  sub_36 = sqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg42_1, div_28);  arg42_1 = div_28 = None
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, arg43_1);  mul_44 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_133: "f32[512, 768]" = torch.ops.aten.reshape.default(add_65, [512, 768])
    permute_63: "f32[768, 2304]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    mm_7: "f32[512, 2304]" = torch.ops.aten.mm.default(view_133, permute_63);  view_133 = permute_63 = None
    view_134: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_7, [1, 512, 2304]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_135: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_134, [1, 512, 12, -1]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_64: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_7 = torch.ops.aten.split.Tensor(permute_64, 64, -1);  permute_64 = None
    getitem_21: "f32[1, 12, 512, 64]" = split_7[0]
    getitem_22: "f32[1, 12, 512, 64]" = split_7[1]
    getitem_23: "f32[1, 12, 512, 64]" = split_7[2];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant14: "f32[]" = self._tensor_constant14
    lift_fresh_copy_14: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
    mul_45: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_14, 1);  lift_fresh_copy_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_7: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant15: "f32[]" = self._tensor_constant15
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_32: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_30: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_31: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_32: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg44_1, 0);  arg44_1 = None
    unsqueeze_33: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 1);  unsqueeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_136: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_33, [1, 1, 12, -1]);  unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_65: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_66: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_21, permute_65);  getitem_21 = permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_29: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_29: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_66, full_default_29);  add_66 = full_default_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_35: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_29, [1, 12, 512, 64]);  div_29 = None
    view_138: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_35, [12, 512, 64]);  expand_35 = None
    permute_67: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_22, [0, 1, 3, 2]);  getitem_22 = None
    expand_36: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_67, [1, 12, 64, 512]);  permute_67 = None
    view_139: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_36, [12, 64, 512]);  expand_36 = None
    bmm_14: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_138, view_139);  view_138 = view_139 = None
    view_140: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [1, 12, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_14: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_30, full_default_31, view_140);  full_default_31 = view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_7: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_14, [-1], True)
    sub_37: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_14, amax_7);  where_14 = amax_7 = None
    exp_7: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_8: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_30: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_15: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_30, full_default_32, div_30);  full_default_30 = full_default_32 = div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_39: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_15, [1, 12, 512, 512]);  where_15 = None
    view_143: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_39, [12, 512, 512]);  expand_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_34: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg45_1, 0);  arg45_1 = None
    unsqueeze_35: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 1);  unsqueeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_137: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_35, [1, 1, 12, -1]);  unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_66: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_67: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_23, permute_66);  getitem_23 = permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_38: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_67, [1, 12, 512, 64]);  add_67 = None
    view_142: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_38, [12, 512, 64]);  expand_38 = None
    bmm_15: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_143, view_142);  view_143 = view_142 = None
    view_144: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_68: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
    clone_7: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_145: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 512, -1]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_146: "f32[512, 768]" = torch.ops.aten.reshape.default(view_145, [512, 768]);  view_145 = None
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(arg127_1, view_146, permute_69);  arg127_1 = view_146 = permute_69 = None
    view_147: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_68: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_147, add_65);  view_147 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_30: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_68, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_39: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, mean_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_38: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, mean_30);  add_68 = mean_30 = None
    pow_16: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_38, 2);  sub_38 = None
    mean_31: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_69: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-07);  mean_31 = None
    sqrt_23: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    div_31: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_39, sqrt_23);  sub_39 = sqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_46: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg46_1, div_31);  arg46_1 = div_31 = None
    add_70: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, arg47_1);  mul_46 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[512, 768]" = torch.ops.aten.reshape.default(add_70, [512, 768])
    permute_70: "f32[768, 3072]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg129_1, view_148, permute_70);  arg129_1 = view_148 = permute_70 = None
    view_149: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_71: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_49: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_71);  mul_47 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_49, [512, 3072]);  mul_49 = None
    permute_71: "f32[3072, 768]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(arg131_1, view_150, permute_71);  arg131_1 = view_150 = permute_71 = None
    view_151: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_72: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_151, add_70);  view_151 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_32: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_72, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_41: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_72, mean_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_72, mean_32);  add_72 = mean_32 = None
    pow_17: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_40, 2);  sub_40 = None
    mean_33: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_73: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-07);  mean_33 = None
    sqrt_24: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    div_32: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_41, sqrt_24);  sub_41 = sqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg48_1, div_32);  arg48_1 = div_32 = None
    add_74: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, arg49_1);  mul_50 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_152: "f32[512, 768]" = torch.ops.aten.reshape.default(add_74, [512, 768])
    permute_72: "f32[768, 2304]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    mm_8: "f32[512, 2304]" = torch.ops.aten.mm.default(view_152, permute_72);  view_152 = permute_72 = None
    view_153: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_8, [1, 512, 2304]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_154: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_153, [1, 512, 12, -1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_73: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_8 = torch.ops.aten.split.Tensor(permute_73, 64, -1);  permute_73 = None
    getitem_24: "f32[1, 12, 512, 64]" = split_8[0]
    getitem_25: "f32[1, 12, 512, 64]" = split_8[1]
    getitem_26: "f32[1, 12, 512, 64]" = split_8[2];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant16: "f32[]" = self._tensor_constant16
    lift_fresh_copy_16: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
    mul_51: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_16, 1);  lift_fresh_copy_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_8: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant17: "f32[]" = self._tensor_constant17
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_36: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_34: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_35: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_36: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg50_1, 0);  arg50_1 = None
    unsqueeze_37: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 1);  unsqueeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_155: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_37, [1, 1, 12, -1]);  unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_74: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_75: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_24, permute_74);  getitem_24 = permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_33: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_33: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_75, full_default_33);  add_75 = full_default_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_40: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_33, [1, 12, 512, 64]);  div_33 = None
    view_157: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_40, [12, 512, 64]);  expand_40 = None
    permute_76: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_25, [0, 1, 3, 2]);  getitem_25 = None
    expand_41: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_76, [1, 12, 64, 512]);  permute_76 = None
    view_158: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_41, [12, 64, 512]);  expand_41 = None
    bmm_16: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_157, view_158);  view_157 = view_158 = None
    view_159: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [1, 12, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_16: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_34, full_default_35, view_159);  full_default_35 = view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_8: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_16, [-1], True)
    sub_42: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_16, amax_8);  where_16 = amax_8 = None
    exp_8: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_9: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_34: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_17: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_34, full_default_36, div_34);  full_default_34 = full_default_36 = div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_44: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_17, [1, 12, 512, 512]);  where_17 = None
    view_162: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_44, [12, 512, 512]);  expand_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_38: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg51_1, 0);  arg51_1 = None
    unsqueeze_39: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 1);  unsqueeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_156: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_39, [1, 1, 12, -1]);  unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_75: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_76: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_26, permute_75);  getitem_26 = permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_43: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_76, [1, 12, 512, 64]);  add_76 = None
    view_161: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_43, [12, 512, 64]);  expand_43 = None
    bmm_17: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_162, view_161);  view_162 = view_161 = None
    view_163: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_77: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    clone_8: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_164: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_8, [1, 512, -1]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_165: "f32[512, 768]" = torch.ops.aten.reshape.default(view_164, [512, 768]);  view_164 = None
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(arg134_1, view_165, permute_78);  arg134_1 = view_165 = permute_78 = None
    view_166: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [1, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_166, add_74);  view_166 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_34: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_77, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, mean_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_43: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, mean_34);  add_77 = mean_34 = None
    pow_18: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_43, 2);  sub_43 = None
    mean_35: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-07);  mean_35 = None
    sqrt_26: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    div_35: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_44, sqrt_26);  sub_44 = sqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg52_1, div_35);  arg52_1 = div_35 = None
    add_79: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, arg53_1);  mul_52 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_167: "f32[512, 768]" = torch.ops.aten.reshape.default(add_79, [512, 768])
    permute_79: "f32[768, 3072]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_25: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg136_1, view_167, permute_79);  arg136_1 = view_167 = permute_79 = None
    view_168: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_25, [1, 512, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_53: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_168, 0.5)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_168, 0.7071067811865476);  view_168 = None
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_80);  mul_53 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_169: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_55, [512, 3072]);  mul_55 = None
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(arg138_1, view_169, permute_80);  arg138_1 = view_169 = permute_80 = None
    view_170: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_170, add_79);  view_170 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_36: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_81, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_36);  add_81 = mean_36 = None
    pow_19: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_45, 2);  sub_45 = None
    mean_37: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-07);  mean_37 = None
    sqrt_27: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    div_36: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_46, sqrt_27);  sub_46 = sqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_56: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg54_1, div_36);  arg54_1 = div_36 = None
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_56, arg55_1);  mul_56 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_171: "f32[512, 768]" = torch.ops.aten.reshape.default(add_83, [512, 768])
    permute_81: "f32[768, 2304]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    mm_9: "f32[512, 2304]" = torch.ops.aten.mm.default(view_171, permute_81);  view_171 = permute_81 = None
    view_172: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_9, [1, 512, 2304]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_173: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_172, [1, 512, 12, -1]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_9 = torch.ops.aten.split.Tensor(permute_82, 64, -1);  permute_82 = None
    getitem_27: "f32[1, 12, 512, 64]" = split_9[0]
    getitem_28: "f32[1, 12, 512, 64]" = split_9[1]
    getitem_29: "f32[1, 12, 512, 64]" = split_9[2];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant18: "f32[]" = self._tensor_constant18
    lift_fresh_copy_18: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
    mul_57: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_18, 1);  lift_fresh_copy_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_9: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant19: "f32[]" = self._tensor_constant19
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_40: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_38: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_39: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_40: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg56_1, 0);  arg56_1 = None
    unsqueeze_41: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 1);  unsqueeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_174: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_41, [1, 1, 12, -1]);  unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_84: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_27, permute_83);  getitem_27 = permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_37: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_37: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_84, full_default_37);  add_84 = full_default_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_45: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_37, [1, 12, 512, 64]);  div_37 = None
    view_176: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_45, [12, 512, 64]);  expand_45 = None
    permute_85: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
    expand_46: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_85, [1, 12, 64, 512]);  permute_85 = None
    view_177: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_46, [12, 64, 512]);  expand_46 = None
    bmm_18: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_176, view_177);  view_176 = view_177 = None
    view_178: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [1, 12, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_18: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_38, full_default_39, view_178);  full_default_39 = view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_9: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_18, [-1], True)
    sub_47: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_18, amax_9);  where_18 = amax_9 = None
    exp_9: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
    sum_10: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_38: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_19: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_38, full_default_40, div_38);  full_default_38 = full_default_40 = div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_49: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_19, [1, 12, 512, 512]);  where_19 = None
    view_181: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_49, [12, 512, 512]);  expand_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_42: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg57_1, 0);  arg57_1 = None
    unsqueeze_43: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 1);  unsqueeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_175: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_43, [1, 1, 12, -1]);  unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_84: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_85: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_29, permute_84);  getitem_29 = permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_48: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_85, [1, 12, 512, 64]);  add_85 = None
    view_180: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_48, [12, 512, 64]);  expand_48 = None
    bmm_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_181, view_180);  view_181 = view_180 = None
    view_182: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_86: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    clone_9: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_183: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_9, [1, 512, -1]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_184: "f32[512, 768]" = torch.ops.aten.reshape.default(view_183, [512, 768]);  view_183 = None
    permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(arg141_1, view_184, permute_87);  arg141_1 = view_184 = permute_87 = None
    view_185: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_86: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_185, add_83);  view_185 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_38: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_86, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_86, mean_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_86, mean_38);  add_86 = mean_38 = None
    pow_20: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_48, 2);  sub_48 = None
    mean_39: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_87: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-07);  mean_39 = None
    sqrt_29: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    div_39: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_49, sqrt_29);  sub_49 = sqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg58_1, div_39);  arg58_1 = div_39 = None
    add_88: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, arg59_1);  mul_58 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_186: "f32[512, 768]" = torch.ops.aten.reshape.default(add_88, [512, 768])
    permute_88: "f32[768, 3072]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_28: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg143_1, view_186, permute_88);  arg143_1 = view_186 = permute_88 = None
    view_187: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_59: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_187, 0.5)
    mul_60: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_187, 0.7071067811865476);  view_187 = None
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_89: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_61: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_59, add_89);  mul_59 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_188: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_61, [512, 3072]);  mul_61 = None
    permute_89: "f32[3072, 768]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_29: "f32[512, 768]" = torch.ops.aten.addmm.default(arg145_1, view_188, permute_89);  arg145_1 = view_188 = permute_89 = None
    view_189: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 512, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_90: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_189, add_88);  view_189 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_40: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_90, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_51: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_90, mean_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_50: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_90, mean_40);  add_90 = mean_40 = None
    pow_21: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_50, 2);  sub_50 = None
    mean_41: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_91: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-07);  mean_41 = None
    sqrt_30: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    div_40: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_51, sqrt_30);  sub_51 = sqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_62: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg60_1, div_40);  arg60_1 = div_40 = None
    add_92: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_62, arg61_1);  mul_62 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_190: "f32[512, 768]" = torch.ops.aten.reshape.default(add_92, [512, 768])
    permute_90: "f32[768, 2304]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    mm_10: "f32[512, 2304]" = torch.ops.aten.mm.default(view_190, permute_90);  view_190 = permute_90 = None
    view_191: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_10, [1, 512, 2304]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_192: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_191, [1, 512, 12, -1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_91: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_10 = torch.ops.aten.split.Tensor(permute_91, 64, -1);  permute_91 = None
    getitem_30: "f32[1, 12, 512, 64]" = split_10[0]
    getitem_31: "f32[1, 12, 512, 64]" = split_10[1]
    getitem_32: "f32[1, 12, 512, 64]" = split_10[2];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant20: "f32[]" = self._tensor_constant20
    lift_fresh_copy_20: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
    mul_63: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_20, 1);  lift_fresh_copy_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_10: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant21: "f32[]" = self._tensor_constant21
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_44: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_42: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_43: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_44: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg62_1, 0);  arg62_1 = None
    unsqueeze_45: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, 1);  unsqueeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_193: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_45, [1, 1, 12, -1]);  unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_93: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_30, permute_92);  getitem_30 = permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_41: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_41: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_93, full_default_41);  add_93 = full_default_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_50: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_41, [1, 12, 512, 64]);  div_41 = None
    view_195: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_50, [12, 512, 64]);  expand_50 = None
    permute_94: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_31, [0, 1, 3, 2]);  getitem_31 = None
    expand_51: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_94, [1, 12, 64, 512]);  permute_94 = None
    view_196: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_51, [12, 64, 512]);  expand_51 = None
    bmm_20: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_195, view_196);  view_195 = view_196 = None
    view_197: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [1, 12, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_20: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_42, full_default_43, view_197);  full_default_43 = view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_10: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_20, [-1], True)
    sub_52: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_20, amax_10);  where_20 = amax_10 = None
    exp_10: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_11: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_42: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_21: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_42, full_default_44, div_42);  full_default_42 = full_default_44 = div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_54: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_21, [1, 12, 512, 512]);  where_21 = None
    view_200: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_54, [12, 512, 512]);  expand_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_46: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg63_1, 0);  arg63_1 = None
    unsqueeze_47: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 1);  unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_194: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_47, [1, 1, 12, -1]);  unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_94: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_32, permute_93);  getitem_32 = permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_53: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_94, [1, 12, 512, 64]);  add_94 = None
    view_199: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_53, [12, 512, 64]);  expand_53 = None
    bmm_21: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_200, view_199);  view_200 = view_199 = None
    view_201: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_202: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_10, [1, 512, -1]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_203: "f32[512, 768]" = torch.ops.aten.reshape.default(view_202, [512, 768]);  view_202 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_30: "f32[512, 768]" = torch.ops.aten.addmm.default(arg148_1, view_203, permute_96);  arg148_1 = view_203 = permute_96 = None
    view_204: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [1, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_204, add_92);  view_204 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_42: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_95, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_54: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, mean_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, mean_42);  add_95 = mean_42 = None
    pow_22: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_53, 2);  sub_53 = None
    mean_43: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_96: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_43, 1e-07);  mean_43 = None
    sqrt_32: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    div_43: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_54, sqrt_32);  sub_54 = sqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_64: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg64_1, div_43);  arg64_1 = div_43 = None
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_64, arg65_1);  mul_64 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_205: "f32[512, 768]" = torch.ops.aten.reshape.default(add_97, [512, 768])
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_31: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg150_1, view_205, permute_97);  arg150_1 = view_205 = permute_97 = None
    view_206: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_31, [1, 512, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_65: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.5)
    mul_66: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476);  view_206 = None
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_66);  mul_66 = None
    add_98: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_67: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_65, add_98);  mul_65 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_207: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_67, [512, 3072]);  mul_67 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(arg152_1, view_207, permute_98);  arg152_1 = view_207 = permute_98 = None
    view_208: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [1, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_208, add_97);  view_208 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_44: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_99, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, mean_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, mean_44);  add_99 = mean_44 = None
    pow_23: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_55, 2);  sub_55 = None
    mean_45: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_100: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_45, 1e-07);  mean_45 = None
    sqrt_33: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    div_44: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_56, sqrt_33);  sub_56 = sqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_68: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg66_1, div_44);  arg66_1 = div_44 = None
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, arg67_1);  mul_68 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_209: "f32[512, 768]" = torch.ops.aten.reshape.default(add_101, [512, 768])
    permute_99: "f32[768, 2304]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    mm_11: "f32[512, 2304]" = torch.ops.aten.mm.default(view_209, permute_99);  view_209 = permute_99 = None
    view_210: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_11, [1, 512, 2304]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_211: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_210, [1, 512, 12, -1]);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_100: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_11 = torch.ops.aten.split.Tensor(permute_100, 64, -1);  permute_100 = None
    getitem_33: "f32[1, 12, 512, 64]" = split_11[0]
    getitem_34: "f32[1, 12, 512, 64]" = split_11[1]
    getitem_35: "f32[1, 12, 512, 64]" = split_11[2];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant22: "f32[]" = self._tensor_constant22
    lift_fresh_copy_22: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
    mul_69: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_22, 1);  lift_fresh_copy_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_11: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bool);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant23: "f32[]" = self._tensor_constant23
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    full_default_48: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:119, code: rmask = ~(mask.to(torch.bool))
    full_default_46: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    full_default_47: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_48: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg68_1, 0);  arg68_1 = None
    unsqueeze_49: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 1);  unsqueeze_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_212: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_49, [1, 1, 12, -1]);  unsqueeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_102: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_33, permute_101);  getitem_33 = permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_45: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_45: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_102, full_default_45);  add_102 = full_default_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    expand_55: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_45, [1, 12, 512, 64]);  div_45 = None
    view_214: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_55, [12, 512, 64]);  expand_55 = None
    permute_103: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_34, [0, 1, 3, 2]);  getitem_34 = None
    expand_56: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_103, [1, 12, 64, 512]);  permute_103 = None
    view_215: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_56, [12, 64, 512]);  expand_56 = None
    bmm_22: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_214, view_215);  view_214 = view_215 = None
    view_216: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [1, 12, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:121, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    where_22: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_46, full_default_47, view_216);  full_default_47 = view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:122, code: output = torch.softmax(output, self.dim)
    amax_11: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_22, [-1], True)
    sub_57: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_22, amax_11);  where_22 = amax_11 = None
    exp_11: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_12: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_46: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:123, code: output.masked_fill_(rmask, 0)
    where_23: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_46, full_default_48, div_46);  full_default_46 = full_default_48 = div_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_59: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_23, [1, 12, 512, 512]);  where_23 = None
    view_219: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_59, [12, 512, 512]);  expand_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_50: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(arg69_1, 0);  arg69_1 = None
    unsqueeze_51: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 1);  unsqueeze_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_213: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_51, [1, 1, 12, -1]);  unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_102: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_103: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_35, permute_102);  getitem_35 = permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_58: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_103, [1, 12, 512, 64]);  add_103 = None
    view_218: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_58, [12, 512, 64]);  expand_58 = None
    bmm_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_219, view_218);  view_219 = view_218 = None
    view_220: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_104: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
    clone_11: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_221: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_11, [1, 512, -1]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_222: "f32[512, 768]" = torch.ops.aten.reshape.default(view_221, [512, 768]);  view_221 = None
    permute_105: "f32[768, 768]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_33: "f32[512, 768]" = torch.ops.aten.addmm.default(arg155_1, view_222, permute_105);  arg155_1 = view_222 = permute_105 = None
    view_223: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_104: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_223, add_101);  view_223 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_46: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_104, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_104, mean_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_58: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_104, mean_46);  add_104 = mean_46 = None
    pow_24: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_58, 2);  sub_58 = None
    mean_47: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_105: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_47, 1e-07);  mean_47 = None
    sqrt_35: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    div_47: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_59, sqrt_35);  sub_59 = sqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_70: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg70_1, div_47);  arg70_1 = div_47 = None
    add_106: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_70, arg71_1);  mul_70 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_224: "f32[512, 768]" = torch.ops.aten.reshape.default(add_106, [512, 768])
    permute_106: "f32[768, 3072]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_34: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg157_1, view_224, permute_106);  arg157_1 = view_224 = permute_106 = None
    view_225: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_225, 0.5)
    mul_72: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_225, 0.7071067811865476);  view_225 = None
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_107: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_73: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_71, add_107);  mul_71 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_226: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_73, [512, 3072]);  mul_73 = None
    permute_107: "f32[3072, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_35: "f32[512, 768]" = torch.ops.aten.addmm.default(arg159_1, view_226, permute_107);  arg159_1 = view_226 = permute_107 = None
    view_227: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_108: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_227, add_106);  view_227 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_48: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_108, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_108, mean_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_108, mean_48);  add_108 = mean_48 = None
    pow_25: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_60, 2);  sub_60 = None
    mean_49: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_109: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_49, 1e-07);  mean_49 = None
    sqrt_36: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
    div_48: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_61, sqrt_36);  sub_61 = sqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg72_1, div_48);  arg72_1 = div_48 = None
    add_110: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, arg73_1);  mul_74 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1116, code: hidden_states = self.dense(hidden_states)
    view_228: "f32[512, 768]" = torch.ops.aten.reshape.default(add_110, [512, 768]);  add_110 = None
    permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_36: "f32[512, 768]" = torch.ops.aten.addmm.default(arg161_1, view_228, permute_108);  arg161_1 = view_228 = permute_108 = None
    view_229: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [1, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_229, 0.5)
    mul_76: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_229, 0.7071067811865476);  view_229 = None
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_77: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_75, add_111);  mul_75 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1118, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(mul_77, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1]" = var_mean[0]
    getitem_37: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1089, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_233: "i64[512]" = torch.ops.aten.reshape.default(arg168_1, [-1]);  arg168_1 = None
    ne_1: "b8[512]" = torch.ops.aten.ne.Scalar(view_233, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1118, code: hidden_states = self.LayerNorm(hidden_states)
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_77, getitem_37);  mul_77 = getitem_37 = None
    add_112: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-07);  getitem_36 = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    mul_78: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt);  sub_62 = rsqrt = None
    mul_79: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg162_1);  mul_78 = arg162_1 = None
    add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, arg163_1);  mul_79 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1139, code: hidden_states = self.decoder(hidden_states)
    view_230: "f32[512, 768]" = torch.ops.aten.reshape.default(add_113, [512, 768]);  add_113 = None
    permute_109: "f32[768, 50265]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_37: "f32[512, 50265]" = torch.ops.aten.addmm.default(arg165_1, view_230, permute_109);  arg165_1 = view_230 = permute_109 = None
    view_231: "f32[1, 512, 50265]" = torch.ops.aten.reshape.default(addmm_37, [1, 512, 50265]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1089, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_232: "f32[512, 50265]" = torch.ops.aten.reshape.default(view_231, [-1, 50265])
    amax_12: "f32[512, 1]" = torch.ops.aten.amax.default(view_232, [1], True)
    sub_63: "f32[512, 50265]" = torch.ops.aten.sub.Tensor(view_232, amax_12);  view_232 = amax_12 = None
    exp_12: "f32[512, 50265]" = torch.ops.aten.exp.default(sub_63)
    sum_13: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_64: "f32[512, 50265]" = torch.ops.aten.sub.Tensor(sub_63, log);  sub_63 = log = None
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_233, -100)
    full_default_49: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_24: "i64[512]" = torch.ops.aten.where.self(ne, view_233, full_default_49);  ne = full_default_49 = None
    unsqueeze_52: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_64, 1, unsqueeze_52);  sub_64 = unsqueeze_52 = None
    squeeze_1: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
    full_default_50: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_25: "f32[512]" = torch.ops.aten.where.self(ne_1, neg, full_default_50);  ne_1 = neg = full_default_50 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
    ne_2: "b8[512]" = torch.ops.aten.ne.Scalar(view_233, -100);  view_233 = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_12: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    div_49: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_12);  sum_15 = convert_element_type_12 = None
    return (div_49, view_231)
    