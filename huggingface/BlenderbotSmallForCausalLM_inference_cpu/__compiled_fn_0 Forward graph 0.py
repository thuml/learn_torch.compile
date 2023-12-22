from __future__ import annotations



def forward(self, arg0_1: "f32[512, 512]", arg1_1: "f32[50265, 512]", arg2_1: "f32[512]", arg3_1: "f32[512]", arg4_1: "f32[512, 512]", arg5_1: "f32[512]", arg6_1: "f32[512, 512]", arg7_1: "f32[512]", arg8_1: "f32[512, 512]", arg9_1: "f32[512]", arg10_1: "f32[512, 512]", arg11_1: "f32[512]", arg12_1: "f32[512]", arg13_1: "f32[512]", arg14_1: "f32[2048, 512]", arg15_1: "f32[2048]", arg16_1: "f32[512, 2048]", arg17_1: "f32[512]", arg18_1: "f32[512]", arg19_1: "f32[512]", arg20_1: "f32[512, 512]", arg21_1: "f32[512]", arg22_1: "f32[512, 512]", arg23_1: "f32[512]", arg24_1: "f32[512, 512]", arg25_1: "f32[512]", arg26_1: "f32[512, 512]", arg27_1: "f32[512]", arg28_1: "f32[512]", arg29_1: "f32[512]", arg30_1: "f32[2048, 512]", arg31_1: "f32[2048]", arg32_1: "f32[512, 2048]", arg33_1: "f32[512]", arg34_1: "f32[512]", arg35_1: "f32[512]", arg36_1: "f32[512, 512]", arg37_1: "f32[512]", arg38_1: "f32[512, 512]", arg39_1: "f32[512]", arg40_1: "f32[512, 512]", arg41_1: "f32[512]", arg42_1: "f32[512, 512]", arg43_1: "f32[512]", arg44_1: "f32[512]", arg45_1: "f32[512]", arg46_1: "f32[2048, 512]", arg47_1: "f32[2048]", arg48_1: "f32[512, 2048]", arg49_1: "f32[512]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[512, 512]", arg53_1: "f32[512]", arg54_1: "f32[512, 512]", arg55_1: "f32[512]", arg56_1: "f32[512, 512]", arg57_1: "f32[512]", arg58_1: "f32[512, 512]", arg59_1: "f32[512]", arg60_1: "f32[512]", arg61_1: "f32[512]", arg62_1: "f32[2048, 512]", arg63_1: "f32[2048]", arg64_1: "f32[512, 2048]", arg65_1: "f32[512]", arg66_1: "f32[512]", arg67_1: "f32[512]", arg68_1: "f32[512, 512]", arg69_1: "f32[512]", arg70_1: "f32[512, 512]", arg71_1: "f32[512]", arg72_1: "f32[512, 512]", arg73_1: "f32[512]", arg74_1: "f32[512, 512]", arg75_1: "f32[512]", arg76_1: "f32[512]", arg77_1: "f32[512]", arg78_1: "f32[2048, 512]", arg79_1: "f32[2048]", arg80_1: "f32[512, 2048]", arg81_1: "f32[512]", arg82_1: "f32[512]", arg83_1: "f32[512]", arg84_1: "f32[512, 512]", arg85_1: "f32[512]", arg86_1: "f32[512, 512]", arg87_1: "f32[512]", arg88_1: "f32[512, 512]", arg89_1: "f32[512]", arg90_1: "f32[512, 512]", arg91_1: "f32[512]", arg92_1: "f32[512]", arg93_1: "f32[512]", arg94_1: "f32[2048, 512]", arg95_1: "f32[2048]", arg96_1: "f32[512, 2048]", arg97_1: "f32[512]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[512, 512]", arg101_1: "f32[512]", arg102_1: "f32[512, 512]", arg103_1: "f32[512]", arg104_1: "f32[512, 512]", arg105_1: "f32[512]", arg106_1: "f32[512, 512]", arg107_1: "f32[512]", arg108_1: "f32[512]", arg109_1: "f32[512]", arg110_1: "f32[2048, 512]", arg111_1: "f32[2048]", arg112_1: "f32[512, 2048]", arg113_1: "f32[512]", arg114_1: "f32[512]", arg115_1: "f32[512]", arg116_1: "f32[512, 512]", arg117_1: "f32[512]", arg118_1: "f32[512, 512]", arg119_1: "f32[512]", arg120_1: "f32[512, 512]", arg121_1: "f32[512]", arg122_1: "f32[512, 512]", arg123_1: "f32[512]", arg124_1: "f32[512]", arg125_1: "f32[512]", arg126_1: "f32[2048, 512]", arg127_1: "f32[2048]", arg128_1: "f32[512, 2048]", arg129_1: "f32[512]", arg130_1: "f32[512]", arg131_1: "f32[512]", arg132_1: "f32[50265, 512]", arg133_1: "i64[1, 128]", arg134_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:969, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.view.default(arg133_1, [-1, 128]);  arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:979, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg1_1, view, 0);  arg1_1 = view = None
    mul: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:82, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:83, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:84, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add: "i64[128]" = torch.ops.aten.add.Tensor(iota, 1)
    view_1: "i64[128, 1]" = torch.ops.aten.view.default(add, [128, 1]);  add = None
    lt: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[128, 128]" = torch.ops.aten.where.self(lt, scalar_tensor, full);  lt = scalar_tensor = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:119, code: positions = torch.arange(
    iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[128, 512]" = torch.ops.aten.embedding.default(arg0_1, iota_1);  arg0_1 = iota_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:994, code: inputs_embeds = self.layernorm_embedding(inputs_embeds)
    var_mean = torch.ops.aten.var_mean.correction(mul, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(mul, getitem_1);  mul = getitem_1 = None
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_2: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
    add_2: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:995, code: hidden_states = inputs_embeds + positions
    add_3: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_2, embedding_1);  add_2 = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:997, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone: "f32[1, 128, 512]" = torch.ops.aten.clone.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_2: "f32[128, 512]" = torch.ops.aten.view.default(clone, [128, 512])
    permute: "f32[512, 512]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm: "f32[128, 512]" = torch.ops.aten.addmm.default(arg5_1, view_2, permute);  arg5_1 = view_2 = permute = None
    view_3: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm, [1, 128, 512]);  addmm = None
    mul_3: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_3, 0.1767766952966369);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_4: "f32[128, 512]" = torch.ops.aten.view.default(clone, [128, 512])
    permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm_1: "f32[128, 512]" = torch.ops.aten.addmm.default(arg7_1, view_4, permute_1);  arg7_1 = view_4 = permute_1 = None
    view_5: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_1, [1, 128, 512]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_5, [1, -1, 16, 32]);  view_5 = None
    permute_2: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    clone_1: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_7: "f32[128, 512]" = torch.ops.aten.view.default(clone, [128, 512])
    permute_3: "f32[512, 512]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    addmm_2: "f32[128, 512]" = torch.ops.aten.addmm.default(arg9_1, view_7, permute_3);  arg9_1 = view_7 = permute_3 = None
    view_8: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_2, [1, 128, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_8, [1, -1, 16, 32]);  view_8 = None
    permute_4: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_2: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_10: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(mul_3, [1, 128, 16, 32]);  mul_3 = None
    permute_5: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_3: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_11: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_3, [16, -1, 32]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_12: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_1, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_13: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_2, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_14: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, [1, 16, 128, 128]);  bmm = None
    unsqueeze_2: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    slice_3: "f32[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 2, 0, 9223372036854775807);  unsqueeze_3 = None
    slice_4: "f32[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 9223372036854775807);  slice_3 = None
    expand_1: "f32[1, 1, 128, 128]" = torch.ops.aten.expand.default(slice_4, [1, 1, 128, 128]);  slice_4 = None
    add_4: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_14, expand_1);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_15: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_4, [16, 128, 128]);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_15, [-1], True)
    sub_1: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
    exp: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_4: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(clone_4, view_13);  clone_4 = view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_16: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_1, [1, 16, 128, 32]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_17: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_5, [1, 128, 512]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_18: "f32[128, 512]" = torch.ops.aten.view.default(view_17, [128, 512]);  view_17 = None
    permute_8: "f32[512, 512]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm_3: "f32[128, 512]" = torch.ops.aten.addmm.default(arg11_1, view_18, permute_8);  arg11_1 = view_18 = permute_8 = None
    view_19: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_3, [1, 128, 512]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_6: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_19);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_5: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone, clone_6);  clone = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_2: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
    mul_4: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_4, arg12_1);  mul_4 = arg12_1 = None
    add_7: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_5, arg13_1);  mul_5 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_20: "f32[128, 512]" = torch.ops.aten.view.default(add_7, [128, 512])
    permute_9: "f32[512, 2048]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    addmm_4: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg15_1, view_20, permute_9);  arg15_1 = view_20 = permute_9 = None
    view_21: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_4, [1, 128, 2048]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_7: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
    erf: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_8: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:450, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_7: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_22: "f32[128, 2048]" = torch.ops.aten.view.default(clone_7, [128, 2048]);  clone_7 = None
    permute_10: "f32[2048, 512]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_5: "f32[128, 512]" = torch.ops.aten.addmm.default(arg17_1, view_22, permute_10);  arg17_1 = view_22 = permute_10 = None
    view_23: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_5, [1, 128, 512]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_8: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_23);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_9: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_7, clone_8);  add_7 = clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
    mul_9: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_10: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_9, arg18_1);  mul_9 = arg18_1 = None
    add_11: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_10, arg19_1);  mul_10 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_24: "f32[128, 512]" = torch.ops.aten.view.default(add_11, [128, 512])
    permute_11: "f32[512, 512]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_6: "f32[128, 512]" = torch.ops.aten.addmm.default(arg21_1, view_24, permute_11);  arg21_1 = view_24 = permute_11 = None
    view_25: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_6, [1, 128, 512]);  addmm_6 = None
    mul_11: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_25, 0.1767766952966369);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_26: "f32[128, 512]" = torch.ops.aten.view.default(add_11, [128, 512])
    permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    addmm_7: "f32[128, 512]" = torch.ops.aten.addmm.default(arg23_1, view_26, permute_12);  arg23_1 = view_26 = permute_12 = None
    view_27: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_7, [1, 128, 512]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_27, [1, -1, 16, 32]);  view_27 = None
    permute_13: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_9: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_29: "f32[128, 512]" = torch.ops.aten.view.default(add_11, [128, 512])
    permute_14: "f32[512, 512]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_8: "f32[128, 512]" = torch.ops.aten.addmm.default(arg25_1, view_29, permute_14);  arg25_1 = view_29 = permute_14 = None
    view_30: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_8, [1, 128, 512]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_31: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_30, [1, -1, 16, 32]);  view_30 = None
    permute_15: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    clone_10: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_32: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(mul_11, [1, 128, 16, 32]);  mul_11 = None
    permute_16: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    clone_11: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_33: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_11, [16, -1, 32]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_34: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_9, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_35: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_10, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_33, permute_17);  view_33 = permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_36: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, [1, 16, 128, 128]);  bmm_2 = None
    add_12: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_36, expand_1);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_37: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_12, [16, 128, 128]);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_37, [-1], True)
    sub_4: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_37, amax_1);  view_37 = amax_1 = None
    exp_1: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_12: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(clone_12, view_35);  clone_12 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_38: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_3, [1, 16, 128, 32]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_39: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_13, [1, 128, 512]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_40: "f32[128, 512]" = torch.ops.aten.view.default(view_39, [128, 512]);  view_39 = None
    permute_19: "f32[512, 512]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_9: "f32[128, 512]" = torch.ops.aten.addmm.default(arg27_1, view_40, permute_19);  arg27_1 = view_40 = permute_19 = None
    view_41: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_9, [1, 128, 512]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_14: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_41);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_13: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_11, clone_14);  add_11 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_14: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_5: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
    mul_12: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_13: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_12, arg28_1);  mul_12 = arg28_1 = None
    add_15: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_13, arg29_1);  mul_13 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_42: "f32[128, 512]" = torch.ops.aten.view.default(add_15, [128, 512])
    permute_20: "f32[512, 2048]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_10: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg31_1, view_42, permute_20);  arg31_1 = view_42 = permute_20 = None
    view_43: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_10, [1, 128, 2048]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_15: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_1: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_16: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_14, add_16);  mul_14 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:450, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_15: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_44: "f32[128, 2048]" = torch.ops.aten.view.default(clone_15, [128, 2048]);  clone_15 = None
    permute_21: "f32[2048, 512]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    addmm_11: "f32[128, 512]" = torch.ops.aten.addmm.default(arg33_1, view_44, permute_21);  arg33_1 = view_44 = permute_21 = None
    view_45: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_11, [1, 128, 512]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_16: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_45);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_17: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_15, clone_16);  add_15 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_6: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
    mul_17: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_18: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_17, arg34_1);  mul_17 = arg34_1 = None
    add_19: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_18, arg35_1);  mul_18 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_46: "f32[128, 512]" = torch.ops.aten.view.default(add_19, [128, 512])
    permute_22: "f32[512, 512]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_12: "f32[128, 512]" = torch.ops.aten.addmm.default(arg37_1, view_46, permute_22);  arg37_1 = view_46 = permute_22 = None
    view_47: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_12, [1, 128, 512]);  addmm_12 = None
    mul_19: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_47, 0.1767766952966369);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_48: "f32[128, 512]" = torch.ops.aten.view.default(add_19, [128, 512])
    permute_23: "f32[512, 512]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    addmm_13: "f32[128, 512]" = torch.ops.aten.addmm.default(arg39_1, view_48, permute_23);  arg39_1 = view_48 = permute_23 = None
    view_49: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_13, [1, 128, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_49, [1, -1, 16, 32]);  view_49 = None
    permute_24: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_17: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_51: "f32[128, 512]" = torch.ops.aten.view.default(add_19, [128, 512])
    permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    addmm_14: "f32[128, 512]" = torch.ops.aten.addmm.default(arg41_1, view_51, permute_25);  arg41_1 = view_51 = permute_25 = None
    view_52: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_14, [1, 128, 512]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_53: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_52, [1, -1, 16, 32]);  view_52 = None
    permute_26: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    clone_18: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_54: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(mul_19, [1, 128, 16, 32]);  mul_19 = None
    permute_27: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    clone_19: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_55: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_19, [16, -1, 32]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_56: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_17, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_57: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_18, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_55, permute_28);  view_55 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_58: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, [1, 16, 128, 128]);  bmm_4 = None
    add_20: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_58, expand_1);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_59: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_20, [16, 128, 128]);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_59, [-1], True)
    sub_7: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_59, amax_2);  view_59 = amax_2 = None
    exp_2: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_20: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(clone_20, view_57);  clone_20 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_60: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_5, [1, 16, 128, 32]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_61: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_21, [1, 128, 512]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_62: "f32[128, 512]" = torch.ops.aten.view.default(view_61, [128, 512]);  view_61 = None
    permute_30: "f32[512, 512]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_15: "f32[128, 512]" = torch.ops.aten.addmm.default(arg43_1, view_62, permute_30);  arg43_1 = view_62 = permute_30 = None
    view_63: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_15, [1, 128, 512]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_22: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_63);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_21: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_19, clone_22);  add_19 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_22: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_8: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
    mul_20: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_21: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_20, arg44_1);  mul_20 = arg44_1 = None
    add_23: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_21, arg45_1);  mul_21 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_64: "f32[128, 512]" = torch.ops.aten.view.default(add_23, [128, 512])
    permute_31: "f32[512, 2048]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_16: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg47_1, view_64, permute_31);  arg47_1 = view_64 = permute_31 = None
    view_65: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_16, [1, 128, 2048]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    mul_23: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
    erf_2: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_24: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_22, add_24);  mul_22 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:450, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_23: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_66: "f32[128, 2048]" = torch.ops.aten.view.default(clone_23, [128, 2048]);  clone_23 = None
    permute_32: "f32[2048, 512]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_17: "f32[128, 512]" = torch.ops.aten.addmm.default(arg49_1, view_66, permute_32);  arg49_1 = view_66 = permute_32 = None
    view_67: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_17, [1, 128, 512]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_24: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_25: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_23, clone_24);  add_23 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_26: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_9: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
    mul_25: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_26: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_25, arg50_1);  mul_25 = arg50_1 = None
    add_27: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_26, arg51_1);  mul_26 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_68: "f32[128, 512]" = torch.ops.aten.view.default(add_27, [128, 512])
    permute_33: "f32[512, 512]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_18: "f32[128, 512]" = torch.ops.aten.addmm.default(arg53_1, view_68, permute_33);  arg53_1 = view_68 = permute_33 = None
    view_69: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_18, [1, 128, 512]);  addmm_18 = None
    mul_27: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_69, 0.1767766952966369);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_70: "f32[128, 512]" = torch.ops.aten.view.default(add_27, [128, 512])
    permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_19: "f32[128, 512]" = torch.ops.aten.addmm.default(arg55_1, view_70, permute_34);  arg55_1 = view_70 = permute_34 = None
    view_71: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_19, [1, 128, 512]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_72: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_71, [1, -1, 16, 32]);  view_71 = None
    permute_35: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    clone_25: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_73: "f32[128, 512]" = torch.ops.aten.view.default(add_27, [128, 512])
    permute_36: "f32[512, 512]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_20: "f32[128, 512]" = torch.ops.aten.addmm.default(arg57_1, view_73, permute_36);  arg57_1 = view_73 = permute_36 = None
    view_74: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_20, [1, 128, 512]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_75: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_74, [1, -1, 16, 32]);  view_74 = None
    permute_37: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    clone_26: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_76: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(mul_27, [1, 128, 16, 32]);  mul_27 = None
    permute_38: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    clone_27: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_77: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_27, [16, -1, 32]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_78: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_25, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_79: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_26, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_77, permute_39);  view_77 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_80: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, [1, 16, 128, 128]);  bmm_6 = None
    add_28: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_80, expand_1);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_81: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_28, [16, 128, 128]);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_81, [-1], True)
    sub_10: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_81, amax_3);  view_81 = amax_3 = None
    exp_3: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_28: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(clone_28, view_79);  clone_28 = view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_82: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_7, [1, 16, 128, 32]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_83: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_29, [1, 128, 512]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_84: "f32[128, 512]" = torch.ops.aten.view.default(view_83, [128, 512]);  view_83 = None
    permute_41: "f32[512, 512]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_21: "f32[128, 512]" = torch.ops.aten.addmm.default(arg59_1, view_84, permute_41);  arg59_1 = view_84 = permute_41 = None
    view_85: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_21, [1, 128, 512]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_30: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_29: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_27, clone_30);  add_27 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_30: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_11: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
    mul_28: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_29: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_28, arg60_1);  mul_28 = arg60_1 = None
    add_31: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_29, arg61_1);  mul_29 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_86: "f32[128, 512]" = torch.ops.aten.view.default(add_31, [128, 512])
    permute_42: "f32[512, 2048]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    addmm_22: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg63_1, view_86, permute_42);  arg63_1 = view_86 = permute_42 = None
    view_87: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_22, [1, 128, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_31: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
    erf_3: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_32: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_30, add_32);  mul_30 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:450, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_31: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_88: "f32[128, 2048]" = torch.ops.aten.view.default(clone_31, [128, 2048]);  clone_31 = None
    permute_43: "f32[2048, 512]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_23: "f32[128, 512]" = torch.ops.aten.addmm.default(arg65_1, view_88, permute_43);  arg65_1 = view_88 = permute_43 = None
    view_89: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_23, [1, 128, 512]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_32: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_89);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_33: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_31, clone_32);  add_31 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    add_34: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_12: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
    mul_33: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_34: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_33, arg66_1);  mul_33 = arg66_1 = None
    add_35: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_34, arg67_1);  mul_34 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_90: "f32[128, 512]" = torch.ops.aten.view.default(add_35, [128, 512])
    permute_44: "f32[512, 512]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm_24: "f32[128, 512]" = torch.ops.aten.addmm.default(arg69_1, view_90, permute_44);  arg69_1 = view_90 = permute_44 = None
    view_91: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_24, [1, 128, 512]);  addmm_24 = None
    mul_35: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_91, 0.1767766952966369);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_92: "f32[128, 512]" = torch.ops.aten.view.default(add_35, [128, 512])
    permute_45: "f32[512, 512]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_25: "f32[128, 512]" = torch.ops.aten.addmm.default(arg71_1, view_92, permute_45);  arg71_1 = view_92 = permute_45 = None
    view_93: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_25, [1, 128, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_94: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_93, [1, -1, 16, 32]);  view_93 = None
    permute_46: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    clone_33: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_95: "f32[128, 512]" = torch.ops.aten.view.default(add_35, [128, 512])
    permute_47: "f32[512, 512]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_26: "f32[128, 512]" = torch.ops.aten.addmm.default(arg73_1, view_95, permute_47);  arg73_1 = view_95 = permute_47 = None
    view_96: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_26, [1, 128, 512]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_97: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_96, [1, -1, 16, 32]);  view_96 = None
    permute_48: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    clone_34: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_98: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(mul_35, [1, 128, 16, 32]);  mul_35 = None
    permute_49: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    clone_35: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_99: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_35, [16, -1, 32]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_100: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_33, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_101: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_34, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_99, permute_50);  view_99 = permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_102: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, [1, 16, 128, 128]);  bmm_8 = None
    add_36: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_102, expand_1);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_103: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_36, [16, 128, 128]);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_103, [-1], True)
    sub_13: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_103, amax_4);  view_103 = amax_4 = None
    exp_4: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_36: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(clone_36, view_101);  clone_36 = view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_104: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_9, [1, 16, 128, 32]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_105: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_37, [1, 128, 512]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_106: "f32[128, 512]" = torch.ops.aten.view.default(view_105, [128, 512]);  view_105 = None
    permute_52: "f32[512, 512]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_27: "f32[128, 512]" = torch.ops.aten.addmm.default(arg75_1, view_106, permute_52);  arg75_1 = view_106 = permute_52 = None
    view_107: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_27, [1, 128, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_38: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_107);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_37: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_35, clone_38);  add_35 = clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_38: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_14: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
    mul_36: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_37: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_36, arg76_1);  mul_36 = arg76_1 = None
    add_39: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_37, arg77_1);  mul_37 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_108: "f32[128, 512]" = torch.ops.aten.view.default(add_39, [128, 512])
    permute_53: "f32[512, 2048]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_28: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg79_1, view_108, permute_53);  arg79_1 = view_108 = permute_53 = None
    view_109: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_28, [1, 128, 2048]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_39: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
    erf_4: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_40: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_38, add_40);  mul_38 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:450, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_39: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(mul_40);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_110: "f32[128, 2048]" = torch.ops.aten.view.default(clone_39, [128, 2048]);  clone_39 = None
    permute_54: "f32[2048, 512]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_29: "f32[128, 512]" = torch.ops.aten.addmm.default(arg81_1, view_110, permute_54);  arg81_1 = view_110 = permute_54 = None
    view_111: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_29, [1, 128, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_40: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_111);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_41: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_39, clone_40);  add_39 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_42: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_15: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
    mul_41: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_42: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_41, arg82_1);  mul_41 = arg82_1 = None
    add_43: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_42, arg83_1);  mul_42 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_112: "f32[128, 512]" = torch.ops.aten.view.default(add_43, [128, 512])
    permute_55: "f32[512, 512]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_30: "f32[128, 512]" = torch.ops.aten.addmm.default(arg85_1, view_112, permute_55);  arg85_1 = view_112 = permute_55 = None
    view_113: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_30, [1, 128, 512]);  addmm_30 = None
    mul_43: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_113, 0.1767766952966369);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_114: "f32[128, 512]" = torch.ops.aten.view.default(add_43, [128, 512])
    permute_56: "f32[512, 512]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_31: "f32[128, 512]" = torch.ops.aten.addmm.default(arg87_1, view_114, permute_56);  arg87_1 = view_114 = permute_56 = None
    view_115: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_31, [1, 128, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_116: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_115, [1, -1, 16, 32]);  view_115 = None
    permute_57: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    clone_41: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_117: "f32[128, 512]" = torch.ops.aten.view.default(add_43, [128, 512])
    permute_58: "f32[512, 512]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_32: "f32[128, 512]" = torch.ops.aten.addmm.default(arg89_1, view_117, permute_58);  arg89_1 = view_117 = permute_58 = None
    view_118: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_32, [1, 128, 512]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_119: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_118, [1, -1, 16, 32]);  view_118 = None
    permute_59: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    clone_42: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_120: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(mul_43, [1, 128, 16, 32]);  mul_43 = None
    permute_60: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    clone_43: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_121: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_43, [16, -1, 32]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_122: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_41, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_123: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_42, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_121, permute_61);  view_121 = permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_124: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, [1, 16, 128, 128]);  bmm_10 = None
    add_44: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_124, expand_1);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_125: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_44, [16, 128, 128]);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_125, [-1], True)
    sub_16: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_125, amax_5);  view_125 = amax_5 = None
    exp_5: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_44: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(clone_44, view_123);  clone_44 = view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_126: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_11, [1, 16, 128, 32]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_127: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_45, [1, 128, 512]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_128: "f32[128, 512]" = torch.ops.aten.view.default(view_127, [128, 512]);  view_127 = None
    permute_63: "f32[512, 512]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_33: "f32[128, 512]" = torch.ops.aten.addmm.default(arg91_1, view_128, permute_63);  arg91_1 = view_128 = permute_63 = None
    view_129: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_33, [1, 128, 512]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_46: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_45: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_43, clone_46);  add_43 = clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_46: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_17: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
    mul_44: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_45: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_44, arg92_1);  mul_44 = arg92_1 = None
    add_47: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_45, arg93_1);  mul_45 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_130: "f32[128, 512]" = torch.ops.aten.view.default(add_47, [128, 512])
    permute_64: "f32[512, 2048]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_34: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg95_1, view_130, permute_64);  arg95_1 = view_130 = permute_64 = None
    view_131: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_34, [1, 128, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_47: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
    erf_5: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_48: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:450, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_47: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_132: "f32[128, 2048]" = torch.ops.aten.view.default(clone_47, [128, 2048]);  clone_47 = None
    permute_65: "f32[2048, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_35: "f32[128, 512]" = torch.ops.aten.addmm.default(arg97_1, view_132, permute_65);  arg97_1 = view_132 = permute_65 = None
    view_133: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_35, [1, 128, 512]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_48: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_133);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_49: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_47, clone_48);  add_47 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_50: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_18: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
    mul_49: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_50: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_49, arg98_1);  mul_49 = arg98_1 = None
    add_51: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_50, arg99_1);  mul_50 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_134: "f32[128, 512]" = torch.ops.aten.view.default(add_51, [128, 512])
    permute_66: "f32[512, 512]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_36: "f32[128, 512]" = torch.ops.aten.addmm.default(arg101_1, view_134, permute_66);  arg101_1 = view_134 = permute_66 = None
    view_135: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_36, [1, 128, 512]);  addmm_36 = None
    mul_51: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_135, 0.1767766952966369);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_136: "f32[128, 512]" = torch.ops.aten.view.default(add_51, [128, 512])
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_37: "f32[128, 512]" = torch.ops.aten.addmm.default(arg103_1, view_136, permute_67);  arg103_1 = view_136 = permute_67 = None
    view_137: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_37, [1, 128, 512]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_138: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_137, [1, -1, 16, 32]);  view_137 = None
    permute_68: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_49: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_139: "f32[128, 512]" = torch.ops.aten.view.default(add_51, [128, 512])
    permute_69: "f32[512, 512]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_38: "f32[128, 512]" = torch.ops.aten.addmm.default(arg105_1, view_139, permute_69);  arg105_1 = view_139 = permute_69 = None
    view_140: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_38, [1, 128, 512]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_141: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_140, [1, -1, 16, 32]);  view_140 = None
    permute_70: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    clone_50: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_142: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(mul_51, [1, 128, 16, 32]);  mul_51 = None
    permute_71: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    clone_51: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_143: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_51, [16, -1, 32]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_144: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_49, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_145: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_50, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_143, permute_72);  view_143 = permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_146: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, [1, 16, 128, 128]);  bmm_12 = None
    add_52: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_146, expand_1);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_147: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_52, [16, 128, 128]);  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_147, [-1], True)
    sub_19: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_147, amax_6);  view_147 = amax_6 = None
    exp_6: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_52: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(clone_52, view_145);  clone_52 = view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_148: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_13, [1, 16, 128, 32]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_53: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_149: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_53, [1, 128, 512]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_150: "f32[128, 512]" = torch.ops.aten.view.default(view_149, [128, 512]);  view_149 = None
    permute_74: "f32[512, 512]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_39: "f32[128, 512]" = torch.ops.aten.addmm.default(arg107_1, view_150, permute_74);  arg107_1 = view_150 = permute_74 = None
    view_151: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_39, [1, 128, 512]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_54: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_53: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_51, clone_54);  add_51 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    add_54: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_20: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
    mul_52: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_53: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_52, arg108_1);  mul_52 = arg108_1 = None
    add_55: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_53, arg109_1);  mul_53 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_152: "f32[128, 512]" = torch.ops.aten.view.default(add_55, [128, 512])
    permute_75: "f32[512, 2048]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_40: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg111_1, view_152, permute_75);  arg111_1 = view_152 = permute_75 = None
    view_153: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_40, [1, 128, 2048]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    mul_55: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
    erf_6: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_56: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_56: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_54, add_56);  mul_54 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:450, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_55: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_154: "f32[128, 2048]" = torch.ops.aten.view.default(clone_55, [128, 2048]);  clone_55 = None
    permute_76: "f32[2048, 512]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_41: "f32[128, 512]" = torch.ops.aten.addmm.default(arg113_1, view_154, permute_76);  arg113_1 = view_154 = permute_76 = None
    view_155: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_41, [1, 128, 512]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_56: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_155);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_57: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_55, clone_56);  add_55 = clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_58: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_21: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
    mul_57: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_58: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_57, arg114_1);  mul_57 = arg114_1 = None
    add_59: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_58, arg115_1);  mul_58 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_156: "f32[128, 512]" = torch.ops.aten.view.default(add_59, [128, 512])
    permute_77: "f32[512, 512]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_42: "f32[128, 512]" = torch.ops.aten.addmm.default(arg117_1, view_156, permute_77);  arg117_1 = view_156 = permute_77 = None
    view_157: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_42, [1, 128, 512]);  addmm_42 = None
    mul_59: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_157, 0.1767766952966369);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_158: "f32[128, 512]" = torch.ops.aten.view.default(add_59, [128, 512])
    permute_78: "f32[512, 512]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_43: "f32[128, 512]" = torch.ops.aten.addmm.default(arg119_1, view_158, permute_78);  arg119_1 = view_158 = permute_78 = None
    view_159: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_43, [1, 128, 512]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_160: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_159, [1, -1, 16, 32]);  view_159 = None
    permute_79: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    clone_57: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_161: "f32[128, 512]" = torch.ops.aten.view.default(add_59, [128, 512])
    permute_80: "f32[512, 512]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_44: "f32[128, 512]" = torch.ops.aten.addmm.default(arg121_1, view_161, permute_80);  arg121_1 = view_161 = permute_80 = None
    view_162: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_44, [1, 128, 512]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_163: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(view_162, [1, -1, 16, 32]);  view_162 = None
    permute_81: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    clone_58: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_164: "f32[1, 128, 16, 32]" = torch.ops.aten.view.default(mul_59, [1, 128, 16, 32]);  mul_59 = None
    permute_82: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    clone_59: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_165: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_59, [16, -1, 32]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_166: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_57, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_167: "f32[16, 128, 32]" = torch.ops.aten.view.default(clone_58, [16, -1, 32])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_83: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_165, permute_83);  view_165 = permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_168: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, [1, 16, 128, 128]);  bmm_14 = None
    add_60: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_168, expand_1);  view_168 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_169: "f32[16, 128, 128]" = torch.ops.aten.view.default(add_60, [16, 128, 128]);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_169, [-1], True)
    sub_22: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_169, amax_7);  view_169 = amax_7 = None
    exp_7: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:258, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_60: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_15: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(clone_60, view_167);  clone_60 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_170: "f32[1, 16, 128, 32]" = torch.ops.aten.view.default(bmm_15, [1, 16, 128, 32]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_61: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_171: "f32[1, 128, 512]" = torch.ops.aten.view.default(clone_61, [1, 128, 512]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_172: "f32[128, 512]" = torch.ops.aten.view.default(view_171, [128, 512]);  view_171 = None
    permute_85: "f32[512, 512]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_45: "f32[128, 512]" = torch.ops.aten.addmm.default(arg123_1, view_172, permute_85);  arg123_1 = view_172 = permute_85 = None
    view_173: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_45, [1, 128, 512]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:420, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_62: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_173);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_61: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_59, clone_62);  add_59 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_62: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_23: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
    mul_60: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_61: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_60, arg124_1);  mul_60 = arg124_1 = None
    add_63: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_61, arg125_1);  mul_61 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_174: "f32[128, 512]" = torch.ops.aten.view.default(add_63, [128, 512])
    permute_86: "f32[512, 2048]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_46: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg127_1, view_174, permute_86);  arg127_1 = view_174 = permute_86 = None
    view_175: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_46, [1, 128, 2048]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    mul_63: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
    erf_7: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_64: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_64: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_62, add_64);  mul_62 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:450, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_63: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_176: "f32[128, 2048]" = torch.ops.aten.view.default(clone_63, [128, 2048]);  clone_63 = None
    permute_87: "f32[2048, 512]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_47: "f32[128, 512]" = torch.ops.aten.addmm.default(arg129_1, view_176, permute_87);  arg129_1 = view_176 = permute_87 = None
    view_177: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_47, [1, 128, 512]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:452, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_64: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_177);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_65: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_63, clone_64);  add_63 = clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_66: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_24: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
    mul_65: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_66: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_65, arg130_1);  mul_65 = arg130_1 = None
    add_67: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_66, arg131_1);  mul_66 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:1562, code: logits = self.lm_head(outputs[0])
    permute_88: "f32[512, 50265]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    view_178: "f32[128, 512]" = torch.ops.aten.view.default(add_67, [128, 512]);  add_67 = None
    mm: "f32[128, 50265]" = torch.ops.aten.mm.default(view_178, permute_88);  view_178 = permute_88 = None
    view_179: "f32[1, 128, 50265]" = torch.ops.aten.view.default(mm, [1, 128, 50265]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:1568, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_180: "f32[128, 50265]" = torch.ops.aten.view.default(view_179, [-1, 50265])
    view_181: "i64[128]" = torch.ops.aten.view.default(arg134_1, [-1]);  arg134_1 = None
    amax_8: "f32[128, 1]" = torch.ops.aten.amax.default(view_180, [1], True)
    sub_25: "f32[128, 50265]" = torch.ops.aten.sub.Tensor(view_180, amax_8);  view_180 = amax_8 = None
    exp_8: "f32[128, 50265]" = torch.ops.aten.exp.default(sub_25)
    sum_9: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True);  exp_8 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_9);  sum_9 = None
    sub_26: "f32[128, 50265]" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_181, -100)
    scalar_tensor_1: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_1: "i64[128]" = torch.ops.aten.where.self(ne, view_181, scalar_tensor_1);  ne = scalar_tensor_1 = None
    unsqueeze_4: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_26, 1, unsqueeze_4);  sub_26 = unsqueeze_4 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_181, -100)
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[128]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_2);  ne_1 = neg = scalar_tensor_2 = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_181, -100);  view_181 = None
    sum_10: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_10, torch.float32);  sum_10 = None
    sum_11: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    div_8: "f32[]" = torch.ops.aten.div.Tensor(sum_11, convert_element_type);  sum_11 = convert_element_type = None
    return (div_8, view_179, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42, clone_49, clone_50, clone_57, clone_58)
    