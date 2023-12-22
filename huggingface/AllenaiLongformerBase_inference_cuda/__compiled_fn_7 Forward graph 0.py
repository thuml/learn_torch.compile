from __future__ import annotations



def forward(self, arg0_1: "f32[768, 768]", arg1_1: "f32[768]", arg2_1: "f32[768, 768]", arg3_1: "f32[768]", arg4_1: "f32[768, 768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[768]", arg10_1: "f32[3072, 768]", arg11_1: "f32[3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768, 768]", arg17_1: "f32[768]", arg18_1: "f32[768, 768]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768, 768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768]", arg26_1: "f32[3072, 768]", arg27_1: "f32[3072]", arg28_1: "f32[768, 3072]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768, 768]", arg33_1: "f32[768]", arg34_1: "f32[768, 768]", arg35_1: "f32[768]", arg36_1: "f32[768, 768]", arg37_1: "f32[768]", arg38_1: "f32[768, 768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[3072, 768]", arg43_1: "f32[3072]", arg44_1: "f32[768, 3072]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768, 768]", arg49_1: "f32[768]", arg50_1: "f32[768, 768]", arg51_1: "f32[768]", arg52_1: "f32[768, 768]", arg53_1: "f32[768]", arg54_1: "f32[768, 768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[3072, 768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768, 768]", arg65_1: "f32[768]", arg66_1: "f32[768, 768]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768, 768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768]", arg74_1: "f32[3072, 768]", arg75_1: "f32[3072]", arg76_1: "f32[768, 3072]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[768, 768]", arg81_1: "f32[768]", arg82_1: "f32[768, 768]", arg83_1: "f32[768]", arg84_1: "f32[768, 768]", arg85_1: "f32[768]", arg86_1: "f32[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[3072, 768]", arg91_1: "f32[3072]", arg92_1: "f32[768, 3072]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[768, 768]", arg97_1: "f32[768]", arg98_1: "f32[768, 768]", arg99_1: "f32[768]", arg100_1: "f32[768, 768]", arg101_1: "f32[768]", arg102_1: "f32[768, 768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[3072, 768]", arg107_1: "f32[3072]", arg108_1: "f32[768, 3072]", arg109_1: "f32[768]", arg110_1: "f32[768]", arg111_1: "f32[768]", arg112_1: "f32[768, 768]", arg113_1: "f32[768]", arg114_1: "f32[768, 768]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768, 768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768]", arg122_1: "f32[3072, 768]", arg123_1: "f32[3072]", arg124_1: "f32[768, 3072]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[768]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[768, 768]", arg131_1: "f32[768]", arg132_1: "f32[768, 768]", arg133_1: "f32[768]", arg134_1: "f32[768, 768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[3072, 768]", arg139_1: "f32[3072]", arg140_1: "f32[768, 3072]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[768, 768]", arg145_1: "f32[768]", arg146_1: "f32[768, 768]", arg147_1: "f32[768]", arg148_1: "f32[768, 768]", arg149_1: "f32[768]", arg150_1: "f32[768, 768]", arg151_1: "f32[768]", arg152_1: "f32[768]", arg153_1: "f32[768]", arg154_1: "f32[3072, 768]", arg155_1: "f32[3072]", arg156_1: "f32[768, 3072]", arg157_1: "f32[768]", arg158_1: "f32[768]", arg159_1: "f32[768]", arg160_1: "f32[768, 768]", arg161_1: "f32[768]", arg162_1: "f32[768, 768]", arg163_1: "f32[768]", arg164_1: "f32[768, 768]", arg165_1: "f32[768]", arg166_1: "f32[768, 768]", arg167_1: "f32[768]", arg168_1: "f32[768]", arg169_1: "f32[768]", arg170_1: "f32[3072, 768]", arg171_1: "f32[3072]", arg172_1: "f32[768, 3072]", arg173_1: "f32[768]", arg174_1: "f32[768]", arg175_1: "f32[768]", arg176_1: "f32[768, 768]", arg177_1: "f32[768]", arg178_1: "f32[768, 768]", arg179_1: "f32[768]", arg180_1: "f32[768, 768]", arg181_1: "f32[768]", arg182_1: "f32[768, 768]", arg183_1: "f32[768]", arg184_1: "f32[768]", arg185_1: "f32[768]", arg186_1: "f32[3072, 768]", arg187_1: "f32[3072]", arg188_1: "f32[768, 3072]", arg189_1: "f32[768]", arg190_1: "f32[768]", arg191_1: "f32[768]", arg192_1: "f32[1, 1024, 768]", arg193_1: "f32[1, 1024]", arg194_1: "b8[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(arg192_1, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view: "f32[1024, 768]" = torch.ops.aten.view.default(permute, [1024, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
    addmm: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg1_1, view, permute_1);  arg1_1 = view = permute_1 = None
    view_1: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm, [1024, 1, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_2: "f32[1024, 768]" = torch.ops.aten.view.default(permute, [1024, 768])
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
    addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg3_1, view_2, permute_2);  arg3_1 = view_2 = permute_2 = None
    view_3: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_1, [1024, 1, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_4: "f32[1024, 768]" = torch.ops.aten.view.default(permute, [1024, 768]);  permute = None
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm_2: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg5_1, view_4, permute_3);  arg5_1 = view_4 = permute_3 = None
    view_5: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_2, [1024, 1, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_1, 8.0);  view_1 = None
    view_6: "f32[1024, 768]" = torch.ops.aten.view.default(div, [1024, 768]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_9: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_3, [1024, 1, 12, 64]);  view_3 = None
    permute_5: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_9, [1, 0, 2, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
    view_11: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_7, [12, 1024, 64]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_13: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_11, [12, 2, 512, 64]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_1: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_13, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_1: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
    permute_9: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_1, [0, 1, 4, 2, 3]);  unsqueeze_1 = None
    view_14: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_6, [1024, 1, 768]);  view_6 = None
    view_15: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_14, [1024, 1, 12, 64]);  view_14 = None
    permute_11: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_15, [1, 0, 2, 3]);  view_15 = None
    permute_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_11, [0, 2, 1, 3]);  permute_11 = None
    view_16: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_12, [12, 1024, 64]);  permute_12 = None
    view_17: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_16, [12, 2, 512, 64]);  view_16 = None
    as_strided_2: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_17, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_17 = None
    unsqueeze_2: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_2, 4);  as_strided_2 = None
    permute_13: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 2, 4, 3]);  unsqueeze_2 = None
    permute_14: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_13, [0, 1, 2, 4, 3]);  permute_13 = None
    clone: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_18: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone, [36, 512, 64]);  clone = None
    permute_15: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_9, [0, 1, 4, 3, 2]);  permute_9 = None
    clone_1: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_19: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_1, [36, 64, 512]);  clone_1 = None
    bmm: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_18, view_19);  view_18 = view_19 = None
    view_20: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm, [12, 3, 512, 1, 512]);  bmm = None
    permute_16: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_20, [0, 1, 2, 4, 3]);  view_20 = None
    view_21: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_16, [12, 3, 512, 512]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_21, [0, 0, 0, 1], 0.0);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_22: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd, [12, 3, 512, 513]);  constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_22, 0, 0, 9223372036854775807)
    slice_2: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    slice_3: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 256);  slice_2 = None
    slice_4: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 257);  slice_3 = None
    slice_5: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_6: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, -1);  slice_5 = None
    slice_7: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 9223372036854775807);  slice_6 = None
    slice_8: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_7, 3, 256, 9223372036854775807);  slice_7 = None
    copy: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_4);  slice_8 = slice_4 = None
    slice_9: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_10: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, -1)
    slice_11: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_10, 2, 0, 9223372036854775807)
    slice_scatter: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_11, copy, 3, 256, 9223372036854775807);  slice_11 = copy = None
    slice_scatter_1: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_10, slice_scatter, 2, 0, 9223372036854775807);  slice_10 = slice_scatter = None
    slice_scatter_2: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_9, slice_scatter_1, 1, 0, -1);  slice_9 = slice_scatter_1 = None
    slice_scatter_3: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_2, 0, 0, 9223372036854775807);  full = slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_16: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_22, 0, 0, 9223372036854775807)
    select: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_16, 1, -1);  slice_16 = None
    slice_17: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
    slice_18: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_17, 2, 0, 257);  slice_17 = None
    slice_22: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 0, 9223372036854775807)
    select_2: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_22, 1, -1);  slice_22 = None
    slice_23: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_2, 1, 0, 9223372036854775807);  select_2 = None
    slice_24: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_23, 2, 256, 9223372036854775807);  slice_23 = None
    copy_1: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_24, slice_18);  slice_24 = slice_18 = None
    slice_25: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 0, 9223372036854775807)
    select_3: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_25, 1, -1)
    slice_26: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_3, 1, 0, 9223372036854775807)
    slice_scatter_4: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_26, copy_1, 2, 256, 9223372036854775807);  slice_26 = copy_1 = None
    slice_scatter_5: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_3, slice_scatter_4, 1, 0, 9223372036854775807);  select_3 = slice_scatter_4 = None
    select_scatter: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_25, slice_scatter_5, 1, -1);  slice_25 = slice_scatter_5 = None
    slice_scatter_6: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_3, select_scatter, 0, 0, 9223372036854775807);  slice_scatter_3 = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_30: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_22, 0, 0, 9223372036854775807)
    slice_31: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_30, 1, 0, 9223372036854775807);  slice_30 = None
    slice_32: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_31, 2, -257, -1);  slice_31 = None
    slice_33: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_32, 3, 257, 9223372036854775807);  slice_32 = None
    slice_38: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 0, 9223372036854775807)
    slice_39: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_38, 1, 1, 9223372036854775807);  slice_38 = None
    slice_40: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_39, 2, 0, 9223372036854775807);  slice_39 = None
    slice_41: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_40, 3, 0, 256);  slice_40 = None
    copy_2: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_41, slice_33);  slice_41 = slice_33 = None
    slice_42: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 0, 9223372036854775807)
    slice_43: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_42, 1, 1, 9223372036854775807)
    slice_44: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_43, 2, 0, 9223372036854775807)
    slice_scatter_7: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_44, copy_2, 3, 0, 256);  slice_44 = copy_2 = None
    slice_scatter_8: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_43, slice_scatter_7, 2, 0, 9223372036854775807);  slice_43 = slice_scatter_7 = None
    slice_scatter_9: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_42, slice_scatter_8, 1, 1, 9223372036854775807);  slice_42 = slice_scatter_8 = None
    slice_scatter_10: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_6, slice_scatter_9, 0, 0, 9223372036854775807);  slice_scatter_6 = slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_49: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_22, 0, 0, 9223372036854775807);  view_22 = None
    select_5: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_49, 1, 0);  slice_49 = None
    slice_50: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_5, 1, 0, 255);  select_5 = None
    slice_51: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_50, 2, -255, 9223372036854775807);  slice_50 = None
    slice_55: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_10, 0, 0, 9223372036854775807)
    select_7: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_55, 1, 0);  slice_55 = None
    slice_56: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_7, 1, 1, 256);  select_7 = None
    slice_57: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_56, 2, 1, 256);  slice_56 = None
    copy_3: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_57, slice_51);  slice_57 = slice_51 = None
    slice_58: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_10, 0, 0, 9223372036854775807)
    select_8: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_58, 1, 0)
    slice_59: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_8, 1, 1, 256)
    slice_scatter_11: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_59, copy_3, 2, 1, 256);  slice_59 = copy_3 = None
    slice_scatter_12: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_8, slice_scatter_11, 1, 1, 256);  select_8 = slice_scatter_11 = None
    select_scatter_1: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_58, slice_scatter_12, 1, 0);  slice_58 = slice_scatter_12 = None
    slice_scatter_13: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_10, select_scatter_1, 0, 0, 9223372036854775807);  slice_scatter_10 = select_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_1: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_3: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
    iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_4: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    sub_1: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
    le: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[256, 257]" = torch.ops.aten.where.self(le, full_1, scalar_tensor);  le = full_1 = scalar_tensor = None
    rev: "f32[256, 257]" = torch.ops.prims.rev.default(where, [0]);  where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_5: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
    slice_63: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_5, 1, 0, 9223372036854775807);  unsqueeze_5 = None
    unsqueeze_6: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_63, 2);  slice_63 = None
    slice_64: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_6, 3, 0, 9223372036854775807);  unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_1: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_64, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_64, [1, 256, 12, 257]);  slice_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_25: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_13, [1, 12, 1024, 513])
    permute_19: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    slice_69: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_19, 0, 0, 9223372036854775807);  permute_19 = None
    slice_70: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_69, 1, 0, 256);  slice_69 = None
    slice_71: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_70, 2, 0, 9223372036854775807);  slice_70 = None
    slice_72: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_71, 3, 0, 257);  slice_71 = None
    full_2: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
    where_1: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_2, slice_72);  convert_element_type = full_2 = slice_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_26: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_13, [1, 12, 1024, 513])
    permute_20: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    slice_77: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_20, 0, 0, 9223372036854775807);  permute_20 = None
    slice_78: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_77, 1, 0, 256);  slice_77 = None
    slice_79: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_78, 2, 0, 9223372036854775807);  slice_78 = None
    slice_80: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_79, 3, 0, 257);  slice_79 = None
    copy_4: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_80, where_1);  slice_80 = where_1 = None
    view_27: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_13, [1, 12, 1024, 513]);  slice_scatter_13 = None
    permute_21: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    slice_81: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_21, 0, 0, 9223372036854775807)
    slice_82: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_81, 1, 0, 256)
    slice_83: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_82, 2, 0, 9223372036854775807)
    slice_scatter_14: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_83, copy_4, 3, 0, 257);  slice_83 = copy_4 = None
    slice_scatter_15: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_82, slice_scatter_14, 2, 0, 9223372036854775807);  slice_82 = slice_scatter_14 = None
    slice_scatter_16: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_81, slice_scatter_15, 1, 0, 256);  slice_81 = slice_scatter_15 = None
    slice_scatter_17: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_21, slice_scatter_16, 0, 0, 9223372036854775807);  permute_21 = slice_scatter_16 = None
    permute_22: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_17, [0, 2, 1, 3]);  slice_scatter_17 = None
    view_28: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_22, [12, 4, 256, 513]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_1: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, [1, 256, 12, 257]);  rev_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_30: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_28, [1, 12, 1024, 513])
    permute_24: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    slice_92: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_24, 0, 0, 9223372036854775807);  permute_24 = None
    slice_93: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_92, 1, -256, 9223372036854775807);  slice_92 = None
    slice_94: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_93, 2, 0, 9223372036854775807);  slice_93 = None
    slice_95: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_94, 3, -257, 9223372036854775807);  slice_94 = None
    full_3: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_1: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
    where_2: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_3, slice_95);  convert_element_type_1 = full_3 = slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_31: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_28, [1, 12, 1024, 513])
    permute_25: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    slice_100: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_25, 0, 0, 9223372036854775807);  permute_25 = None
    slice_101: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_100, 1, -256, 9223372036854775807);  slice_100 = None
    slice_102: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_101, 2, 0, 9223372036854775807);  slice_101 = None
    slice_103: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_102, 3, -257, 9223372036854775807);  slice_102 = None
    copy_5: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_103, where_2);  slice_103 = where_2 = None
    view_32: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_28, [1, 12, 1024, 513]);  view_28 = None
    permute_26: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    slice_104: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_26, 0, 0, 9223372036854775807)
    slice_105: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_104, 1, -256, 9223372036854775807)
    slice_106: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_105, 2, 0, 9223372036854775807)
    slice_scatter_18: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_106, copy_5, 3, -257, 9223372036854775807);  slice_106 = copy_5 = None
    slice_scatter_19: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_105, slice_scatter_18, 2, 0, 9223372036854775807);  slice_105 = slice_scatter_18 = None
    slice_scatter_20: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_104, slice_scatter_19, 1, -256, 9223372036854775807);  slice_104 = slice_scatter_19 = None
    slice_scatter_21: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_26, slice_scatter_20, 0, 0, 9223372036854775807);  permute_26 = slice_scatter_20 = None
    permute_27: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3]);  slice_scatter_21 = None
    view_33: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_27, [12, 4, 256, 513]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_111: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne, 0, 0, 9223372036854775807);  ne = None
    slice_112: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_111, 1, 0, 9223372036854775807);  slice_111 = None
    unsqueeze_7: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_112, 2);  slice_112 = None
    unsqueeze_8: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_2: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_8, torch.float32)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_8, scalar_tensor_1, convert_element_type_2);  unsqueeze_8 = scalar_tensor_1 = convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_4: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_29: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
    view_35: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_29, [1, 1024, 1]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_30: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
    view_36: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_30, [1, 1024, 1]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_37: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_35, [1, 2, 512, 1]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_3: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_37, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_38: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_36, [1, 2, 512, 1]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_4: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_38, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_9: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_3, 4);  as_strided_3 = None
    permute_31: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_9, [0, 1, 2, 4, 3]);  unsqueeze_9 = None
    unsqueeze_10: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_4, 4);  as_strided_4 = None
    permute_32: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_10, [0, 1, 4, 2, 3]);  unsqueeze_10 = None
    mul: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_31, permute_32);  permute_31 = permute_32 = None
    view_39: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul, [1, 3, 512, 512]);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_1: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_39, [0, 0, 0, 1], 0.0);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_40: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_1, [1, 3, 512, 513]);  constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_5: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_113: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_40, 0, 0, 9223372036854775807)
    slice_114: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_113, 1, 0, 9223372036854775807);  slice_113 = None
    slice_115: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_114, 2, 0, 256);  slice_114 = None
    slice_116: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_115, 3, 0, 257);  slice_115 = None
    slice_117: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 0, 0, 9223372036854775807)
    slice_118: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_117, 1, 0, -1);  slice_117 = None
    slice_119: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_118, 2, 0, 9223372036854775807);  slice_118 = None
    slice_120: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_119, 3, 256, 9223372036854775807);  slice_119 = None
    copy_6: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_120, slice_116);  slice_120 = slice_116 = None
    slice_121: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 0, 0, 9223372036854775807)
    slice_122: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_121, 1, 0, -1)
    slice_123: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_122, 2, 0, 9223372036854775807)
    slice_scatter_22: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_123, copy_6, 3, 256, 9223372036854775807);  slice_123 = copy_6 = None
    slice_scatter_23: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_122, slice_scatter_22, 2, 0, 9223372036854775807);  slice_122 = slice_scatter_22 = None
    slice_scatter_24: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_121, slice_scatter_23, 1, 0, -1);  slice_121 = slice_scatter_23 = None
    slice_scatter_25: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_24, 0, 0, 9223372036854775807);  full_5 = slice_scatter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_128: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_40, 0, 0, 9223372036854775807)
    select_10: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_128, 1, -1);  slice_128 = None
    slice_129: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_10, 1, 256, 9223372036854775807);  select_10 = None
    slice_130: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_129, 2, 0, 257);  slice_129 = None
    slice_134: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 0, 9223372036854775807)
    select_12: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_134, 1, -1);  slice_134 = None
    slice_135: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_12, 1, 0, 9223372036854775807);  select_12 = None
    slice_136: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_135, 2, 256, 9223372036854775807);  slice_135 = None
    copy_7: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_136, slice_130);  slice_136 = slice_130 = None
    slice_137: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 0, 9223372036854775807)
    select_13: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_137, 1, -1)
    slice_138: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_13, 1, 0, 9223372036854775807)
    slice_scatter_26: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_138, copy_7, 2, 256, 9223372036854775807);  slice_138 = copy_7 = None
    slice_scatter_27: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_13, slice_scatter_26, 1, 0, 9223372036854775807);  select_13 = slice_scatter_26 = None
    select_scatter_2: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_137, slice_scatter_27, 1, -1);  slice_137 = slice_scatter_27 = None
    slice_scatter_28: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_25, select_scatter_2, 0, 0, 9223372036854775807);  slice_scatter_25 = select_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_142: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_40, 0, 0, 9223372036854775807)
    slice_143: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_142, 1, 0, 9223372036854775807);  slice_142 = None
    slice_144: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_143, 2, -257, -1);  slice_143 = None
    slice_145: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_144, 3, 257, 9223372036854775807);  slice_144 = None
    slice_150: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_28, 0, 0, 9223372036854775807)
    slice_151: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_150, 1, 1, 9223372036854775807);  slice_150 = None
    slice_152: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_151, 2, 0, 9223372036854775807);  slice_151 = None
    slice_153: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_152, 3, 0, 256);  slice_152 = None
    copy_8: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_153, slice_145);  slice_153 = slice_145 = None
    slice_154: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_28, 0, 0, 9223372036854775807)
    slice_155: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_154, 1, 1, 9223372036854775807)
    slice_156: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_155, 2, 0, 9223372036854775807)
    slice_scatter_29: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_156, copy_8, 3, 0, 256);  slice_156 = copy_8 = None
    slice_scatter_30: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_155, slice_scatter_29, 2, 0, 9223372036854775807);  slice_155 = slice_scatter_29 = None
    slice_scatter_31: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_154, slice_scatter_30, 1, 1, 9223372036854775807);  slice_154 = slice_scatter_30 = None
    slice_scatter_32: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_28, slice_scatter_31, 0, 0, 9223372036854775807);  slice_scatter_28 = slice_scatter_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_161: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_40, 0, 0, 9223372036854775807);  view_40 = None
    select_15: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_161, 1, 0);  slice_161 = None
    slice_162: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_15, 1, 0, 255);  select_15 = None
    slice_163: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_162, 2, -255, 9223372036854775807);  slice_162 = None
    slice_167: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 0, 9223372036854775807)
    select_17: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_167, 1, 0);  slice_167 = None
    slice_168: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_17, 1, 1, 256);  select_17 = None
    slice_169: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_168, 2, 1, 256);  slice_168 = None
    copy_9: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_169, slice_163);  slice_169 = slice_163 = None
    slice_170: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 0, 9223372036854775807)
    select_18: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_170, 1, 0)
    slice_171: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_18, 1, 1, 256)
    slice_scatter_33: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_171, copy_9, 2, 1, 256);  slice_171 = copy_9 = None
    slice_scatter_34: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_18, slice_scatter_33, 1, 1, 256);  select_18 = slice_scatter_33 = None
    select_scatter_3: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_170, slice_scatter_34, 1, 0);  slice_170 = slice_scatter_34 = None
    slice_scatter_35: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_32, select_scatter_3, 0, 0, 9223372036854775807);  slice_scatter_32 = select_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_6: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_2: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_11: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_2, -2);  iota_2 = None
    iota_3: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_12: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
    sub_3: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_11, unsqueeze_12);  unsqueeze_11 = unsqueeze_12 = None
    le_1: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_3, 0);  sub_3 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[256, 257]" = torch.ops.aten.where.self(le_1, full_6, scalar_tensor_2);  le_1 = full_6 = scalar_tensor_2 = None
    rev_2: "f32[256, 257]" = torch.ops.prims.rev.default(where_4, [0]);  where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_13: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_2, 0);  rev_2 = None
    slice_175: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 1, 0, 9223372036854775807);  unsqueeze_13 = None
    unsqueeze_14: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_175, 2);  slice_175 = None
    slice_176: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_14, 3, 0, 9223372036854775807);  unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_3: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_176, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_2: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_176, [1, 256, 1, 257]);  slice_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_43: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_35, [1, 1, 1024, 513])
    permute_35: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
    slice_181: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_35, 0, 0, 9223372036854775807);  permute_35 = None
    slice_182: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_181, 1, 0, 256);  slice_181 = None
    slice_183: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_182, 2, 0, 9223372036854775807);  slice_182 = None
    slice_184: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_183, 3, 0, 257);  slice_183 = None
    full_7: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_3: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None
    where_5: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_3, full_7, slice_184);  convert_element_type_3 = full_7 = slice_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_44: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_35, [1, 1, 1024, 513])
    permute_36: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
    slice_189: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_36, 0, 0, 9223372036854775807);  permute_36 = None
    slice_190: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_189, 1, 0, 256);  slice_189 = None
    slice_191: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_190, 2, 0, 9223372036854775807);  slice_190 = None
    slice_192: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_191, 3, 0, 257);  slice_191 = None
    copy_10: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_192, where_5);  slice_192 = where_5 = None
    view_45: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_35, [1, 1, 1024, 513]);  slice_scatter_35 = None
    permute_37: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    slice_193: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_37, 0, 0, 9223372036854775807)
    slice_194: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_193, 1, 0, 256)
    slice_195: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_194, 2, 0, 9223372036854775807)
    slice_scatter_36: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_195, copy_10, 3, 0, 257);  slice_195 = copy_10 = None
    slice_scatter_37: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_194, slice_scatter_36, 2, 0, 9223372036854775807);  slice_194 = slice_scatter_36 = None
    slice_scatter_38: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_193, slice_scatter_37, 1, 0, 256);  slice_193 = slice_scatter_37 = None
    slice_scatter_39: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_37, slice_scatter_38, 0, 0, 9223372036854775807);  permute_37 = slice_scatter_38 = None
    permute_38: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_39, [0, 2, 1, 3]);  slice_scatter_39 = None
    view_46: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_38, [1, 4, 256, 513]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_3: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_3, [1, 256, 1, 257]);  rev_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_48: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_46, [1, 1, 1024, 513])
    permute_40: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    slice_204: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_40, 0, 0, 9223372036854775807);  permute_40 = None
    slice_205: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_204, 1, -256, 9223372036854775807);  slice_204 = None
    slice_206: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_205, 2, 0, 9223372036854775807);  slice_205 = None
    slice_207: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_206, 3, -257, 9223372036854775807);  slice_206 = None
    full_8: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_4: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None
    where_6: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_4, full_8, slice_207);  convert_element_type_4 = full_8 = slice_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_49: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_46, [1, 1, 1024, 513])
    permute_41: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    slice_212: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_41, 0, 0, 9223372036854775807);  permute_41 = None
    slice_213: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_212, 1, -256, 9223372036854775807);  slice_212 = None
    slice_214: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_213, 2, 0, 9223372036854775807);  slice_213 = None
    slice_215: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_214, 3, -257, 9223372036854775807);  slice_214 = None
    copy_11: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_215, where_6);  slice_215 = where_6 = None
    view_50: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_46, [1, 1, 1024, 513]);  view_46 = None
    permute_42: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    slice_216: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_42, 0, 0, 9223372036854775807)
    slice_217: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_216, 1, -256, 9223372036854775807)
    slice_218: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_217, 2, 0, 9223372036854775807)
    slice_scatter_40: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_218, copy_11, 3, -257, 9223372036854775807);  slice_218 = copy_11 = None
    slice_scatter_41: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_217, slice_scatter_40, 2, 0, 9223372036854775807);  slice_217 = slice_scatter_40 = None
    slice_scatter_42: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_216, slice_scatter_41, 1, -256, 9223372036854775807);  slice_216 = slice_scatter_41 = None
    slice_scatter_43: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_42, slice_scatter_42, 0, 0, 9223372036854775807);  permute_42 = slice_scatter_42 = None
    permute_43: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_43, [0, 2, 1, 3]);  slice_scatter_43 = None
    view_51: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_43, [1, 4, 256, 513]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_53: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_33, [1, 12, 1024, 513]);  view_33 = None
    permute_45: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    view_54: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_51, [1, 1, 1024, 513]);  view_51 = None
    permute_46: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    add_2: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_45, permute_46);  permute_45 = permute_46 = None
    permute_47: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_2, [0, 2, 1, 3]);  add_2 = None
    view_56: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_47, [12, 4, 256, 513]);  permute_47 = None
    view_57: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_56, [1, 12, 1024, 513]);  view_56 = None
    permute_48: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_2: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    amax: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_2, [-1], True)
    sub_4: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_2, amax);  clone_2 = amax = None
    exp: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_1: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_7: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_223: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_224: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_223, 1, 0, 9223372036854775807);  slice_223 = None
    unsqueeze_15: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_224, 2);  slice_224 = None
    unsqueeze_16: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 3);  unsqueeze_15 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, scalar_tensor_3, div_7);  unsqueeze_16 = scalar_tensor_3 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_3: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_7);  where_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_58: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_5, [1024, 1, 12, 64]);  view_5 = None
    permute_49: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_58, [1, 0, 2, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_50: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1, 3]);  clone_3 = None
    view_59: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_50, [12, 4, 256, 513]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_51: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_49, [0, 2, 1, 3]);  permute_49 = None
    view_60: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_51, [12, 1024, 64]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_2: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_60, [0, 0, 256, 256], -1.0);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_5: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_2, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_3: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_59, [0, 257], 0.0);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_61: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_3, [12, 4, -1]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_225: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_61, 0, 0, 9223372036854775807);  view_61 = None
    slice_226: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_225, 1, 0, 9223372036854775807);  slice_225 = None
    slice_227: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_226, 2, 0, -256);  slice_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_62: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_227, [12, 4, 256, 769]);  slice_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_228: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_62, 0, 0, 9223372036854775807);  view_62 = None
    slice_229: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_228, 1, 0, 9223372036854775807);  slice_228 = None
    slice_230: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_229, 2, 0, 9223372036854775807);  slice_229 = None
    slice_231: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_230, 3, 0, -1);  slice_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_17: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_231, 4);  slice_231 = None
    permute_52: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_17, [0, 1, 2, 4, 3]);  unsqueeze_17 = None
    unsqueeze_18: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_5, 4);  as_strided_5 = None
    permute_53: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_18, [0, 1, 4, 3, 2]);  unsqueeze_18 = None
    permute_54: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_52, [0, 1, 2, 4, 3]);  permute_52 = None
    view_63: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_54, [48, 256, 768]);  permute_54 = None
    permute_55: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_53, [0, 1, 4, 3, 2]);  permute_53 = None
    clone_4: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    view_64: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_4, [48, 768, 64]);  clone_4 = None
    bmm_1: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_63, view_64);  view_63 = view_64 = None
    view_65: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_1, [12, 4, 256, 1, 64]);  bmm_1 = None
    permute_56: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_65, [0, 1, 2, 4, 3]);  view_65 = None
    view_66: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_56, [12, 4, 256, 64]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_67: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_66, [1, 12, 1024, 64]);  view_66 = None
    permute_57: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_58: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_57, [1, 0, 2, 3]);  permute_57 = None
    clone_5: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    view_68: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_5, [1024, 1, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_59: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_68, [1, 0, 2]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_69: "f32[1024, 768]" = torch.ops.aten.view.default(permute_59, [1024, 768]);  permute_59 = None
    permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm_3: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg7_1, view_69, permute_60);  arg7_1 = view_69 = permute_60 = None
    view_70: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_3, [1, 1024, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_6: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_70);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_6, arg192_1);  clone_6 = arg192_1 = None
    var_mean = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_5: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_6: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_1);  add_4 = getitem_1 = None
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = rsqrt = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg8_1);  mul_1 = arg8_1 = None
    add_6: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_71: "f32[1024, 768]" = torch.ops.aten.view.default(add_6, [1024, 768])
    permute_61: "f32[768, 3072]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm_4: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg11_1, view_71, permute_61);  arg11_1 = view_71 = permute_61 = None
    view_72: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 1024, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_3: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_72, 0.5)
    mul_4: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_72, 0.7071067811865476);  view_72 = None
    erf: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_7: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_5: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_3, add_7);  mul_3 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_73: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_5, [1024, 3072]);  mul_5 = None
    permute_62: "f32[3072, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    addmm_5: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg13_1, view_73, permute_62);  arg13_1 = view_73 = permute_62 = None
    view_74: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_5, [1, 1024, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_7: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_74);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_8: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_7, add_6);  clone_7 = add_6 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_7: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_3);  add_8 = getitem_3 = None
    mul_6: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = rsqrt_1 = None
    mul_7: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, arg14_1);  mul_6 = arg14_1 = None
    add_10: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_7, arg15_1);  mul_7 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_63: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_10, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_75: "f32[1024, 768]" = torch.ops.aten.view.default(permute_63, [1024, 768])
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_6: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg17_1, view_75, permute_64);  arg17_1 = view_75 = permute_64 = None
    view_76: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_6, [1024, 1, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_77: "f32[1024, 768]" = torch.ops.aten.view.default(permute_63, [1024, 768])
    permute_65: "f32[768, 768]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    addmm_7: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg19_1, view_77, permute_65);  arg19_1 = view_77 = permute_65 = None
    view_78: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_7, [1024, 1, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_79: "f32[1024, 768]" = torch.ops.aten.view.default(permute_63, [1024, 768]);  permute_63 = None
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_8: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg21_1, view_79, permute_66);  arg21_1 = view_79 = permute_66 = None
    view_80: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_8, [1024, 1, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_10: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_76, 8.0);  view_76 = None
    view_81: "f32[1024, 768]" = torch.ops.aten.view.default(div_10, [1024, 768]);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_84: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_78, [1024, 1, 12, 64]);  view_78 = None
    permute_68: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_84, [1, 0, 2, 3]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_70: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_68, [0, 2, 1, 3]);  permute_68 = None
    view_86: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_70, [12, 1024, 64]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_88: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_86, [12, 2, 512, 64]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_7: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_88, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_20: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_7, 4);  as_strided_7 = None
    permute_72: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_20, [0, 1, 4, 2, 3]);  unsqueeze_20 = None
    view_89: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_81, [1024, 1, 768]);  view_81 = None
    view_90: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_89, [1024, 1, 12, 64]);  view_89 = None
    permute_74: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_90, [1, 0, 2, 3]);  view_90 = None
    permute_75: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_74, [0, 2, 1, 3]);  permute_74 = None
    view_91: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_75, [12, 1024, 64]);  permute_75 = None
    view_92: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_91, [12, 2, 512, 64]);  view_91 = None
    as_strided_8: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_92, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_92 = None
    unsqueeze_21: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_8, 4);  as_strided_8 = None
    permute_76: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_21, [0, 1, 2, 4, 3]);  unsqueeze_21 = None
    permute_77: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_76, [0, 1, 2, 4, 3]);  permute_76 = None
    clone_8: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_93: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_8, [36, 512, 64]);  clone_8 = None
    permute_78: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_72, [0, 1, 4, 3, 2]);  permute_72 = None
    clone_9: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    view_94: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_9, [36, 64, 512]);  clone_9 = None
    bmm_2: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_93, view_94);  view_93 = view_94 = None
    view_95: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_2, [12, 3, 512, 1, 512]);  bmm_2 = None
    permute_79: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_95, [0, 1, 2, 4, 3]);  view_95 = None
    view_96: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_79, [12, 3, 512, 512]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_4: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_96, [0, 0, 0, 1], 0.0);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_97: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_4, [12, 3, 512, 513]);  constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_9: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_232: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_97, 0, 0, 9223372036854775807)
    slice_233: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_232, 1, 0, 9223372036854775807);  slice_232 = None
    slice_234: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_233, 2, 0, 256);  slice_233 = None
    slice_235: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_234, 3, 0, 257);  slice_234 = None
    slice_236: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_9, 0, 0, 9223372036854775807)
    slice_237: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_236, 1, 0, -1);  slice_236 = None
    slice_238: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_237, 2, 0, 9223372036854775807);  slice_237 = None
    slice_239: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_238, 3, 256, 9223372036854775807);  slice_238 = None
    copy_12: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_239, slice_235);  slice_239 = slice_235 = None
    slice_240: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_9, 0, 0, 9223372036854775807)
    slice_241: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_240, 1, 0, -1)
    slice_242: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_241, 2, 0, 9223372036854775807)
    slice_scatter_44: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_242, copy_12, 3, 256, 9223372036854775807);  slice_242 = copy_12 = None
    slice_scatter_45: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_241, slice_scatter_44, 2, 0, 9223372036854775807);  slice_241 = slice_scatter_44 = None
    slice_scatter_46: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_240, slice_scatter_45, 1, 0, -1);  slice_240 = slice_scatter_45 = None
    slice_scatter_47: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_46, 0, 0, 9223372036854775807);  full_9 = slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_247: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_97, 0, 0, 9223372036854775807)
    select_20: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_247, 1, -1);  slice_247 = None
    slice_248: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_20, 1, 256, 9223372036854775807);  select_20 = None
    slice_249: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_248, 2, 0, 257);  slice_248 = None
    slice_253: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 0, 9223372036854775807)
    select_22: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_253, 1, -1);  slice_253 = None
    slice_254: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_22, 1, 0, 9223372036854775807);  select_22 = None
    slice_255: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_254, 2, 256, 9223372036854775807);  slice_254 = None
    copy_13: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_255, slice_249);  slice_255 = slice_249 = None
    slice_256: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 0, 9223372036854775807)
    select_23: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_256, 1, -1)
    slice_257: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_23, 1, 0, 9223372036854775807)
    slice_scatter_48: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_257, copy_13, 2, 256, 9223372036854775807);  slice_257 = copy_13 = None
    slice_scatter_49: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_23, slice_scatter_48, 1, 0, 9223372036854775807);  select_23 = slice_scatter_48 = None
    select_scatter_4: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_256, slice_scatter_49, 1, -1);  slice_256 = slice_scatter_49 = None
    slice_scatter_50: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_47, select_scatter_4, 0, 0, 9223372036854775807);  slice_scatter_47 = select_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_261: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_97, 0, 0, 9223372036854775807)
    slice_262: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_261, 1, 0, 9223372036854775807);  slice_261 = None
    slice_263: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_262, 2, -257, -1);  slice_262 = None
    slice_264: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_263, 3, 257, 9223372036854775807);  slice_263 = None
    slice_269: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 0, 9223372036854775807)
    slice_270: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_269, 1, 1, 9223372036854775807);  slice_269 = None
    slice_271: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_270, 2, 0, 9223372036854775807);  slice_270 = None
    slice_272: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_271, 3, 0, 256);  slice_271 = None
    copy_14: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_272, slice_264);  slice_272 = slice_264 = None
    slice_273: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 0, 9223372036854775807)
    slice_274: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_273, 1, 1, 9223372036854775807)
    slice_275: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_274, 2, 0, 9223372036854775807)
    slice_scatter_51: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_275, copy_14, 3, 0, 256);  slice_275 = copy_14 = None
    slice_scatter_52: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_274, slice_scatter_51, 2, 0, 9223372036854775807);  slice_274 = slice_scatter_51 = None
    slice_scatter_53: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_273, slice_scatter_52, 1, 1, 9223372036854775807);  slice_273 = slice_scatter_52 = None
    slice_scatter_54: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_50, slice_scatter_53, 0, 0, 9223372036854775807);  slice_scatter_50 = slice_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_280: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_97, 0, 0, 9223372036854775807);  view_97 = None
    select_25: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_280, 1, 0);  slice_280 = None
    slice_281: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_25, 1, 0, 255);  select_25 = None
    slice_282: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_281, 2, -255, 9223372036854775807);  slice_281 = None
    slice_286: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_54, 0, 0, 9223372036854775807)
    select_27: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_286, 1, 0);  slice_286 = None
    slice_287: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_27, 1, 1, 256);  select_27 = None
    slice_288: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_287, 2, 1, 256);  slice_287 = None
    copy_15: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_288, slice_282);  slice_288 = slice_282 = None
    slice_289: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_54, 0, 0, 9223372036854775807)
    select_28: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_289, 1, 0)
    slice_290: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_28, 1, 1, 256)
    slice_scatter_55: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_290, copy_15, 2, 1, 256);  slice_290 = copy_15 = None
    slice_scatter_56: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_28, slice_scatter_55, 1, 1, 256);  select_28 = slice_scatter_55 = None
    select_scatter_5: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_289, slice_scatter_56, 1, 0);  slice_289 = slice_scatter_56 = None
    slice_scatter_57: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_54, select_scatter_5, 0, 0, 9223372036854775807);  slice_scatter_54 = select_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_10: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_4: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_22: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_4, -2);  iota_4 = None
    iota_5: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_23: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
    sub_9: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_22, unsqueeze_23);  unsqueeze_22 = unsqueeze_23 = None
    le_2: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_9, 0);  sub_9 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[256, 257]" = torch.ops.aten.where.self(le_2, full_10, scalar_tensor_4);  le_2 = full_10 = scalar_tensor_4 = None
    rev_4: "f32[256, 257]" = torch.ops.prims.rev.default(where_8, [0]);  where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_24: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_4, 0);  rev_4 = None
    slice_294: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_24, 1, 0, 9223372036854775807);  unsqueeze_24 = None
    unsqueeze_25: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_294, 2);  slice_294 = None
    slice_295: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_25, 3, 0, 9223372036854775807);  unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_5: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_295, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_4: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_295, [1, 256, 12, 257]);  slice_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_100: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_57, [1, 12, 1024, 513])
    permute_82: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
    slice_300: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_82, 0, 0, 9223372036854775807);  permute_82 = None
    slice_301: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_300, 1, 0, 256);  slice_300 = None
    slice_302: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_301, 2, 0, 9223372036854775807);  slice_301 = None
    slice_303: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_302, 3, 0, 257);  slice_302 = None
    full_11: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_5: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_4, torch.bool);  expand_4 = None
    where_9: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_5, full_11, slice_303);  convert_element_type_5 = full_11 = slice_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_101: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_57, [1, 12, 1024, 513])
    permute_83: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
    slice_308: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_83, 0, 0, 9223372036854775807);  permute_83 = None
    slice_309: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_308, 1, 0, 256);  slice_308 = None
    slice_310: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_309, 2, 0, 9223372036854775807);  slice_309 = None
    slice_311: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_310, 3, 0, 257);  slice_310 = None
    copy_16: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_311, where_9);  slice_311 = where_9 = None
    view_102: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_57, [1, 12, 1024, 513]);  slice_scatter_57 = None
    permute_84: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    slice_312: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_84, 0, 0, 9223372036854775807)
    slice_313: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_312, 1, 0, 256)
    slice_314: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_313, 2, 0, 9223372036854775807)
    slice_scatter_58: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_314, copy_16, 3, 0, 257);  slice_314 = copy_16 = None
    slice_scatter_59: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_313, slice_scatter_58, 2, 0, 9223372036854775807);  slice_313 = slice_scatter_58 = None
    slice_scatter_60: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_312, slice_scatter_59, 1, 0, 256);  slice_312 = slice_scatter_59 = None
    slice_scatter_61: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_84, slice_scatter_60, 0, 0, 9223372036854775807);  permute_84 = slice_scatter_60 = None
    permute_85: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_61, [0, 2, 1, 3]);  slice_scatter_61 = None
    view_103: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_85, [12, 4, 256, 513]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_5: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_5, [1, 256, 12, 257]);  rev_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_105: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_103, [1, 12, 1024, 513])
    permute_87: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    slice_323: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_87, 0, 0, 9223372036854775807);  permute_87 = None
    slice_324: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_323, 1, -256, 9223372036854775807);  slice_323 = None
    slice_325: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_324, 2, 0, 9223372036854775807);  slice_324 = None
    slice_326: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_325, 3, -257, 9223372036854775807);  slice_325 = None
    full_12: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_6: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_5, torch.bool);  expand_5 = None
    where_10: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_6, full_12, slice_326);  convert_element_type_6 = full_12 = slice_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_106: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_103, [1, 12, 1024, 513])
    permute_88: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    slice_331: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_88, 0, 0, 9223372036854775807);  permute_88 = None
    slice_332: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_331, 1, -256, 9223372036854775807);  slice_331 = None
    slice_333: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_332, 2, 0, 9223372036854775807);  slice_332 = None
    slice_334: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_333, 3, -257, 9223372036854775807);  slice_333 = None
    copy_17: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_334, where_10);  slice_334 = where_10 = None
    view_107: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_103, [1, 12, 1024, 513]);  view_103 = None
    permute_89: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    slice_335: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_89, 0, 0, 9223372036854775807)
    slice_336: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_335, 1, -256, 9223372036854775807)
    slice_337: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_336, 2, 0, 9223372036854775807)
    slice_scatter_62: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_337, copy_17, 3, -257, 9223372036854775807);  slice_337 = copy_17 = None
    slice_scatter_63: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_336, slice_scatter_62, 2, 0, 9223372036854775807);  slice_336 = slice_scatter_62 = None
    slice_scatter_64: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_335, slice_scatter_63, 1, -256, 9223372036854775807);  slice_335 = slice_scatter_63 = None
    slice_scatter_65: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_89, slice_scatter_64, 0, 0, 9223372036854775807);  permute_89 = slice_scatter_64 = None
    permute_90: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_65, [0, 2, 1, 3]);  slice_scatter_65 = None
    view_108: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_90, [12, 4, 256, 513]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_1: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_342: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_1, 0, 0, 9223372036854775807);  ne_1 = None
    slice_343: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_342, 1, 0, 9223372036854775807);  slice_342 = None
    unsqueeze_26: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_343, 2);  slice_343 = None
    unsqueeze_27: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 3);  unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_7: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_27, torch.float32)
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_27, scalar_tensor_5, convert_element_type_7);  unsqueeze_27 = scalar_tensor_5 = convert_element_type_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_13: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_92: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_13, [0, 2, 1, 3]);  full_13 = None
    view_110: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_92, [1, 1024, 1]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_93: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_11, [0, 2, 1, 3]);  where_11 = None
    view_111: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_93, [1, 1024, 1]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_112: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_110, [1, 2, 512, 1]);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_9: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_112, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_113: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_111, [1, 2, 512, 1]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_10: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_113, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_28: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_9, 4);  as_strided_9 = None
    permute_94: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_28, [0, 1, 2, 4, 3]);  unsqueeze_28 = None
    unsqueeze_29: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_10, 4);  as_strided_10 = None
    permute_95: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_29, [0, 1, 4, 2, 3]);  unsqueeze_29 = None
    mul_8: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_94, permute_95);  permute_94 = permute_95 = None
    view_114: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_8, [1, 3, 512, 512]);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_5: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_114, [0, 0, 0, 1], 0.0);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_115: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_5, [1, 3, 512, 513]);  constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_14: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_344: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_115, 0, 0, 9223372036854775807)
    slice_345: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_344, 1, 0, 9223372036854775807);  slice_344 = None
    slice_346: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_345, 2, 0, 256);  slice_345 = None
    slice_347: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_346, 3, 0, 257);  slice_346 = None
    slice_348: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_14, 0, 0, 9223372036854775807)
    slice_349: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_348, 1, 0, -1);  slice_348 = None
    slice_350: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_349, 2, 0, 9223372036854775807);  slice_349 = None
    slice_351: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_350, 3, 256, 9223372036854775807);  slice_350 = None
    copy_18: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_351, slice_347);  slice_351 = slice_347 = None
    slice_352: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_14, 0, 0, 9223372036854775807)
    slice_353: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_352, 1, 0, -1)
    slice_354: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_353, 2, 0, 9223372036854775807)
    slice_scatter_66: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_354, copy_18, 3, 256, 9223372036854775807);  slice_354 = copy_18 = None
    slice_scatter_67: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_353, slice_scatter_66, 2, 0, 9223372036854775807);  slice_353 = slice_scatter_66 = None
    slice_scatter_68: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_352, slice_scatter_67, 1, 0, -1);  slice_352 = slice_scatter_67 = None
    slice_scatter_69: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_14, slice_scatter_68, 0, 0, 9223372036854775807);  full_14 = slice_scatter_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_359: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_115, 0, 0, 9223372036854775807)
    select_30: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_359, 1, -1);  slice_359 = None
    slice_360: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_30, 1, 256, 9223372036854775807);  select_30 = None
    slice_361: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_360, 2, 0, 257);  slice_360 = None
    slice_365: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_69, 0, 0, 9223372036854775807)
    select_32: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_365, 1, -1);  slice_365 = None
    slice_366: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_32, 1, 0, 9223372036854775807);  select_32 = None
    slice_367: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_366, 2, 256, 9223372036854775807);  slice_366 = None
    copy_19: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_367, slice_361);  slice_367 = slice_361 = None
    slice_368: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_69, 0, 0, 9223372036854775807)
    select_33: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_368, 1, -1)
    slice_369: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_33, 1, 0, 9223372036854775807)
    slice_scatter_70: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_369, copy_19, 2, 256, 9223372036854775807);  slice_369 = copy_19 = None
    slice_scatter_71: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_33, slice_scatter_70, 1, 0, 9223372036854775807);  select_33 = slice_scatter_70 = None
    select_scatter_6: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_368, slice_scatter_71, 1, -1);  slice_368 = slice_scatter_71 = None
    slice_scatter_72: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_69, select_scatter_6, 0, 0, 9223372036854775807);  slice_scatter_69 = select_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_373: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_115, 0, 0, 9223372036854775807)
    slice_374: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_373, 1, 0, 9223372036854775807);  slice_373 = None
    slice_375: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_374, 2, -257, -1);  slice_374 = None
    slice_376: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_375, 3, 257, 9223372036854775807);  slice_375 = None
    slice_381: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_72, 0, 0, 9223372036854775807)
    slice_382: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_381, 1, 1, 9223372036854775807);  slice_381 = None
    slice_383: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_382, 2, 0, 9223372036854775807);  slice_382 = None
    slice_384: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_383, 3, 0, 256);  slice_383 = None
    copy_20: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_384, slice_376);  slice_384 = slice_376 = None
    slice_385: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_72, 0, 0, 9223372036854775807)
    slice_386: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_385, 1, 1, 9223372036854775807)
    slice_387: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_386, 2, 0, 9223372036854775807)
    slice_scatter_73: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_387, copy_20, 3, 0, 256);  slice_387 = copy_20 = None
    slice_scatter_74: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_386, slice_scatter_73, 2, 0, 9223372036854775807);  slice_386 = slice_scatter_73 = None
    slice_scatter_75: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_385, slice_scatter_74, 1, 1, 9223372036854775807);  slice_385 = slice_scatter_74 = None
    slice_scatter_76: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_72, slice_scatter_75, 0, 0, 9223372036854775807);  slice_scatter_72 = slice_scatter_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_392: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_115, 0, 0, 9223372036854775807);  view_115 = None
    select_35: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_392, 1, 0);  slice_392 = None
    slice_393: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_35, 1, 0, 255);  select_35 = None
    slice_394: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_393, 2, -255, 9223372036854775807);  slice_393 = None
    slice_398: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_76, 0, 0, 9223372036854775807)
    select_37: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_398, 1, 0);  slice_398 = None
    slice_399: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_37, 1, 1, 256);  select_37 = None
    slice_400: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_399, 2, 1, 256);  slice_399 = None
    copy_21: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_400, slice_394);  slice_400 = slice_394 = None
    slice_401: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_76, 0, 0, 9223372036854775807)
    select_38: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_401, 1, 0)
    slice_402: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_38, 1, 1, 256)
    slice_scatter_77: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_402, copy_21, 2, 1, 256);  slice_402 = copy_21 = None
    slice_scatter_78: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_38, slice_scatter_77, 1, 1, 256);  select_38 = slice_scatter_77 = None
    select_scatter_7: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_401, slice_scatter_78, 1, 0);  slice_401 = slice_scatter_78 = None
    slice_scatter_79: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_76, select_scatter_7, 0, 0, 9223372036854775807);  slice_scatter_76 = select_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_15: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_6: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_30: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_6, -2);  iota_6 = None
    iota_7: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_31: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
    sub_11: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_30, unsqueeze_31);  unsqueeze_30 = unsqueeze_31 = None
    le_3: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_11, 0);  sub_11 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[256, 257]" = torch.ops.aten.where.self(le_3, full_15, scalar_tensor_6);  le_3 = full_15 = scalar_tensor_6 = None
    rev_6: "f32[256, 257]" = torch.ops.prims.rev.default(where_12, [0]);  where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_32: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_6, 0);  rev_6 = None
    slice_406: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_32, 1, 0, 9223372036854775807);  unsqueeze_32 = None
    unsqueeze_33: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_406, 2);  slice_406 = None
    slice_407: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_33, 3, 0, 9223372036854775807);  unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_7: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_407, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_6: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_407, [1, 256, 1, 257]);  slice_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_118: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_79, [1, 1, 1024, 513])
    permute_98: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    slice_412: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_98, 0, 0, 9223372036854775807);  permute_98 = None
    slice_413: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_412, 1, 0, 256);  slice_412 = None
    slice_414: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_413, 2, 0, 9223372036854775807);  slice_413 = None
    slice_415: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_414, 3, 0, 257);  slice_414 = None
    full_16: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_8: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_6, torch.bool);  expand_6 = None
    where_13: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_8, full_16, slice_415);  convert_element_type_8 = full_16 = slice_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_119: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_79, [1, 1, 1024, 513])
    permute_99: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    slice_420: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_99, 0, 0, 9223372036854775807);  permute_99 = None
    slice_421: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_420, 1, 0, 256);  slice_420 = None
    slice_422: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_421, 2, 0, 9223372036854775807);  slice_421 = None
    slice_423: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_422, 3, 0, 257);  slice_422 = None
    copy_22: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_423, where_13);  slice_423 = where_13 = None
    view_120: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_79, [1, 1, 1024, 513]);  slice_scatter_79 = None
    permute_100: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    slice_424: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_100, 0, 0, 9223372036854775807)
    slice_425: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_424, 1, 0, 256)
    slice_426: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_425, 2, 0, 9223372036854775807)
    slice_scatter_80: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_426, copy_22, 3, 0, 257);  slice_426 = copy_22 = None
    slice_scatter_81: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_425, slice_scatter_80, 2, 0, 9223372036854775807);  slice_425 = slice_scatter_80 = None
    slice_scatter_82: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_424, slice_scatter_81, 1, 0, 256);  slice_424 = slice_scatter_81 = None
    slice_scatter_83: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_100, slice_scatter_82, 0, 0, 9223372036854775807);  permute_100 = slice_scatter_82 = None
    permute_101: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_83, [0, 2, 1, 3]);  slice_scatter_83 = None
    view_121: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_101, [1, 4, 256, 513]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_7: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_7, [1, 256, 1, 257]);  rev_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_123: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_121, [1, 1, 1024, 513])
    permute_103: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    slice_435: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_103, 0, 0, 9223372036854775807);  permute_103 = None
    slice_436: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_435, 1, -256, 9223372036854775807);  slice_435 = None
    slice_437: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_436, 2, 0, 9223372036854775807);  slice_436 = None
    slice_438: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_437, 3, -257, 9223372036854775807);  slice_437 = None
    full_17: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_9: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_7, torch.bool);  expand_7 = None
    where_14: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_9, full_17, slice_438);  convert_element_type_9 = full_17 = slice_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_124: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_121, [1, 1, 1024, 513])
    permute_104: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
    slice_443: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_104, 0, 0, 9223372036854775807);  permute_104 = None
    slice_444: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_443, 1, -256, 9223372036854775807);  slice_443 = None
    slice_445: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_444, 2, 0, 9223372036854775807);  slice_444 = None
    slice_446: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_445, 3, -257, 9223372036854775807);  slice_445 = None
    copy_23: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_446, where_14);  slice_446 = where_14 = None
    view_125: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_121, [1, 1, 1024, 513]);  view_121 = None
    permute_105: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
    slice_447: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_105, 0, 0, 9223372036854775807)
    slice_448: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_447, 1, -256, 9223372036854775807)
    slice_449: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_448, 2, 0, 9223372036854775807)
    slice_scatter_84: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_449, copy_23, 3, -257, 9223372036854775807);  slice_449 = copy_23 = None
    slice_scatter_85: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_448, slice_scatter_84, 2, 0, 9223372036854775807);  slice_448 = slice_scatter_84 = None
    slice_scatter_86: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_447, slice_scatter_85, 1, -256, 9223372036854775807);  slice_447 = slice_scatter_85 = None
    slice_scatter_87: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_105, slice_scatter_86, 0, 0, 9223372036854775807);  permute_105 = slice_scatter_86 = None
    permute_106: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_87, [0, 2, 1, 3]);  slice_scatter_87 = None
    view_126: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_106, [1, 4, 256, 513]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_128: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_108, [1, 12, 1024, 513]);  view_108 = None
    permute_108: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    view_129: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_126, [1, 1, 1024, 513]);  view_126 = None
    permute_109: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    add_13: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_108, permute_109);  permute_108 = permute_109 = None
    permute_110: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_13, [0, 2, 1, 3]);  add_13 = None
    view_131: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_110, [12, 4, 256, 513]);  permute_110 = None
    view_132: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_131, [1, 12, 1024, 513]);  view_131 = None
    permute_111: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_10: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    amax_1: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_10, [-1], True)
    sub_12: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_10, amax_1);  clone_10 = amax_1 = None
    exp_1: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_2: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_17: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_454: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_455: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_454, 1, 0, 9223372036854775807);  slice_454 = None
    unsqueeze_34: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_455, 2);  slice_455 = None
    unsqueeze_35: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_35, scalar_tensor_7, div_17);  unsqueeze_35 = scalar_tensor_7 = div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_11: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_15);  where_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_133: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_80, [1024, 1, 12, 64]);  view_80 = None
    permute_112: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_133, [1, 0, 2, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_113: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_11, [0, 2, 1, 3]);  clone_11 = None
    view_134: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_113, [12, 4, 256, 513]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_114: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_112, [0, 2, 1, 3]);  permute_112 = None
    view_135: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_114, [12, 1024, 64]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_6: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_135, [0, 0, 256, 256], -1.0);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_11: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_6, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_7: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_134, [0, 257], 0.0);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_136: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_7, [12, 4, -1]);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_456: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_136, 0, 0, 9223372036854775807);  view_136 = None
    slice_457: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_456, 1, 0, 9223372036854775807);  slice_456 = None
    slice_458: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_457, 2, 0, -256);  slice_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_137: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_458, [12, 4, 256, 769]);  slice_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_459: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_137, 0, 0, 9223372036854775807);  view_137 = None
    slice_460: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_459, 1, 0, 9223372036854775807);  slice_459 = None
    slice_461: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_460, 2, 0, 9223372036854775807);  slice_460 = None
    slice_462: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_461, 3, 0, -1);  slice_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_36: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_462, 4);  slice_462 = None
    permute_115: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_36, [0, 1, 2, 4, 3]);  unsqueeze_36 = None
    unsqueeze_37: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_11, 4);  as_strided_11 = None
    permute_116: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_37, [0, 1, 4, 3, 2]);  unsqueeze_37 = None
    permute_117: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_115, [0, 1, 2, 4, 3]);  permute_115 = None
    view_138: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_117, [48, 256, 768]);  permute_117 = None
    permute_118: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_116, [0, 1, 4, 3, 2]);  permute_116 = None
    clone_12: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_139: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_12, [48, 768, 64]);  clone_12 = None
    bmm_3: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_138, view_139);  view_138 = view_139 = None
    view_140: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_3, [12, 4, 256, 1, 64]);  bmm_3 = None
    permute_119: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_140, [0, 1, 2, 4, 3]);  view_140 = None
    view_141: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_119, [12, 4, 256, 64]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_142: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_141, [1, 12, 1024, 64]);  view_141 = None
    permute_120: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_121: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_120, [1, 0, 2, 3]);  permute_120 = None
    clone_13: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_143: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_13, [1024, 1, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_122: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_143, [1, 0, 2]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_144: "f32[1024, 768]" = torch.ops.aten.view.default(permute_122, [1024, 768]);  permute_122 = None
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    addmm_9: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg23_1, view_144, permute_123);  arg23_1 = view_144 = permute_123 = None
    view_145: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_9, [1, 1024, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_14: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_145);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_14, add_10);  clone_14 = add_10 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    add_16: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_14: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_5);  add_15 = getitem_5 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_2);  sub_14 = rsqrt_2 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg24_1);  mul_9 = arg24_1 = None
    add_17: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_10, arg25_1);  mul_10 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_146: "f32[1024, 768]" = torch.ops.aten.view.default(add_17, [1024, 768])
    permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_10: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg27_1, view_146, permute_124);  arg27_1 = view_146 = permute_124 = None
    view_147: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 1024, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_11: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_147, 0.5)
    mul_12: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_147, 0.7071067811865476);  view_147 = None
    erf_1: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_18: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_18);  mul_11 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_13, [1024, 3072]);  mul_13 = None
    permute_125: "f32[3072, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
    addmm_11: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg29_1, view_148, permute_125);  arg29_1 = view_148 = permute_125 = None
    view_149: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_11, [1, 1024, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_15: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_149);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_19: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_15, add_17);  clone_15 = add_17 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    add_20: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_15: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_7);  add_19 = getitem_7 = None
    mul_14: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_3);  sub_15 = rsqrt_3 = None
    mul_15: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg30_1);  mul_14 = arg30_1 = None
    add_21: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_15, arg31_1);  mul_15 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_126: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_21, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_150: "f32[1024, 768]" = torch.ops.aten.view.default(permute_126, [1024, 768])
    permute_127: "f32[768, 768]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    addmm_12: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg33_1, view_150, permute_127);  arg33_1 = view_150 = permute_127 = None
    view_151: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_12, [1024, 1, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_152: "f32[1024, 768]" = torch.ops.aten.view.default(permute_126, [1024, 768])
    permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
    addmm_13: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg35_1, view_152, permute_128);  arg35_1 = view_152 = permute_128 = None
    view_153: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_13, [1024, 1, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_154: "f32[1024, 768]" = torch.ops.aten.view.default(permute_126, [1024, 768]);  permute_126 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_14: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg37_1, view_154, permute_129);  arg37_1 = view_154 = permute_129 = None
    view_155: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_14, [1024, 1, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_20: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_151, 8.0);  view_151 = None
    view_156: "f32[1024, 768]" = torch.ops.aten.view.default(div_20, [1024, 768]);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_159: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_153, [1024, 1, 12, 64]);  view_153 = None
    permute_131: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_159, [1, 0, 2, 3]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_133: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_131, [0, 2, 1, 3]);  permute_131 = None
    view_161: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_133, [12, 1024, 64]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_163: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_161, [12, 2, 512, 64]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_13: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_163, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_39: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_13, 4);  as_strided_13 = None
    permute_135: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_39, [0, 1, 4, 2, 3]);  unsqueeze_39 = None
    view_164: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_156, [1024, 1, 768]);  view_156 = None
    view_165: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_164, [1024, 1, 12, 64]);  view_164 = None
    permute_137: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_165, [1, 0, 2, 3]);  view_165 = None
    permute_138: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_137, [0, 2, 1, 3]);  permute_137 = None
    view_166: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_138, [12, 1024, 64]);  permute_138 = None
    view_167: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_166, [12, 2, 512, 64]);  view_166 = None
    as_strided_14: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_167, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_167 = None
    unsqueeze_40: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_14, 4);  as_strided_14 = None
    permute_139: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_40, [0, 1, 2, 4, 3]);  unsqueeze_40 = None
    permute_140: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_139, [0, 1, 2, 4, 3]);  permute_139 = None
    clone_16: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    view_168: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_16, [36, 512, 64]);  clone_16 = None
    permute_141: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_135, [0, 1, 4, 3, 2]);  permute_135 = None
    clone_17: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
    view_169: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_17, [36, 64, 512]);  clone_17 = None
    bmm_4: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
    view_170: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_4, [12, 3, 512, 1, 512]);  bmm_4 = None
    permute_142: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_170, [0, 1, 2, 4, 3]);  view_170 = None
    view_171: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_142, [12, 3, 512, 512]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_8: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_171, [0, 0, 0, 1], 0.0);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_172: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_8, [12, 3, 512, 513]);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_18: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_463: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_172, 0, 0, 9223372036854775807)
    slice_464: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_463, 1, 0, 9223372036854775807);  slice_463 = None
    slice_465: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_464, 2, 0, 256);  slice_464 = None
    slice_466: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_465, 3, 0, 257);  slice_465 = None
    slice_467: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_18, 0, 0, 9223372036854775807)
    slice_468: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_467, 1, 0, -1);  slice_467 = None
    slice_469: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_468, 2, 0, 9223372036854775807);  slice_468 = None
    slice_470: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_469, 3, 256, 9223372036854775807);  slice_469 = None
    copy_24: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_470, slice_466);  slice_470 = slice_466 = None
    slice_471: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_18, 0, 0, 9223372036854775807)
    slice_472: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_471, 1, 0, -1)
    slice_473: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_472, 2, 0, 9223372036854775807)
    slice_scatter_88: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_473, copy_24, 3, 256, 9223372036854775807);  slice_473 = copy_24 = None
    slice_scatter_89: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_472, slice_scatter_88, 2, 0, 9223372036854775807);  slice_472 = slice_scatter_88 = None
    slice_scatter_90: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_471, slice_scatter_89, 1, 0, -1);  slice_471 = slice_scatter_89 = None
    slice_scatter_91: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_18, slice_scatter_90, 0, 0, 9223372036854775807);  full_18 = slice_scatter_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_478: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_172, 0, 0, 9223372036854775807)
    select_40: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_478, 1, -1);  slice_478 = None
    slice_479: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_40, 1, 256, 9223372036854775807);  select_40 = None
    slice_480: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_479, 2, 0, 257);  slice_479 = None
    slice_484: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_91, 0, 0, 9223372036854775807)
    select_42: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_484, 1, -1);  slice_484 = None
    slice_485: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_42, 1, 0, 9223372036854775807);  select_42 = None
    slice_486: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_485, 2, 256, 9223372036854775807);  slice_485 = None
    copy_25: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_486, slice_480);  slice_486 = slice_480 = None
    slice_487: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_91, 0, 0, 9223372036854775807)
    select_43: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_487, 1, -1)
    slice_488: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_43, 1, 0, 9223372036854775807)
    slice_scatter_92: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_488, copy_25, 2, 256, 9223372036854775807);  slice_488 = copy_25 = None
    slice_scatter_93: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_43, slice_scatter_92, 1, 0, 9223372036854775807);  select_43 = slice_scatter_92 = None
    select_scatter_8: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_487, slice_scatter_93, 1, -1);  slice_487 = slice_scatter_93 = None
    slice_scatter_94: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_91, select_scatter_8, 0, 0, 9223372036854775807);  slice_scatter_91 = select_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_492: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_172, 0, 0, 9223372036854775807)
    slice_493: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_492, 1, 0, 9223372036854775807);  slice_492 = None
    slice_494: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_493, 2, -257, -1);  slice_493 = None
    slice_495: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_494, 3, 257, 9223372036854775807);  slice_494 = None
    slice_500: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_94, 0, 0, 9223372036854775807)
    slice_501: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_500, 1, 1, 9223372036854775807);  slice_500 = None
    slice_502: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_501, 2, 0, 9223372036854775807);  slice_501 = None
    slice_503: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_502, 3, 0, 256);  slice_502 = None
    copy_26: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_503, slice_495);  slice_503 = slice_495 = None
    slice_504: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_94, 0, 0, 9223372036854775807)
    slice_505: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_504, 1, 1, 9223372036854775807)
    slice_506: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_505, 2, 0, 9223372036854775807)
    slice_scatter_95: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_506, copy_26, 3, 0, 256);  slice_506 = copy_26 = None
    slice_scatter_96: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_505, slice_scatter_95, 2, 0, 9223372036854775807);  slice_505 = slice_scatter_95 = None
    slice_scatter_97: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_504, slice_scatter_96, 1, 1, 9223372036854775807);  slice_504 = slice_scatter_96 = None
    slice_scatter_98: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_94, slice_scatter_97, 0, 0, 9223372036854775807);  slice_scatter_94 = slice_scatter_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_511: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_172, 0, 0, 9223372036854775807);  view_172 = None
    select_45: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_511, 1, 0);  slice_511 = None
    slice_512: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_45, 1, 0, 255);  select_45 = None
    slice_513: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_512, 2, -255, 9223372036854775807);  slice_512 = None
    slice_517: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_98, 0, 0, 9223372036854775807)
    select_47: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_517, 1, 0);  slice_517 = None
    slice_518: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_47, 1, 1, 256);  select_47 = None
    slice_519: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_518, 2, 1, 256);  slice_518 = None
    copy_27: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_519, slice_513);  slice_519 = slice_513 = None
    slice_520: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_98, 0, 0, 9223372036854775807)
    select_48: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_520, 1, 0)
    slice_521: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_48, 1, 1, 256)
    slice_scatter_99: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_521, copy_27, 2, 1, 256);  slice_521 = copy_27 = None
    slice_scatter_100: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_48, slice_scatter_99, 1, 1, 256);  select_48 = slice_scatter_99 = None
    select_scatter_9: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_520, slice_scatter_100, 1, 0);  slice_520 = slice_scatter_100 = None
    slice_scatter_101: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_98, select_scatter_9, 0, 0, 9223372036854775807);  slice_scatter_98 = select_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_19: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_8: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_41: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_8, -2);  iota_8 = None
    iota_9: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_42: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
    sub_17: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_41, unsqueeze_42);  unsqueeze_41 = unsqueeze_42 = None
    le_4: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_17, 0);  sub_17 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[256, 257]" = torch.ops.aten.where.self(le_4, full_19, scalar_tensor_8);  le_4 = full_19 = scalar_tensor_8 = None
    rev_8: "f32[256, 257]" = torch.ops.prims.rev.default(where_16, [0]);  where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_43: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_8, 0);  rev_8 = None
    slice_525: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_43, 1, 0, 9223372036854775807);  unsqueeze_43 = None
    unsqueeze_44: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_525, 2);  slice_525 = None
    slice_526: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_44, 3, 0, 9223372036854775807);  unsqueeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_9: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_526, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_8: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_526, [1, 256, 12, 257]);  slice_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_175: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_101, [1, 12, 1024, 513])
    permute_145: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    slice_531: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_145, 0, 0, 9223372036854775807);  permute_145 = None
    slice_532: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_531, 1, 0, 256);  slice_531 = None
    slice_533: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_532, 2, 0, 9223372036854775807);  slice_532 = None
    slice_534: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_533, 3, 0, 257);  slice_533 = None
    full_20: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_10: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_8, torch.bool);  expand_8 = None
    where_17: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_10, full_20, slice_534);  convert_element_type_10 = full_20 = slice_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_176: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_101, [1, 12, 1024, 513])
    permute_146: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    slice_539: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_146, 0, 0, 9223372036854775807);  permute_146 = None
    slice_540: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_539, 1, 0, 256);  slice_539 = None
    slice_541: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_540, 2, 0, 9223372036854775807);  slice_540 = None
    slice_542: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_541, 3, 0, 257);  slice_541 = None
    copy_28: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_542, where_17);  slice_542 = where_17 = None
    view_177: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_101, [1, 12, 1024, 513]);  slice_scatter_101 = None
    permute_147: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
    slice_543: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_147, 0, 0, 9223372036854775807)
    slice_544: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_543, 1, 0, 256)
    slice_545: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_544, 2, 0, 9223372036854775807)
    slice_scatter_102: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_545, copy_28, 3, 0, 257);  slice_545 = copy_28 = None
    slice_scatter_103: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_544, slice_scatter_102, 2, 0, 9223372036854775807);  slice_544 = slice_scatter_102 = None
    slice_scatter_104: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_543, slice_scatter_103, 1, 0, 256);  slice_543 = slice_scatter_103 = None
    slice_scatter_105: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_147, slice_scatter_104, 0, 0, 9223372036854775807);  permute_147 = slice_scatter_104 = None
    permute_148: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_105, [0, 2, 1, 3]);  slice_scatter_105 = None
    view_178: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_148, [12, 4, 256, 513]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_9: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_9, [1, 256, 12, 257]);  rev_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_180: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_178, [1, 12, 1024, 513])
    permute_150: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    slice_554: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_150, 0, 0, 9223372036854775807);  permute_150 = None
    slice_555: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_554, 1, -256, 9223372036854775807);  slice_554 = None
    slice_556: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_555, 2, 0, 9223372036854775807);  slice_555 = None
    slice_557: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_556, 3, -257, 9223372036854775807);  slice_556 = None
    full_21: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_11: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_9, torch.bool);  expand_9 = None
    where_18: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_11, full_21, slice_557);  convert_element_type_11 = full_21 = slice_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_181: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_178, [1, 12, 1024, 513])
    permute_151: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
    slice_562: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_151, 0, 0, 9223372036854775807);  permute_151 = None
    slice_563: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_562, 1, -256, 9223372036854775807);  slice_562 = None
    slice_564: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_563, 2, 0, 9223372036854775807);  slice_563 = None
    slice_565: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_564, 3, -257, 9223372036854775807);  slice_564 = None
    copy_29: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_565, where_18);  slice_565 = where_18 = None
    view_182: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_178, [1, 12, 1024, 513]);  view_178 = None
    permute_152: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    slice_566: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_152, 0, 0, 9223372036854775807)
    slice_567: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_566, 1, -256, 9223372036854775807)
    slice_568: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_567, 2, 0, 9223372036854775807)
    slice_scatter_106: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_568, copy_29, 3, -257, 9223372036854775807);  slice_568 = copy_29 = None
    slice_scatter_107: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_567, slice_scatter_106, 2, 0, 9223372036854775807);  slice_567 = slice_scatter_106 = None
    slice_scatter_108: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_566, slice_scatter_107, 1, -256, 9223372036854775807);  slice_566 = slice_scatter_107 = None
    slice_scatter_109: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_152, slice_scatter_108, 0, 0, 9223372036854775807);  permute_152 = slice_scatter_108 = None
    permute_153: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_109, [0, 2, 1, 3]);  slice_scatter_109 = None
    view_183: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_153, [12, 4, 256, 513]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_2: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_573: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_2, 0, 0, 9223372036854775807);  ne_2 = None
    slice_574: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_573, 1, 0, 9223372036854775807);  slice_573 = None
    unsqueeze_45: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_574, 2);  slice_574 = None
    unsqueeze_46: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 3);  unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_12: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_46, torch.float32)
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_46, scalar_tensor_9, convert_element_type_12);  unsqueeze_46 = scalar_tensor_9 = convert_element_type_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_22: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_155: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_22, [0, 2, 1, 3]);  full_22 = None
    view_185: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_155, [1, 1024, 1]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_156: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_19, [0, 2, 1, 3]);  where_19 = None
    view_186: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_156, [1, 1024, 1]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_187: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_185, [1, 2, 512, 1]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_15: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_187, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_188: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_186, [1, 2, 512, 1]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_16: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_188, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_47: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_15, 4);  as_strided_15 = None
    permute_157: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_47, [0, 1, 2, 4, 3]);  unsqueeze_47 = None
    unsqueeze_48: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_16, 4);  as_strided_16 = None
    permute_158: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_48, [0, 1, 4, 2, 3]);  unsqueeze_48 = None
    mul_16: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_157, permute_158);  permute_157 = permute_158 = None
    view_189: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_16, [1, 3, 512, 512]);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_9: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_189, [0, 0, 0, 1], 0.0);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_190: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_9, [1, 3, 512, 513]);  constant_pad_nd_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_23: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_575: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_190, 0, 0, 9223372036854775807)
    slice_576: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_575, 1, 0, 9223372036854775807);  slice_575 = None
    slice_577: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_576, 2, 0, 256);  slice_576 = None
    slice_578: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_577, 3, 0, 257);  slice_577 = None
    slice_579: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_23, 0, 0, 9223372036854775807)
    slice_580: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_579, 1, 0, -1);  slice_579 = None
    slice_581: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_580, 2, 0, 9223372036854775807);  slice_580 = None
    slice_582: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_581, 3, 256, 9223372036854775807);  slice_581 = None
    copy_30: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_582, slice_578);  slice_582 = slice_578 = None
    slice_583: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_23, 0, 0, 9223372036854775807)
    slice_584: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_583, 1, 0, -1)
    slice_585: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_584, 2, 0, 9223372036854775807)
    slice_scatter_110: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_585, copy_30, 3, 256, 9223372036854775807);  slice_585 = copy_30 = None
    slice_scatter_111: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_584, slice_scatter_110, 2, 0, 9223372036854775807);  slice_584 = slice_scatter_110 = None
    slice_scatter_112: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_583, slice_scatter_111, 1, 0, -1);  slice_583 = slice_scatter_111 = None
    slice_scatter_113: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_23, slice_scatter_112, 0, 0, 9223372036854775807);  full_23 = slice_scatter_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_590: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_190, 0, 0, 9223372036854775807)
    select_50: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_590, 1, -1);  slice_590 = None
    slice_591: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_50, 1, 256, 9223372036854775807);  select_50 = None
    slice_592: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_591, 2, 0, 257);  slice_591 = None
    slice_596: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_113, 0, 0, 9223372036854775807)
    select_52: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_596, 1, -1);  slice_596 = None
    slice_597: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_52, 1, 0, 9223372036854775807);  select_52 = None
    slice_598: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_597, 2, 256, 9223372036854775807);  slice_597 = None
    copy_31: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_598, slice_592);  slice_598 = slice_592 = None
    slice_599: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_113, 0, 0, 9223372036854775807)
    select_53: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_599, 1, -1)
    slice_600: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_53, 1, 0, 9223372036854775807)
    slice_scatter_114: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_600, copy_31, 2, 256, 9223372036854775807);  slice_600 = copy_31 = None
    slice_scatter_115: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_53, slice_scatter_114, 1, 0, 9223372036854775807);  select_53 = slice_scatter_114 = None
    select_scatter_10: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_599, slice_scatter_115, 1, -1);  slice_599 = slice_scatter_115 = None
    slice_scatter_116: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_113, select_scatter_10, 0, 0, 9223372036854775807);  slice_scatter_113 = select_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_604: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_190, 0, 0, 9223372036854775807)
    slice_605: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_604, 1, 0, 9223372036854775807);  slice_604 = None
    slice_606: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_605, 2, -257, -1);  slice_605 = None
    slice_607: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_606, 3, 257, 9223372036854775807);  slice_606 = None
    slice_612: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_116, 0, 0, 9223372036854775807)
    slice_613: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_612, 1, 1, 9223372036854775807);  slice_612 = None
    slice_614: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_613, 2, 0, 9223372036854775807);  slice_613 = None
    slice_615: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_614, 3, 0, 256);  slice_614 = None
    copy_32: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_615, slice_607);  slice_615 = slice_607 = None
    slice_616: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_116, 0, 0, 9223372036854775807)
    slice_617: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_616, 1, 1, 9223372036854775807)
    slice_618: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_617, 2, 0, 9223372036854775807)
    slice_scatter_117: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_618, copy_32, 3, 0, 256);  slice_618 = copy_32 = None
    slice_scatter_118: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_617, slice_scatter_117, 2, 0, 9223372036854775807);  slice_617 = slice_scatter_117 = None
    slice_scatter_119: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_616, slice_scatter_118, 1, 1, 9223372036854775807);  slice_616 = slice_scatter_118 = None
    slice_scatter_120: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_116, slice_scatter_119, 0, 0, 9223372036854775807);  slice_scatter_116 = slice_scatter_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_623: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_190, 0, 0, 9223372036854775807);  view_190 = None
    select_55: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_623, 1, 0);  slice_623 = None
    slice_624: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_55, 1, 0, 255);  select_55 = None
    slice_625: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_624, 2, -255, 9223372036854775807);  slice_624 = None
    slice_629: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_120, 0, 0, 9223372036854775807)
    select_57: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_629, 1, 0);  slice_629 = None
    slice_630: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_57, 1, 1, 256);  select_57 = None
    slice_631: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_630, 2, 1, 256);  slice_630 = None
    copy_33: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_631, slice_625);  slice_631 = slice_625 = None
    slice_632: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_120, 0, 0, 9223372036854775807)
    select_58: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_632, 1, 0)
    slice_633: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_58, 1, 1, 256)
    slice_scatter_121: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_633, copy_33, 2, 1, 256);  slice_633 = copy_33 = None
    slice_scatter_122: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_58, slice_scatter_121, 1, 1, 256);  select_58 = slice_scatter_121 = None
    select_scatter_11: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_632, slice_scatter_122, 1, 0);  slice_632 = slice_scatter_122 = None
    slice_scatter_123: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_120, select_scatter_11, 0, 0, 9223372036854775807);  slice_scatter_120 = select_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_24: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_10: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_49: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_10, -2);  iota_10 = None
    iota_11: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_50: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
    sub_19: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_49, unsqueeze_50);  unsqueeze_49 = unsqueeze_50 = None
    le_5: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_19, 0);  sub_19 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_20: "f32[256, 257]" = torch.ops.aten.where.self(le_5, full_24, scalar_tensor_10);  le_5 = full_24 = scalar_tensor_10 = None
    rev_10: "f32[256, 257]" = torch.ops.prims.rev.default(where_20, [0]);  where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_51: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_10, 0);  rev_10 = None
    slice_637: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_51, 1, 0, 9223372036854775807);  unsqueeze_51 = None
    unsqueeze_52: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_637, 2);  slice_637 = None
    slice_638: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_52, 3, 0, 9223372036854775807);  unsqueeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_11: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_638, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_10: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_638, [1, 256, 1, 257]);  slice_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_193: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_123, [1, 1, 1024, 513])
    permute_161: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
    slice_643: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_161, 0, 0, 9223372036854775807);  permute_161 = None
    slice_644: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_643, 1, 0, 256);  slice_643 = None
    slice_645: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_644, 2, 0, 9223372036854775807);  slice_644 = None
    slice_646: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_645, 3, 0, 257);  slice_645 = None
    full_25: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_13: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_10, torch.bool);  expand_10 = None
    where_21: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_13, full_25, slice_646);  convert_element_type_13 = full_25 = slice_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_194: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_123, [1, 1, 1024, 513])
    permute_162: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
    slice_651: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_162, 0, 0, 9223372036854775807);  permute_162 = None
    slice_652: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_651, 1, 0, 256);  slice_651 = None
    slice_653: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_652, 2, 0, 9223372036854775807);  slice_652 = None
    slice_654: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_653, 3, 0, 257);  slice_653 = None
    copy_34: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_654, where_21);  slice_654 = where_21 = None
    view_195: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_123, [1, 1, 1024, 513]);  slice_scatter_123 = None
    permute_163: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
    slice_655: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_163, 0, 0, 9223372036854775807)
    slice_656: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_655, 1, 0, 256)
    slice_657: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_656, 2, 0, 9223372036854775807)
    slice_scatter_124: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_657, copy_34, 3, 0, 257);  slice_657 = copy_34 = None
    slice_scatter_125: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_656, slice_scatter_124, 2, 0, 9223372036854775807);  slice_656 = slice_scatter_124 = None
    slice_scatter_126: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_655, slice_scatter_125, 1, 0, 256);  slice_655 = slice_scatter_125 = None
    slice_scatter_127: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_163, slice_scatter_126, 0, 0, 9223372036854775807);  permute_163 = slice_scatter_126 = None
    permute_164: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_127, [0, 2, 1, 3]);  slice_scatter_127 = None
    view_196: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_164, [1, 4, 256, 513]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_11: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_11, [1, 256, 1, 257]);  rev_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_198: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_196, [1, 1, 1024, 513])
    permute_166: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
    slice_666: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_166, 0, 0, 9223372036854775807);  permute_166 = None
    slice_667: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_666, 1, -256, 9223372036854775807);  slice_666 = None
    slice_668: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_667, 2, 0, 9223372036854775807);  slice_667 = None
    slice_669: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_668, 3, -257, 9223372036854775807);  slice_668 = None
    full_26: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_14: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_11, torch.bool);  expand_11 = None
    where_22: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_14, full_26, slice_669);  convert_element_type_14 = full_26 = slice_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_199: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_196, [1, 1, 1024, 513])
    permute_167: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
    slice_674: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_167, 0, 0, 9223372036854775807);  permute_167 = None
    slice_675: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_674, 1, -256, 9223372036854775807);  slice_674 = None
    slice_676: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_675, 2, 0, 9223372036854775807);  slice_675 = None
    slice_677: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_676, 3, -257, 9223372036854775807);  slice_676 = None
    copy_35: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_677, where_22);  slice_677 = where_22 = None
    view_200: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_196, [1, 1, 1024, 513]);  view_196 = None
    permute_168: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
    slice_678: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_168, 0, 0, 9223372036854775807)
    slice_679: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_678, 1, -256, 9223372036854775807)
    slice_680: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_679, 2, 0, 9223372036854775807)
    slice_scatter_128: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_680, copy_35, 3, -257, 9223372036854775807);  slice_680 = copy_35 = None
    slice_scatter_129: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_679, slice_scatter_128, 2, 0, 9223372036854775807);  slice_679 = slice_scatter_128 = None
    slice_scatter_130: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_678, slice_scatter_129, 1, -256, 9223372036854775807);  slice_678 = slice_scatter_129 = None
    slice_scatter_131: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_168, slice_scatter_130, 0, 0, 9223372036854775807);  permute_168 = slice_scatter_130 = None
    permute_169: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_131, [0, 2, 1, 3]);  slice_scatter_131 = None
    view_201: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_169, [1, 4, 256, 513]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_203: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_183, [1, 12, 1024, 513]);  view_183 = None
    permute_171: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    view_204: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_201, [1, 1, 1024, 513]);  view_201 = None
    permute_172: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    add_24: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_171, permute_172);  permute_171 = permute_172 = None
    permute_173: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_24, [0, 2, 1, 3]);  add_24 = None
    view_206: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_173, [12, 4, 256, 513]);  permute_173 = None
    view_207: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_206, [1, 12, 1024, 513]);  view_206 = None
    permute_174: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_18: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
    amax_2: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_18, [-1], True)
    sub_20: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_18, amax_2);  clone_18 = amax_2 = None
    exp_2: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_3: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_27: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_685: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_686: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_685, 1, 0, 9223372036854775807);  slice_685 = None
    unsqueeze_53: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_686, 2);  slice_686 = None
    unsqueeze_54: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 3);  unsqueeze_53 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_54, scalar_tensor_11, div_27);  unsqueeze_54 = scalar_tensor_11 = div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_19: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_23);  where_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_208: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_155, [1024, 1, 12, 64]);  view_155 = None
    permute_175: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_208, [1, 0, 2, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_176: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_19, [0, 2, 1, 3]);  clone_19 = None
    view_209: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_176, [12, 4, 256, 513]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_177: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_175, [0, 2, 1, 3]);  permute_175 = None
    view_210: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_177, [12, 1024, 64]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_10: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_210, [0, 0, 256, 256], -1.0);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_17: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_10, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_11: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_209, [0, 257], 0.0);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_211: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_11, [12, 4, -1]);  constant_pad_nd_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_687: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_211, 0, 0, 9223372036854775807);  view_211 = None
    slice_688: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_687, 1, 0, 9223372036854775807);  slice_687 = None
    slice_689: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_688, 2, 0, -256);  slice_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_212: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_689, [12, 4, 256, 769]);  slice_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_690: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_212, 0, 0, 9223372036854775807);  view_212 = None
    slice_691: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_690, 1, 0, 9223372036854775807);  slice_690 = None
    slice_692: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_691, 2, 0, 9223372036854775807);  slice_691 = None
    slice_693: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_692, 3, 0, -1);  slice_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_55: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_693, 4);  slice_693 = None
    permute_178: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_55, [0, 1, 2, 4, 3]);  unsqueeze_55 = None
    unsqueeze_56: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_17, 4);  as_strided_17 = None
    permute_179: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_56, [0, 1, 4, 3, 2]);  unsqueeze_56 = None
    permute_180: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_178, [0, 1, 2, 4, 3]);  permute_178 = None
    view_213: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_180, [48, 256, 768]);  permute_180 = None
    permute_181: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_179, [0, 1, 4, 3, 2]);  permute_179 = None
    clone_20: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    view_214: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_20, [48, 768, 64]);  clone_20 = None
    bmm_5: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_213, view_214);  view_213 = view_214 = None
    view_215: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_5, [12, 4, 256, 1, 64]);  bmm_5 = None
    permute_182: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_215, [0, 1, 2, 4, 3]);  view_215 = None
    view_216: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_182, [12, 4, 256, 64]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_217: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_216, [1, 12, 1024, 64]);  view_216 = None
    permute_183: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_184: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_183, [1, 0, 2, 3]);  permute_183 = None
    clone_21: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    view_218: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_21, [1024, 1, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_185: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_218, [1, 0, 2]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_219: "f32[1024, 768]" = torch.ops.aten.view.default(permute_185, [1024, 768]);  permute_185 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    addmm_15: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg39_1, view_219, permute_186);  arg39_1 = view_219 = permute_186 = None
    view_220: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_15, [1, 1024, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_22: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_220);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_26: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_22, add_21);  clone_22 = add_21 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    add_27: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_22: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_26, getitem_9);  add_26 = getitem_9 = None
    mul_17: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_4);  sub_22 = rsqrt_4 = None
    mul_18: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg40_1);  mul_17 = arg40_1 = None
    add_28: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_18, arg41_1);  mul_18 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_221: "f32[1024, 768]" = torch.ops.aten.view.default(add_28, [1024, 768])
    permute_187: "f32[768, 3072]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_16: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg43_1, view_221, permute_187);  arg43_1 = view_221 = permute_187 = None
    view_222: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 1024, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_222, 0.5)
    mul_20: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_222, 0.7071067811865476);  view_222 = None
    erf_2: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_29: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_29);  mul_19 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_223: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_21, [1024, 3072]);  mul_21 = None
    permute_188: "f32[3072, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    addmm_17: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg45_1, view_223, permute_188);  arg45_1 = view_223 = permute_188 = None
    view_224: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_17, [1, 1024, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_23: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_224);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_30: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_23, add_28);  clone_23 = add_28 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    add_31: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_23: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_11);  add_30 = getitem_11 = None
    mul_22: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_5);  sub_23 = rsqrt_5 = None
    mul_23: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg46_1);  mul_22 = arg46_1 = None
    add_32: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_23, arg47_1);  mul_23 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_189: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_32, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_225: "f32[1024, 768]" = torch.ops.aten.view.default(permute_189, [1024, 768])
    permute_190: "f32[768, 768]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_18: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg49_1, view_225, permute_190);  arg49_1 = view_225 = permute_190 = None
    view_226: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_18, [1024, 1, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_227: "f32[1024, 768]" = torch.ops.aten.view.default(permute_189, [1024, 768])
    permute_191: "f32[768, 768]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    addmm_19: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg51_1, view_227, permute_191);  arg51_1 = view_227 = permute_191 = None
    view_228: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_19, [1024, 1, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_229: "f32[1024, 768]" = torch.ops.aten.view.default(permute_189, [1024, 768]);  permute_189 = None
    permute_192: "f32[768, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_20: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg53_1, view_229, permute_192);  arg53_1 = view_229 = permute_192 = None
    view_230: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_20, [1024, 1, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_30: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_226, 8.0);  view_226 = None
    view_231: "f32[1024, 768]" = torch.ops.aten.view.default(div_30, [1024, 768]);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_234: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_228, [1024, 1, 12, 64]);  view_228 = None
    permute_194: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_234, [1, 0, 2, 3]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_196: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_194, [0, 2, 1, 3]);  permute_194 = None
    view_236: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_196, [12, 1024, 64]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_238: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_236, [12, 2, 512, 64]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_19: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_238, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_58: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_19, 4);  as_strided_19 = None
    permute_198: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_58, [0, 1, 4, 2, 3]);  unsqueeze_58 = None
    view_239: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_231, [1024, 1, 768]);  view_231 = None
    view_240: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_239, [1024, 1, 12, 64]);  view_239 = None
    permute_200: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_240, [1, 0, 2, 3]);  view_240 = None
    permute_201: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_200, [0, 2, 1, 3]);  permute_200 = None
    view_241: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_201, [12, 1024, 64]);  permute_201 = None
    view_242: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_241, [12, 2, 512, 64]);  view_241 = None
    as_strided_20: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_242, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_242 = None
    unsqueeze_59: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_20, 4);  as_strided_20 = None
    permute_202: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_59, [0, 1, 2, 4, 3]);  unsqueeze_59 = None
    permute_203: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_202, [0, 1, 2, 4, 3]);  permute_202 = None
    clone_24: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    view_243: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_24, [36, 512, 64]);  clone_24 = None
    permute_204: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_198, [0, 1, 4, 3, 2]);  permute_198 = None
    clone_25: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
    view_244: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_25, [36, 64, 512]);  clone_25 = None
    bmm_6: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_243, view_244);  view_243 = view_244 = None
    view_245: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_6, [12, 3, 512, 1, 512]);  bmm_6 = None
    permute_205: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_245, [0, 1, 2, 4, 3]);  view_245 = None
    view_246: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_205, [12, 3, 512, 512]);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_12: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_246, [0, 0, 0, 1], 0.0);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_247: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_12, [12, 3, 512, 513]);  constant_pad_nd_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_27: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_694: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_247, 0, 0, 9223372036854775807)
    slice_695: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_694, 1, 0, 9223372036854775807);  slice_694 = None
    slice_696: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_695, 2, 0, 256);  slice_695 = None
    slice_697: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_696, 3, 0, 257);  slice_696 = None
    slice_698: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_27, 0, 0, 9223372036854775807)
    slice_699: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_698, 1, 0, -1);  slice_698 = None
    slice_700: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_699, 2, 0, 9223372036854775807);  slice_699 = None
    slice_701: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_700, 3, 256, 9223372036854775807);  slice_700 = None
    copy_36: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_701, slice_697);  slice_701 = slice_697 = None
    slice_702: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_27, 0, 0, 9223372036854775807)
    slice_703: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_702, 1, 0, -1)
    slice_704: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_703, 2, 0, 9223372036854775807)
    slice_scatter_132: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_704, copy_36, 3, 256, 9223372036854775807);  slice_704 = copy_36 = None
    slice_scatter_133: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_703, slice_scatter_132, 2, 0, 9223372036854775807);  slice_703 = slice_scatter_132 = None
    slice_scatter_134: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_702, slice_scatter_133, 1, 0, -1);  slice_702 = slice_scatter_133 = None
    slice_scatter_135: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_27, slice_scatter_134, 0, 0, 9223372036854775807);  full_27 = slice_scatter_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_709: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_247, 0, 0, 9223372036854775807)
    select_60: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_709, 1, -1);  slice_709 = None
    slice_710: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_60, 1, 256, 9223372036854775807);  select_60 = None
    slice_711: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_710, 2, 0, 257);  slice_710 = None
    slice_715: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_135, 0, 0, 9223372036854775807)
    select_62: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_715, 1, -1);  slice_715 = None
    slice_716: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_62, 1, 0, 9223372036854775807);  select_62 = None
    slice_717: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_716, 2, 256, 9223372036854775807);  slice_716 = None
    copy_37: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_717, slice_711);  slice_717 = slice_711 = None
    slice_718: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_135, 0, 0, 9223372036854775807)
    select_63: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_718, 1, -1)
    slice_719: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_63, 1, 0, 9223372036854775807)
    slice_scatter_136: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_719, copy_37, 2, 256, 9223372036854775807);  slice_719 = copy_37 = None
    slice_scatter_137: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_63, slice_scatter_136, 1, 0, 9223372036854775807);  select_63 = slice_scatter_136 = None
    select_scatter_12: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_718, slice_scatter_137, 1, -1);  slice_718 = slice_scatter_137 = None
    slice_scatter_138: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_135, select_scatter_12, 0, 0, 9223372036854775807);  slice_scatter_135 = select_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_723: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_247, 0, 0, 9223372036854775807)
    slice_724: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_723, 1, 0, 9223372036854775807);  slice_723 = None
    slice_725: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_724, 2, -257, -1);  slice_724 = None
    slice_726: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_725, 3, 257, 9223372036854775807);  slice_725 = None
    slice_731: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_138, 0, 0, 9223372036854775807)
    slice_732: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_731, 1, 1, 9223372036854775807);  slice_731 = None
    slice_733: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_732, 2, 0, 9223372036854775807);  slice_732 = None
    slice_734: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_733, 3, 0, 256);  slice_733 = None
    copy_38: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_734, slice_726);  slice_734 = slice_726 = None
    slice_735: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_138, 0, 0, 9223372036854775807)
    slice_736: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_735, 1, 1, 9223372036854775807)
    slice_737: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_736, 2, 0, 9223372036854775807)
    slice_scatter_139: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_737, copy_38, 3, 0, 256);  slice_737 = copy_38 = None
    slice_scatter_140: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_736, slice_scatter_139, 2, 0, 9223372036854775807);  slice_736 = slice_scatter_139 = None
    slice_scatter_141: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_735, slice_scatter_140, 1, 1, 9223372036854775807);  slice_735 = slice_scatter_140 = None
    slice_scatter_142: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_138, slice_scatter_141, 0, 0, 9223372036854775807);  slice_scatter_138 = slice_scatter_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_742: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_247, 0, 0, 9223372036854775807);  view_247 = None
    select_65: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_742, 1, 0);  slice_742 = None
    slice_743: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_65, 1, 0, 255);  select_65 = None
    slice_744: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_743, 2, -255, 9223372036854775807);  slice_743 = None
    slice_748: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_142, 0, 0, 9223372036854775807)
    select_67: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_748, 1, 0);  slice_748 = None
    slice_749: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_67, 1, 1, 256);  select_67 = None
    slice_750: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_749, 2, 1, 256);  slice_749 = None
    copy_39: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_750, slice_744);  slice_750 = slice_744 = None
    slice_751: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_142, 0, 0, 9223372036854775807)
    select_68: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_751, 1, 0)
    slice_752: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_68, 1, 1, 256)
    slice_scatter_143: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_752, copy_39, 2, 1, 256);  slice_752 = copy_39 = None
    slice_scatter_144: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_68, slice_scatter_143, 1, 1, 256);  select_68 = slice_scatter_143 = None
    select_scatter_13: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_751, slice_scatter_144, 1, 0);  slice_751 = slice_scatter_144 = None
    slice_scatter_145: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_142, select_scatter_13, 0, 0, 9223372036854775807);  slice_scatter_142 = select_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_28: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_12: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_60: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_12, -2);  iota_12 = None
    iota_13: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_61: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
    sub_25: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_60, unsqueeze_61);  unsqueeze_60 = unsqueeze_61 = None
    le_6: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_25, 0);  sub_25 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_24: "f32[256, 257]" = torch.ops.aten.where.self(le_6, full_28, scalar_tensor_12);  le_6 = full_28 = scalar_tensor_12 = None
    rev_12: "f32[256, 257]" = torch.ops.prims.rev.default(where_24, [0]);  where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_62: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_12, 0);  rev_12 = None
    slice_756: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_62, 1, 0, 9223372036854775807);  unsqueeze_62 = None
    unsqueeze_63: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_756, 2);  slice_756 = None
    slice_757: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_63, 3, 0, 9223372036854775807);  unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_13: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_757, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_12: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_757, [1, 256, 12, 257]);  slice_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_250: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_145, [1, 12, 1024, 513])
    permute_208: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    slice_762: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_208, 0, 0, 9223372036854775807);  permute_208 = None
    slice_763: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_762, 1, 0, 256);  slice_762 = None
    slice_764: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_763, 2, 0, 9223372036854775807);  slice_763 = None
    slice_765: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_764, 3, 0, 257);  slice_764 = None
    full_29: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_15: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_12, torch.bool);  expand_12 = None
    where_25: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_15, full_29, slice_765);  convert_element_type_15 = full_29 = slice_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_251: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_145, [1, 12, 1024, 513])
    permute_209: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    slice_770: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_209, 0, 0, 9223372036854775807);  permute_209 = None
    slice_771: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_770, 1, 0, 256);  slice_770 = None
    slice_772: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_771, 2, 0, 9223372036854775807);  slice_771 = None
    slice_773: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_772, 3, 0, 257);  slice_772 = None
    copy_40: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_773, where_25);  slice_773 = where_25 = None
    view_252: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_145, [1, 12, 1024, 513]);  slice_scatter_145 = None
    permute_210: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    slice_774: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_210, 0, 0, 9223372036854775807)
    slice_775: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_774, 1, 0, 256)
    slice_776: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_775, 2, 0, 9223372036854775807)
    slice_scatter_146: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_776, copy_40, 3, 0, 257);  slice_776 = copy_40 = None
    slice_scatter_147: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_775, slice_scatter_146, 2, 0, 9223372036854775807);  slice_775 = slice_scatter_146 = None
    slice_scatter_148: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_774, slice_scatter_147, 1, 0, 256);  slice_774 = slice_scatter_147 = None
    slice_scatter_149: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_210, slice_scatter_148, 0, 0, 9223372036854775807);  permute_210 = slice_scatter_148 = None
    permute_211: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_149, [0, 2, 1, 3]);  slice_scatter_149 = None
    view_253: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_211, [12, 4, 256, 513]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_13: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_13, [1, 256, 12, 257]);  rev_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_255: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_253, [1, 12, 1024, 513])
    permute_213: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    slice_785: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_213, 0, 0, 9223372036854775807);  permute_213 = None
    slice_786: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_785, 1, -256, 9223372036854775807);  slice_785 = None
    slice_787: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_786, 2, 0, 9223372036854775807);  slice_786 = None
    slice_788: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_787, 3, -257, 9223372036854775807);  slice_787 = None
    full_30: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_16: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_13, torch.bool);  expand_13 = None
    where_26: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_16, full_30, slice_788);  convert_element_type_16 = full_30 = slice_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_256: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_253, [1, 12, 1024, 513])
    permute_214: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    slice_793: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_214, 0, 0, 9223372036854775807);  permute_214 = None
    slice_794: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_793, 1, -256, 9223372036854775807);  slice_793 = None
    slice_795: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_794, 2, 0, 9223372036854775807);  slice_794 = None
    slice_796: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_795, 3, -257, 9223372036854775807);  slice_795 = None
    copy_41: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_796, where_26);  slice_796 = where_26 = None
    view_257: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_253, [1, 12, 1024, 513]);  view_253 = None
    permute_215: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
    slice_797: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_215, 0, 0, 9223372036854775807)
    slice_798: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_797, 1, -256, 9223372036854775807)
    slice_799: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_798, 2, 0, 9223372036854775807)
    slice_scatter_150: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_799, copy_41, 3, -257, 9223372036854775807);  slice_799 = copy_41 = None
    slice_scatter_151: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_798, slice_scatter_150, 2, 0, 9223372036854775807);  slice_798 = slice_scatter_150 = None
    slice_scatter_152: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_797, slice_scatter_151, 1, -256, 9223372036854775807);  slice_797 = slice_scatter_151 = None
    slice_scatter_153: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_215, slice_scatter_152, 0, 0, 9223372036854775807);  permute_215 = slice_scatter_152 = None
    permute_216: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_153, [0, 2, 1, 3]);  slice_scatter_153 = None
    view_258: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_216, [12, 4, 256, 513]);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_3: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_804: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_3, 0, 0, 9223372036854775807);  ne_3 = None
    slice_805: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_804, 1, 0, 9223372036854775807);  slice_804 = None
    unsqueeze_64: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_805, 2);  slice_805 = None
    unsqueeze_65: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, 3);  unsqueeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_17: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_65, torch.float32)
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_65, scalar_tensor_13, convert_element_type_17);  unsqueeze_65 = scalar_tensor_13 = convert_element_type_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_31: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_218: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_31, [0, 2, 1, 3]);  full_31 = None
    view_260: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_218, [1, 1024, 1]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_219: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_27, [0, 2, 1, 3]);  where_27 = None
    view_261: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_219, [1, 1024, 1]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_262: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_260, [1, 2, 512, 1]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_21: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_262, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_263: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_261, [1, 2, 512, 1]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_22: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_263, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_66: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_21, 4);  as_strided_21 = None
    permute_220: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_66, [0, 1, 2, 4, 3]);  unsqueeze_66 = None
    unsqueeze_67: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_22, 4);  as_strided_22 = None
    permute_221: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_67, [0, 1, 4, 2, 3]);  unsqueeze_67 = None
    mul_24: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_220, permute_221);  permute_220 = permute_221 = None
    view_264: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_24, [1, 3, 512, 512]);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_13: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_264, [0, 0, 0, 1], 0.0);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_265: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_13, [1, 3, 512, 513]);  constant_pad_nd_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_32: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_806: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_265, 0, 0, 9223372036854775807)
    slice_807: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_806, 1, 0, 9223372036854775807);  slice_806 = None
    slice_808: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_807, 2, 0, 256);  slice_807 = None
    slice_809: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_808, 3, 0, 257);  slice_808 = None
    slice_810: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_32, 0, 0, 9223372036854775807)
    slice_811: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_810, 1, 0, -1);  slice_810 = None
    slice_812: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_811, 2, 0, 9223372036854775807);  slice_811 = None
    slice_813: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_812, 3, 256, 9223372036854775807);  slice_812 = None
    copy_42: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_813, slice_809);  slice_813 = slice_809 = None
    slice_814: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_32, 0, 0, 9223372036854775807)
    slice_815: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_814, 1, 0, -1)
    slice_816: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_815, 2, 0, 9223372036854775807)
    slice_scatter_154: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_816, copy_42, 3, 256, 9223372036854775807);  slice_816 = copy_42 = None
    slice_scatter_155: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_815, slice_scatter_154, 2, 0, 9223372036854775807);  slice_815 = slice_scatter_154 = None
    slice_scatter_156: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_814, slice_scatter_155, 1, 0, -1);  slice_814 = slice_scatter_155 = None
    slice_scatter_157: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_32, slice_scatter_156, 0, 0, 9223372036854775807);  full_32 = slice_scatter_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_821: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_265, 0, 0, 9223372036854775807)
    select_70: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_821, 1, -1);  slice_821 = None
    slice_822: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_70, 1, 256, 9223372036854775807);  select_70 = None
    slice_823: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_822, 2, 0, 257);  slice_822 = None
    slice_827: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_157, 0, 0, 9223372036854775807)
    select_72: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_827, 1, -1);  slice_827 = None
    slice_828: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_72, 1, 0, 9223372036854775807);  select_72 = None
    slice_829: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_828, 2, 256, 9223372036854775807);  slice_828 = None
    copy_43: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_829, slice_823);  slice_829 = slice_823 = None
    slice_830: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_157, 0, 0, 9223372036854775807)
    select_73: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_830, 1, -1)
    slice_831: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_73, 1, 0, 9223372036854775807)
    slice_scatter_158: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_831, copy_43, 2, 256, 9223372036854775807);  slice_831 = copy_43 = None
    slice_scatter_159: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_73, slice_scatter_158, 1, 0, 9223372036854775807);  select_73 = slice_scatter_158 = None
    select_scatter_14: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_830, slice_scatter_159, 1, -1);  slice_830 = slice_scatter_159 = None
    slice_scatter_160: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_157, select_scatter_14, 0, 0, 9223372036854775807);  slice_scatter_157 = select_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_835: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_265, 0, 0, 9223372036854775807)
    slice_836: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_835, 1, 0, 9223372036854775807);  slice_835 = None
    slice_837: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_836, 2, -257, -1);  slice_836 = None
    slice_838: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_837, 3, 257, 9223372036854775807);  slice_837 = None
    slice_843: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_160, 0, 0, 9223372036854775807)
    slice_844: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_843, 1, 1, 9223372036854775807);  slice_843 = None
    slice_845: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_844, 2, 0, 9223372036854775807);  slice_844 = None
    slice_846: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_845, 3, 0, 256);  slice_845 = None
    copy_44: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_846, slice_838);  slice_846 = slice_838 = None
    slice_847: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_160, 0, 0, 9223372036854775807)
    slice_848: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_847, 1, 1, 9223372036854775807)
    slice_849: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_848, 2, 0, 9223372036854775807)
    slice_scatter_161: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_849, copy_44, 3, 0, 256);  slice_849 = copy_44 = None
    slice_scatter_162: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_848, slice_scatter_161, 2, 0, 9223372036854775807);  slice_848 = slice_scatter_161 = None
    slice_scatter_163: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_847, slice_scatter_162, 1, 1, 9223372036854775807);  slice_847 = slice_scatter_162 = None
    slice_scatter_164: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_160, slice_scatter_163, 0, 0, 9223372036854775807);  slice_scatter_160 = slice_scatter_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_854: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_265, 0, 0, 9223372036854775807);  view_265 = None
    select_75: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_854, 1, 0);  slice_854 = None
    slice_855: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_75, 1, 0, 255);  select_75 = None
    slice_856: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_855, 2, -255, 9223372036854775807);  slice_855 = None
    slice_860: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_164, 0, 0, 9223372036854775807)
    select_77: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_860, 1, 0);  slice_860 = None
    slice_861: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_77, 1, 1, 256);  select_77 = None
    slice_862: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_861, 2, 1, 256);  slice_861 = None
    copy_45: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_862, slice_856);  slice_862 = slice_856 = None
    slice_863: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_164, 0, 0, 9223372036854775807)
    select_78: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_863, 1, 0)
    slice_864: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_78, 1, 1, 256)
    slice_scatter_165: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_864, copy_45, 2, 1, 256);  slice_864 = copy_45 = None
    slice_scatter_166: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_78, slice_scatter_165, 1, 1, 256);  select_78 = slice_scatter_165 = None
    select_scatter_15: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_863, slice_scatter_166, 1, 0);  slice_863 = slice_scatter_166 = None
    slice_scatter_167: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_164, select_scatter_15, 0, 0, 9223372036854775807);  slice_scatter_164 = select_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_33: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_14: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_68: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_14, -2);  iota_14 = None
    iota_15: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_69: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
    sub_27: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_68, unsqueeze_69);  unsqueeze_68 = unsqueeze_69 = None
    le_7: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_27, 0);  sub_27 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_28: "f32[256, 257]" = torch.ops.aten.where.self(le_7, full_33, scalar_tensor_14);  le_7 = full_33 = scalar_tensor_14 = None
    rev_14: "f32[256, 257]" = torch.ops.prims.rev.default(where_28, [0]);  where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_70: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_14, 0);  rev_14 = None
    slice_868: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_70, 1, 0, 9223372036854775807);  unsqueeze_70 = None
    unsqueeze_71: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_868, 2);  slice_868 = None
    slice_869: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_71, 3, 0, 9223372036854775807);  unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_15: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_869, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_14: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_869, [1, 256, 1, 257]);  slice_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_268: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_167, [1, 1, 1024, 513])
    permute_224: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
    slice_874: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_224, 0, 0, 9223372036854775807);  permute_224 = None
    slice_875: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_874, 1, 0, 256);  slice_874 = None
    slice_876: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_875, 2, 0, 9223372036854775807);  slice_875 = None
    slice_877: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_876, 3, 0, 257);  slice_876 = None
    full_34: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_18: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_14, torch.bool);  expand_14 = None
    where_29: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_18, full_34, slice_877);  convert_element_type_18 = full_34 = slice_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_269: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_167, [1, 1, 1024, 513])
    permute_225: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    slice_882: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_225, 0, 0, 9223372036854775807);  permute_225 = None
    slice_883: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_882, 1, 0, 256);  slice_882 = None
    slice_884: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_883, 2, 0, 9223372036854775807);  slice_883 = None
    slice_885: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_884, 3, 0, 257);  slice_884 = None
    copy_46: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_885, where_29);  slice_885 = where_29 = None
    view_270: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_167, [1, 1, 1024, 513]);  slice_scatter_167 = None
    permute_226: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
    slice_886: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_226, 0, 0, 9223372036854775807)
    slice_887: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_886, 1, 0, 256)
    slice_888: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_887, 2, 0, 9223372036854775807)
    slice_scatter_168: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_888, copy_46, 3, 0, 257);  slice_888 = copy_46 = None
    slice_scatter_169: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_887, slice_scatter_168, 2, 0, 9223372036854775807);  slice_887 = slice_scatter_168 = None
    slice_scatter_170: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_886, slice_scatter_169, 1, 0, 256);  slice_886 = slice_scatter_169 = None
    slice_scatter_171: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_226, slice_scatter_170, 0, 0, 9223372036854775807);  permute_226 = slice_scatter_170 = None
    permute_227: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_171, [0, 2, 1, 3]);  slice_scatter_171 = None
    view_271: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_227, [1, 4, 256, 513]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_15: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_15, [1, 256, 1, 257]);  rev_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_273: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_271, [1, 1, 1024, 513])
    permute_229: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    slice_897: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_229, 0, 0, 9223372036854775807);  permute_229 = None
    slice_898: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_897, 1, -256, 9223372036854775807);  slice_897 = None
    slice_899: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_898, 2, 0, 9223372036854775807);  slice_898 = None
    slice_900: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_899, 3, -257, 9223372036854775807);  slice_899 = None
    full_35: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_19: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_15, torch.bool);  expand_15 = None
    where_30: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_19, full_35, slice_900);  convert_element_type_19 = full_35 = slice_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_274: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_271, [1, 1, 1024, 513])
    permute_230: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
    slice_905: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_230, 0, 0, 9223372036854775807);  permute_230 = None
    slice_906: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_905, 1, -256, 9223372036854775807);  slice_905 = None
    slice_907: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_906, 2, 0, 9223372036854775807);  slice_906 = None
    slice_908: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_907, 3, -257, 9223372036854775807);  slice_907 = None
    copy_47: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_908, where_30);  slice_908 = where_30 = None
    view_275: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_271, [1, 1, 1024, 513]);  view_271 = None
    permute_231: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None
    slice_909: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_231, 0, 0, 9223372036854775807)
    slice_910: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_909, 1, -256, 9223372036854775807)
    slice_911: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_910, 2, 0, 9223372036854775807)
    slice_scatter_172: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_911, copy_47, 3, -257, 9223372036854775807);  slice_911 = copy_47 = None
    slice_scatter_173: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_910, slice_scatter_172, 2, 0, 9223372036854775807);  slice_910 = slice_scatter_172 = None
    slice_scatter_174: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_909, slice_scatter_173, 1, -256, 9223372036854775807);  slice_909 = slice_scatter_173 = None
    slice_scatter_175: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_231, slice_scatter_174, 0, 0, 9223372036854775807);  permute_231 = slice_scatter_174 = None
    permute_232: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_175, [0, 2, 1, 3]);  slice_scatter_175 = None
    view_276: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_232, [1, 4, 256, 513]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_278: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_258, [1, 12, 1024, 513]);  view_258 = None
    permute_234: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    view_279: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_276, [1, 1, 1024, 513]);  view_276 = None
    permute_235: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
    add_35: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_234, permute_235);  permute_234 = permute_235 = None
    permute_236: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_35, [0, 2, 1, 3]);  add_35 = None
    view_281: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_236, [12, 4, 256, 513]);  permute_236 = None
    view_282: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_281, [1, 12, 1024, 513]);  view_281 = None
    permute_237: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_26: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
    amax_3: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_26, [-1], True)
    sub_28: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_26, amax_3);  clone_26 = amax_3 = None
    exp_3: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_4: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_37: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_916: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_917: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_916, 1, 0, 9223372036854775807);  slice_916 = None
    unsqueeze_72: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_917, 2);  slice_917 = None
    unsqueeze_73: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_73, scalar_tensor_15, div_37);  unsqueeze_73 = scalar_tensor_15 = div_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_27: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_31);  where_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_283: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_230, [1024, 1, 12, 64]);  view_230 = None
    permute_238: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_283, [1, 0, 2, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_239: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_27, [0, 2, 1, 3]);  clone_27 = None
    view_284: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_239, [12, 4, 256, 513]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_240: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_238, [0, 2, 1, 3]);  permute_238 = None
    view_285: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_240, [12, 1024, 64]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_14: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_285, [0, 0, 256, 256], -1.0);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_23: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_14, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_15: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_284, [0, 257], 0.0);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_286: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_15, [12, 4, -1]);  constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_918: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_286, 0, 0, 9223372036854775807);  view_286 = None
    slice_919: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_918, 1, 0, 9223372036854775807);  slice_918 = None
    slice_920: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_919, 2, 0, -256);  slice_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_287: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_920, [12, 4, 256, 769]);  slice_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_921: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_287, 0, 0, 9223372036854775807);  view_287 = None
    slice_922: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_921, 1, 0, 9223372036854775807);  slice_921 = None
    slice_923: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_922, 2, 0, 9223372036854775807);  slice_922 = None
    slice_924: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_923, 3, 0, -1);  slice_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_74: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_924, 4);  slice_924 = None
    permute_241: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_74, [0, 1, 2, 4, 3]);  unsqueeze_74 = None
    unsqueeze_75: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_23, 4);  as_strided_23 = None
    permute_242: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 3, 2]);  unsqueeze_75 = None
    permute_243: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_241, [0, 1, 2, 4, 3]);  permute_241 = None
    view_288: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_243, [48, 256, 768]);  permute_243 = None
    permute_244: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_242, [0, 1, 4, 3, 2]);  permute_242 = None
    clone_28: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    view_289: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_28, [48, 768, 64]);  clone_28 = None
    bmm_7: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_288, view_289);  view_288 = view_289 = None
    view_290: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_7, [12, 4, 256, 1, 64]);  bmm_7 = None
    permute_245: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_290, [0, 1, 2, 4, 3]);  view_290 = None
    view_291: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_245, [12, 4, 256, 64]);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_292: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_291, [1, 12, 1024, 64]);  view_291 = None
    permute_246: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_247: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_246, [1, 0, 2, 3]);  permute_246 = None
    clone_29: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    view_293: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_29, [1024, 1, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_248: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_293, [1, 0, 2]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_294: "f32[1024, 768]" = torch.ops.aten.view.default(permute_248, [1024, 768]);  permute_248 = None
    permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_21: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg55_1, view_294, permute_249);  arg55_1 = view_294 = permute_249 = None
    view_295: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_21, [1, 1024, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_30: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_295);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_30, add_32);  clone_30 = add_32 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    add_38: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_30: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_13);  add_37 = getitem_13 = None
    mul_25: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_6);  sub_30 = rsqrt_6 = None
    mul_26: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg56_1);  mul_25 = arg56_1 = None
    add_39: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_26, arg57_1);  mul_26 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_296: "f32[1024, 768]" = torch.ops.aten.view.default(add_39, [1024, 768])
    permute_250: "f32[768, 3072]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_22: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg59_1, view_296, permute_250);  arg59_1 = view_296 = permute_250 = None
    view_297: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 1024, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_297, 0.5)
    mul_28: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_297, 0.7071067811865476);  view_297 = None
    erf_3: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_40: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_29: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_40);  mul_27 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_298: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_29, [1024, 3072]);  mul_29 = None
    permute_251: "f32[3072, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    addmm_23: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg61_1, view_298, permute_251);  arg61_1 = view_298 = permute_251 = None
    view_299: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_23, [1, 1024, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_31: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_299);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_31, add_39);  clone_31 = add_39 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    add_42: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_31: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_15);  add_41 = getitem_15 = None
    mul_30: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_7);  sub_31 = rsqrt_7 = None
    mul_31: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg62_1);  mul_30 = arg62_1 = None
    add_43: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_31, arg63_1);  mul_31 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_252: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_43, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_300: "f32[1024, 768]" = torch.ops.aten.view.default(permute_252, [1024, 768])
    permute_253: "f32[768, 768]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_24: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg65_1, view_300, permute_253);  arg65_1 = view_300 = permute_253 = None
    view_301: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_24, [1024, 1, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_302: "f32[1024, 768]" = torch.ops.aten.view.default(permute_252, [1024, 768])
    permute_254: "f32[768, 768]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    addmm_25: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg67_1, view_302, permute_254);  arg67_1 = view_302 = permute_254 = None
    view_303: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_25, [1024, 1, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_304: "f32[1024, 768]" = torch.ops.aten.view.default(permute_252, [1024, 768]);  permute_252 = None
    permute_255: "f32[768, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm_26: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg69_1, view_304, permute_255);  arg69_1 = view_304 = permute_255 = None
    view_305: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_26, [1024, 1, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_40: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_301, 8.0);  view_301 = None
    view_306: "f32[1024, 768]" = torch.ops.aten.view.default(div_40, [1024, 768]);  div_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_309: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_303, [1024, 1, 12, 64]);  view_303 = None
    permute_257: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_309, [1, 0, 2, 3]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_259: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_257, [0, 2, 1, 3]);  permute_257 = None
    view_311: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_259, [12, 1024, 64]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_313: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_311, [12, 2, 512, 64]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_25: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_313, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_77: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_25, 4);  as_strided_25 = None
    permute_261: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_77, [0, 1, 4, 2, 3]);  unsqueeze_77 = None
    view_314: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_306, [1024, 1, 768]);  view_306 = None
    view_315: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_314, [1024, 1, 12, 64]);  view_314 = None
    permute_263: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_315, [1, 0, 2, 3]);  view_315 = None
    permute_264: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_263, [0, 2, 1, 3]);  permute_263 = None
    view_316: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_264, [12, 1024, 64]);  permute_264 = None
    view_317: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_316, [12, 2, 512, 64]);  view_316 = None
    as_strided_26: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_317, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_317 = None
    unsqueeze_78: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_26, 4);  as_strided_26 = None
    permute_265: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_78, [0, 1, 2, 4, 3]);  unsqueeze_78 = None
    permute_266: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_265, [0, 1, 2, 4, 3]);  permute_265 = None
    clone_32: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    view_318: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_32, [36, 512, 64]);  clone_32 = None
    permute_267: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_261, [0, 1, 4, 3, 2]);  permute_261 = None
    clone_33: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    view_319: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_33, [36, 64, 512]);  clone_33 = None
    bmm_8: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_318, view_319);  view_318 = view_319 = None
    view_320: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_8, [12, 3, 512, 1, 512]);  bmm_8 = None
    permute_268: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_320, [0, 1, 2, 4, 3]);  view_320 = None
    view_321: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_268, [12, 3, 512, 512]);  permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_16: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_321, [0, 0, 0, 1], 0.0);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_322: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_16, [12, 3, 512, 513]);  constant_pad_nd_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_36: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_925: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_322, 0, 0, 9223372036854775807)
    slice_926: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_925, 1, 0, 9223372036854775807);  slice_925 = None
    slice_927: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_926, 2, 0, 256);  slice_926 = None
    slice_928: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_927, 3, 0, 257);  slice_927 = None
    slice_929: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_36, 0, 0, 9223372036854775807)
    slice_930: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_929, 1, 0, -1);  slice_929 = None
    slice_931: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_930, 2, 0, 9223372036854775807);  slice_930 = None
    slice_932: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_931, 3, 256, 9223372036854775807);  slice_931 = None
    copy_48: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_932, slice_928);  slice_932 = slice_928 = None
    slice_933: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_36, 0, 0, 9223372036854775807)
    slice_934: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_933, 1, 0, -1)
    slice_935: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_934, 2, 0, 9223372036854775807)
    slice_scatter_176: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_935, copy_48, 3, 256, 9223372036854775807);  slice_935 = copy_48 = None
    slice_scatter_177: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_934, slice_scatter_176, 2, 0, 9223372036854775807);  slice_934 = slice_scatter_176 = None
    slice_scatter_178: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_933, slice_scatter_177, 1, 0, -1);  slice_933 = slice_scatter_177 = None
    slice_scatter_179: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_36, slice_scatter_178, 0, 0, 9223372036854775807);  full_36 = slice_scatter_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_940: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_322, 0, 0, 9223372036854775807)
    select_80: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_940, 1, -1);  slice_940 = None
    slice_941: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_80, 1, 256, 9223372036854775807);  select_80 = None
    slice_942: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_941, 2, 0, 257);  slice_941 = None
    slice_946: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_179, 0, 0, 9223372036854775807)
    select_82: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_946, 1, -1);  slice_946 = None
    slice_947: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_82, 1, 0, 9223372036854775807);  select_82 = None
    slice_948: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_947, 2, 256, 9223372036854775807);  slice_947 = None
    copy_49: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_948, slice_942);  slice_948 = slice_942 = None
    slice_949: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_179, 0, 0, 9223372036854775807)
    select_83: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_949, 1, -1)
    slice_950: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_83, 1, 0, 9223372036854775807)
    slice_scatter_180: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_950, copy_49, 2, 256, 9223372036854775807);  slice_950 = copy_49 = None
    slice_scatter_181: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_83, slice_scatter_180, 1, 0, 9223372036854775807);  select_83 = slice_scatter_180 = None
    select_scatter_16: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_949, slice_scatter_181, 1, -1);  slice_949 = slice_scatter_181 = None
    slice_scatter_182: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_179, select_scatter_16, 0, 0, 9223372036854775807);  slice_scatter_179 = select_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_954: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_322, 0, 0, 9223372036854775807)
    slice_955: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_954, 1, 0, 9223372036854775807);  slice_954 = None
    slice_956: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_955, 2, -257, -1);  slice_955 = None
    slice_957: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_956, 3, 257, 9223372036854775807);  slice_956 = None
    slice_962: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_182, 0, 0, 9223372036854775807)
    slice_963: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_962, 1, 1, 9223372036854775807);  slice_962 = None
    slice_964: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_963, 2, 0, 9223372036854775807);  slice_963 = None
    slice_965: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_964, 3, 0, 256);  slice_964 = None
    copy_50: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_965, slice_957);  slice_965 = slice_957 = None
    slice_966: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_182, 0, 0, 9223372036854775807)
    slice_967: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_966, 1, 1, 9223372036854775807)
    slice_968: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_967, 2, 0, 9223372036854775807)
    slice_scatter_183: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_968, copy_50, 3, 0, 256);  slice_968 = copy_50 = None
    slice_scatter_184: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_967, slice_scatter_183, 2, 0, 9223372036854775807);  slice_967 = slice_scatter_183 = None
    slice_scatter_185: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_966, slice_scatter_184, 1, 1, 9223372036854775807);  slice_966 = slice_scatter_184 = None
    slice_scatter_186: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_182, slice_scatter_185, 0, 0, 9223372036854775807);  slice_scatter_182 = slice_scatter_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_973: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_322, 0, 0, 9223372036854775807);  view_322 = None
    select_85: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_973, 1, 0);  slice_973 = None
    slice_974: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_85, 1, 0, 255);  select_85 = None
    slice_975: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_974, 2, -255, 9223372036854775807);  slice_974 = None
    slice_979: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_186, 0, 0, 9223372036854775807)
    select_87: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_979, 1, 0);  slice_979 = None
    slice_980: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_87, 1, 1, 256);  select_87 = None
    slice_981: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_980, 2, 1, 256);  slice_980 = None
    copy_51: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_981, slice_975);  slice_981 = slice_975 = None
    slice_982: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_186, 0, 0, 9223372036854775807)
    select_88: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_982, 1, 0)
    slice_983: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_88, 1, 1, 256)
    slice_scatter_187: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_983, copy_51, 2, 1, 256);  slice_983 = copy_51 = None
    slice_scatter_188: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_88, slice_scatter_187, 1, 1, 256);  select_88 = slice_scatter_187 = None
    select_scatter_17: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_982, slice_scatter_188, 1, 0);  slice_982 = slice_scatter_188 = None
    slice_scatter_189: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_186, select_scatter_17, 0, 0, 9223372036854775807);  slice_scatter_186 = select_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_37: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_16: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_79: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_16, -2);  iota_16 = None
    iota_17: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_80: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
    sub_33: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_79, unsqueeze_80);  unsqueeze_79 = unsqueeze_80 = None
    le_8: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_33, 0);  sub_33 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_32: "f32[256, 257]" = torch.ops.aten.where.self(le_8, full_37, scalar_tensor_16);  le_8 = full_37 = scalar_tensor_16 = None
    rev_16: "f32[256, 257]" = torch.ops.prims.rev.default(where_32, [0]);  where_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_81: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_16, 0);  rev_16 = None
    slice_987: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_81, 1, 0, 9223372036854775807);  unsqueeze_81 = None
    unsqueeze_82: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_987, 2);  slice_987 = None
    slice_988: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_82, 3, 0, 9223372036854775807);  unsqueeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_17: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_988, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_16: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_988, [1, 256, 12, 257]);  slice_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_325: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_189, [1, 12, 1024, 513])
    permute_271: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
    slice_993: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_271, 0, 0, 9223372036854775807);  permute_271 = None
    slice_994: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_993, 1, 0, 256);  slice_993 = None
    slice_995: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_994, 2, 0, 9223372036854775807);  slice_994 = None
    slice_996: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_995, 3, 0, 257);  slice_995 = None
    full_38: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_20: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_16, torch.bool);  expand_16 = None
    where_33: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_20, full_38, slice_996);  convert_element_type_20 = full_38 = slice_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_326: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_189, [1, 12, 1024, 513])
    permute_272: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
    slice_1001: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_272, 0, 0, 9223372036854775807);  permute_272 = None
    slice_1002: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1001, 1, 0, 256);  slice_1001 = None
    slice_1003: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1002, 2, 0, 9223372036854775807);  slice_1002 = None
    slice_1004: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1003, 3, 0, 257);  slice_1003 = None
    copy_52: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1004, where_33);  slice_1004 = where_33 = None
    view_327: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_189, [1, 12, 1024, 513]);  slice_scatter_189 = None
    permute_273: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    slice_1005: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_273, 0, 0, 9223372036854775807)
    slice_1006: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1005, 1, 0, 256)
    slice_1007: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1006, 2, 0, 9223372036854775807)
    slice_scatter_190: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1007, copy_52, 3, 0, 257);  slice_1007 = copy_52 = None
    slice_scatter_191: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1006, slice_scatter_190, 2, 0, 9223372036854775807);  slice_1006 = slice_scatter_190 = None
    slice_scatter_192: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1005, slice_scatter_191, 1, 0, 256);  slice_1005 = slice_scatter_191 = None
    slice_scatter_193: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_273, slice_scatter_192, 0, 0, 9223372036854775807);  permute_273 = slice_scatter_192 = None
    permute_274: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_193, [0, 2, 1, 3]);  slice_scatter_193 = None
    view_328: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_274, [12, 4, 256, 513]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_17: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_17, [1, 256, 12, 257]);  rev_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_330: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_328, [1, 12, 1024, 513])
    permute_276: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    slice_1016: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_276, 0, 0, 9223372036854775807);  permute_276 = None
    slice_1017: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1016, 1, -256, 9223372036854775807);  slice_1016 = None
    slice_1018: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1017, 2, 0, 9223372036854775807);  slice_1017 = None
    slice_1019: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1018, 3, -257, 9223372036854775807);  slice_1018 = None
    full_39: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_21: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_17, torch.bool);  expand_17 = None
    where_34: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_21, full_39, slice_1019);  convert_element_type_21 = full_39 = slice_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_331: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_328, [1, 12, 1024, 513])
    permute_277: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    slice_1024: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_277, 0, 0, 9223372036854775807);  permute_277 = None
    slice_1025: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1024, 1, -256, 9223372036854775807);  slice_1024 = None
    slice_1026: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1025, 2, 0, 9223372036854775807);  slice_1025 = None
    slice_1027: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1026, 3, -257, 9223372036854775807);  slice_1026 = None
    copy_53: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1027, where_34);  slice_1027 = where_34 = None
    view_332: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_328, [1, 12, 1024, 513]);  view_328 = None
    permute_278: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
    slice_1028: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_278, 0, 0, 9223372036854775807)
    slice_1029: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1028, 1, -256, 9223372036854775807)
    slice_1030: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1029, 2, 0, 9223372036854775807)
    slice_scatter_194: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1030, copy_53, 3, -257, 9223372036854775807);  slice_1030 = copy_53 = None
    slice_scatter_195: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1029, slice_scatter_194, 2, 0, 9223372036854775807);  slice_1029 = slice_scatter_194 = None
    slice_scatter_196: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1028, slice_scatter_195, 1, -256, 9223372036854775807);  slice_1028 = slice_scatter_195 = None
    slice_scatter_197: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_278, slice_scatter_196, 0, 0, 9223372036854775807);  permute_278 = slice_scatter_196 = None
    permute_279: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_197, [0, 2, 1, 3]);  slice_scatter_197 = None
    view_333: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_279, [12, 4, 256, 513]);  permute_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_4: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_1035: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_4, 0, 0, 9223372036854775807);  ne_4 = None
    slice_1036: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_1035, 1, 0, 9223372036854775807);  slice_1035 = None
    unsqueeze_83: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_1036, 2);  slice_1036 = None
    unsqueeze_84: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 3);  unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_22: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_84, torch.float32)
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_84, scalar_tensor_17, convert_element_type_22);  unsqueeze_84 = scalar_tensor_17 = convert_element_type_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_40: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_281: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_40, [0, 2, 1, 3]);  full_40 = None
    view_335: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_281, [1, 1024, 1]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_282: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_35, [0, 2, 1, 3]);  where_35 = None
    view_336: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_282, [1, 1024, 1]);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_337: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_335, [1, 2, 512, 1]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_27: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_337, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_338: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_336, [1, 2, 512, 1]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_28: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_338, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_85: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_27, 4);  as_strided_27 = None
    permute_283: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_85, [0, 1, 2, 4, 3]);  unsqueeze_85 = None
    unsqueeze_86: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_28, 4);  as_strided_28 = None
    permute_284: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_86, [0, 1, 4, 2, 3]);  unsqueeze_86 = None
    mul_32: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_283, permute_284);  permute_283 = permute_284 = None
    view_339: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_32, [1, 3, 512, 512]);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_17: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_339, [0, 0, 0, 1], 0.0);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_340: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_17, [1, 3, 512, 513]);  constant_pad_nd_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_41: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1037: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_340, 0, 0, 9223372036854775807)
    slice_1038: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1037, 1, 0, 9223372036854775807);  slice_1037 = None
    slice_1039: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1038, 2, 0, 256);  slice_1038 = None
    slice_1040: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1039, 3, 0, 257);  slice_1039 = None
    slice_1041: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_41, 0, 0, 9223372036854775807)
    slice_1042: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1041, 1, 0, -1);  slice_1041 = None
    slice_1043: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1042, 2, 0, 9223372036854775807);  slice_1042 = None
    slice_1044: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1043, 3, 256, 9223372036854775807);  slice_1043 = None
    copy_54: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1044, slice_1040);  slice_1044 = slice_1040 = None
    slice_1045: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_41, 0, 0, 9223372036854775807)
    slice_1046: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1045, 1, 0, -1)
    slice_1047: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1046, 2, 0, 9223372036854775807)
    slice_scatter_198: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1047, copy_54, 3, 256, 9223372036854775807);  slice_1047 = copy_54 = None
    slice_scatter_199: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1046, slice_scatter_198, 2, 0, 9223372036854775807);  slice_1046 = slice_scatter_198 = None
    slice_scatter_200: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1045, slice_scatter_199, 1, 0, -1);  slice_1045 = slice_scatter_199 = None
    slice_scatter_201: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_41, slice_scatter_200, 0, 0, 9223372036854775807);  full_41 = slice_scatter_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1052: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_340, 0, 0, 9223372036854775807)
    select_90: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_1052, 1, -1);  slice_1052 = None
    slice_1053: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_90, 1, 256, 9223372036854775807);  select_90 = None
    slice_1054: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1053, 2, 0, 257);  slice_1053 = None
    slice_1058: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_201, 0, 0, 9223372036854775807)
    select_92: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1058, 1, -1);  slice_1058 = None
    slice_1059: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_92, 1, 0, 9223372036854775807);  select_92 = None
    slice_1060: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1059, 2, 256, 9223372036854775807);  slice_1059 = None
    copy_55: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_1060, slice_1054);  slice_1060 = slice_1054 = None
    slice_1061: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_201, 0, 0, 9223372036854775807)
    select_93: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1061, 1, -1)
    slice_1062: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_93, 1, 0, 9223372036854775807)
    slice_scatter_202: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1062, copy_55, 2, 256, 9223372036854775807);  slice_1062 = copy_55 = None
    slice_scatter_203: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_93, slice_scatter_202, 1, 0, 9223372036854775807);  select_93 = slice_scatter_202 = None
    select_scatter_18: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1061, slice_scatter_203, 1, -1);  slice_1061 = slice_scatter_203 = None
    slice_scatter_204: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_201, select_scatter_18, 0, 0, 9223372036854775807);  slice_scatter_201 = select_scatter_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1066: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_340, 0, 0, 9223372036854775807)
    slice_1067: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1066, 1, 0, 9223372036854775807);  slice_1066 = None
    slice_1068: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1067, 2, -257, -1);  slice_1067 = None
    slice_1069: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1068, 3, 257, 9223372036854775807);  slice_1068 = None
    slice_1074: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_204, 0, 0, 9223372036854775807)
    slice_1075: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1074, 1, 1, 9223372036854775807);  slice_1074 = None
    slice_1076: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1075, 2, 0, 9223372036854775807);  slice_1075 = None
    slice_1077: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1076, 3, 0, 256);  slice_1076 = None
    copy_56: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1077, slice_1069);  slice_1077 = slice_1069 = None
    slice_1078: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_204, 0, 0, 9223372036854775807)
    slice_1079: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1078, 1, 1, 9223372036854775807)
    slice_1080: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1079, 2, 0, 9223372036854775807)
    slice_scatter_205: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1080, copy_56, 3, 0, 256);  slice_1080 = copy_56 = None
    slice_scatter_206: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1079, slice_scatter_205, 2, 0, 9223372036854775807);  slice_1079 = slice_scatter_205 = None
    slice_scatter_207: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1078, slice_scatter_206, 1, 1, 9223372036854775807);  slice_1078 = slice_scatter_206 = None
    slice_scatter_208: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_204, slice_scatter_207, 0, 0, 9223372036854775807);  slice_scatter_204 = slice_scatter_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_1085: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_340, 0, 0, 9223372036854775807);  view_340 = None
    select_95: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_1085, 1, 0);  slice_1085 = None
    slice_1086: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_95, 1, 0, 255);  select_95 = None
    slice_1087: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1086, 2, -255, 9223372036854775807);  slice_1086 = None
    slice_1091: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_208, 0, 0, 9223372036854775807)
    select_97: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1091, 1, 0);  slice_1091 = None
    slice_1092: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_97, 1, 1, 256);  select_97 = None
    slice_1093: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1092, 2, 1, 256);  slice_1092 = None
    copy_57: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_1093, slice_1087);  slice_1093 = slice_1087 = None
    slice_1094: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_208, 0, 0, 9223372036854775807)
    select_98: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1094, 1, 0)
    slice_1095: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_98, 1, 1, 256)
    slice_scatter_209: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1095, copy_57, 2, 1, 256);  slice_1095 = copy_57 = None
    slice_scatter_210: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_98, slice_scatter_209, 1, 1, 256);  select_98 = slice_scatter_209 = None
    select_scatter_19: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1094, slice_scatter_210, 1, 0);  slice_1094 = slice_scatter_210 = None
    slice_scatter_211: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_208, select_scatter_19, 0, 0, 9223372036854775807);  slice_scatter_208 = select_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_42: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_18: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_87: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_18, -2);  iota_18 = None
    iota_19: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_88: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
    sub_35: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_87, unsqueeze_88);  unsqueeze_87 = unsqueeze_88 = None
    le_9: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_35, 0);  sub_35 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_36: "f32[256, 257]" = torch.ops.aten.where.self(le_9, full_42, scalar_tensor_18);  le_9 = full_42 = scalar_tensor_18 = None
    rev_18: "f32[256, 257]" = torch.ops.prims.rev.default(where_36, [0]);  where_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_89: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_18, 0);  rev_18 = None
    slice_1099: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_89, 1, 0, 9223372036854775807);  unsqueeze_89 = None
    unsqueeze_90: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_1099, 2);  slice_1099 = None
    slice_1100: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_90, 3, 0, 9223372036854775807);  unsqueeze_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_19: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_1100, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_18: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_1100, [1, 256, 1, 257]);  slice_1100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_343: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_211, [1, 1, 1024, 513])
    permute_287: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    slice_1105: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_287, 0, 0, 9223372036854775807);  permute_287 = None
    slice_1106: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1105, 1, 0, 256);  slice_1105 = None
    slice_1107: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1106, 2, 0, 9223372036854775807);  slice_1106 = None
    slice_1108: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1107, 3, 0, 257);  slice_1107 = None
    full_43: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_23: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_18, torch.bool);  expand_18 = None
    where_37: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_23, full_43, slice_1108);  convert_element_type_23 = full_43 = slice_1108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_344: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_211, [1, 1, 1024, 513])
    permute_288: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    slice_1113: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_288, 0, 0, 9223372036854775807);  permute_288 = None
    slice_1114: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1113, 1, 0, 256);  slice_1113 = None
    slice_1115: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1114, 2, 0, 9223372036854775807);  slice_1114 = None
    slice_1116: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1115, 3, 0, 257);  slice_1115 = None
    copy_58: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1116, where_37);  slice_1116 = where_37 = None
    view_345: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_211, [1, 1, 1024, 513]);  slice_scatter_211 = None
    permute_289: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    slice_1117: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_289, 0, 0, 9223372036854775807)
    slice_1118: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1117, 1, 0, 256)
    slice_1119: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1118, 2, 0, 9223372036854775807)
    slice_scatter_212: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1119, copy_58, 3, 0, 257);  slice_1119 = copy_58 = None
    slice_scatter_213: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1118, slice_scatter_212, 2, 0, 9223372036854775807);  slice_1118 = slice_scatter_212 = None
    slice_scatter_214: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1117, slice_scatter_213, 1, 0, 256);  slice_1117 = slice_scatter_213 = None
    slice_scatter_215: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_289, slice_scatter_214, 0, 0, 9223372036854775807);  permute_289 = slice_scatter_214 = None
    permute_290: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_215, [0, 2, 1, 3]);  slice_scatter_215 = None
    view_346: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_290, [1, 4, 256, 513]);  permute_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_19: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_19, [1, 256, 1, 257]);  rev_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_348: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_346, [1, 1, 1024, 513])
    permute_292: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None
    slice_1128: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_292, 0, 0, 9223372036854775807);  permute_292 = None
    slice_1129: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1128, 1, -256, 9223372036854775807);  slice_1128 = None
    slice_1130: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1129, 2, 0, 9223372036854775807);  slice_1129 = None
    slice_1131: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1130, 3, -257, 9223372036854775807);  slice_1130 = None
    full_44: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_24: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_19, torch.bool);  expand_19 = None
    where_38: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_24, full_44, slice_1131);  convert_element_type_24 = full_44 = slice_1131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_349: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_346, [1, 1, 1024, 513])
    permute_293: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
    slice_1136: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_293, 0, 0, 9223372036854775807);  permute_293 = None
    slice_1137: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1136, 1, -256, 9223372036854775807);  slice_1136 = None
    slice_1138: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1137, 2, 0, 9223372036854775807);  slice_1137 = None
    slice_1139: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1138, 3, -257, 9223372036854775807);  slice_1138 = None
    copy_59: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1139, where_38);  slice_1139 = where_38 = None
    view_350: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_346, [1, 1, 1024, 513]);  view_346 = None
    permute_294: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    slice_1140: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_294, 0, 0, 9223372036854775807)
    slice_1141: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1140, 1, -256, 9223372036854775807)
    slice_1142: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1141, 2, 0, 9223372036854775807)
    slice_scatter_216: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1142, copy_59, 3, -257, 9223372036854775807);  slice_1142 = copy_59 = None
    slice_scatter_217: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1141, slice_scatter_216, 2, 0, 9223372036854775807);  slice_1141 = slice_scatter_216 = None
    slice_scatter_218: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1140, slice_scatter_217, 1, -256, 9223372036854775807);  slice_1140 = slice_scatter_217 = None
    slice_scatter_219: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_294, slice_scatter_218, 0, 0, 9223372036854775807);  permute_294 = slice_scatter_218 = None
    permute_295: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_219, [0, 2, 1, 3]);  slice_scatter_219 = None
    view_351: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_295, [1, 4, 256, 513]);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_353: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_333, [1, 12, 1024, 513]);  view_333 = None
    permute_297: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
    view_354: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_351, [1, 1, 1024, 513]);  view_351 = None
    permute_298: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
    add_46: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_297, permute_298);  permute_297 = permute_298 = None
    permute_299: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_46, [0, 2, 1, 3]);  add_46 = None
    view_356: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_299, [12, 4, 256, 513]);  permute_299 = None
    view_357: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_356, [1, 12, 1024, 513]);  view_356 = None
    permute_300: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_34: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
    amax_4: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_34, [-1], True)
    sub_36: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_34, amax_4);  clone_34 = amax_4 = None
    exp_4: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_5: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_47: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_1147: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_1148: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_1147, 1, 0, 9223372036854775807);  slice_1147 = None
    unsqueeze_91: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_1148, 2);  slice_1148 = None
    unsqueeze_92: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_92, scalar_tensor_19, div_47);  unsqueeze_92 = scalar_tensor_19 = div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_35: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_39);  where_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_358: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_305, [1024, 1, 12, 64]);  view_305 = None
    permute_301: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_358, [1, 0, 2, 3]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_302: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_35, [0, 2, 1, 3]);  clone_35 = None
    view_359: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_302, [12, 4, 256, 513]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_303: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_301, [0, 2, 1, 3]);  permute_301 = None
    view_360: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_303, [12, 1024, 64]);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_18: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_360, [0, 0, 256, 256], -1.0);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_29: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_18, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_19: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_359, [0, 257], 0.0);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_361: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_19, [12, 4, -1]);  constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_1149: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_361, 0, 0, 9223372036854775807);  view_361 = None
    slice_1150: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_1149, 1, 0, 9223372036854775807);  slice_1149 = None
    slice_1151: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_1150, 2, 0, -256);  slice_1150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_362: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_1151, [12, 4, 256, 769]);  slice_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_1152: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_362, 0, 0, 9223372036854775807);  view_362 = None
    slice_1153: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1152, 1, 0, 9223372036854775807);  slice_1152 = None
    slice_1154: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1153, 2, 0, 9223372036854775807);  slice_1153 = None
    slice_1155: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_1154, 3, 0, -1);  slice_1154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_93: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1155, 4);  slice_1155 = None
    permute_304: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_93, [0, 1, 2, 4, 3]);  unsqueeze_93 = None
    unsqueeze_94: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_29, 4);  as_strided_29 = None
    permute_305: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_94, [0, 1, 4, 3, 2]);  unsqueeze_94 = None
    permute_306: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_304, [0, 1, 2, 4, 3]);  permute_304 = None
    view_363: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_306, [48, 256, 768]);  permute_306 = None
    permute_307: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_305, [0, 1, 4, 3, 2]);  permute_305 = None
    clone_36: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
    view_364: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_36, [48, 768, 64]);  clone_36 = None
    bmm_9: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_363, view_364);  view_363 = view_364 = None
    view_365: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_9, [12, 4, 256, 1, 64]);  bmm_9 = None
    permute_308: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_365, [0, 1, 2, 4, 3]);  view_365 = None
    view_366: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_308, [12, 4, 256, 64]);  permute_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_367: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_366, [1, 12, 1024, 64]);  view_366 = None
    permute_309: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_310: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_309, [1, 0, 2, 3]);  permute_309 = None
    clone_37: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
    view_368: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_37, [1024, 1, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_311: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_368, [1, 0, 2]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_369: "f32[1024, 768]" = torch.ops.aten.view.default(permute_311, [1024, 768]);  permute_311 = None
    permute_312: "f32[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_27: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg71_1, view_369, permute_312);  arg71_1 = view_369 = permute_312 = None
    view_370: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_27, [1, 1024, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_38: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_370);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_48: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_38, add_43);  clone_38 = add_43 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    add_49: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_38: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_17);  add_48 = getitem_17 = None
    mul_33: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_8);  sub_38 = rsqrt_8 = None
    mul_34: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg72_1);  mul_33 = arg72_1 = None
    add_50: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_34, arg73_1);  mul_34 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_371: "f32[1024, 768]" = torch.ops.aten.view.default(add_50, [1024, 768])
    permute_313: "f32[768, 3072]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_28: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg75_1, view_371, permute_313);  arg75_1 = view_371 = permute_313 = None
    view_372: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 1024, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_35: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_372, 0.5)
    mul_36: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_372, 0.7071067811865476);  view_372 = None
    erf_4: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_51: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_37: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_35, add_51);  mul_35 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_373: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_37, [1024, 3072]);  mul_37 = None
    permute_314: "f32[3072, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    addmm_29: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg77_1, view_373, permute_314);  arg77_1 = view_373 = permute_314 = None
    view_374: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_29, [1, 1024, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_39: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_374);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_52: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_39, add_50);  clone_39 = add_50 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    add_53: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_39: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_19);  add_52 = getitem_19 = None
    mul_38: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_9);  sub_39 = rsqrt_9 = None
    mul_39: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, arg78_1);  mul_38 = arg78_1 = None
    add_54: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_39, arg79_1);  mul_39 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_315: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_54, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_375: "f32[1024, 768]" = torch.ops.aten.view.default(permute_315, [1024, 768])
    permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_30: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg81_1, view_375, permute_316);  arg81_1 = view_375 = permute_316 = None
    view_376: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_30, [1024, 1, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_377: "f32[1024, 768]" = torch.ops.aten.view.default(permute_315, [1024, 768])
    permute_317: "f32[768, 768]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm_31: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg83_1, view_377, permute_317);  arg83_1 = view_377 = permute_317 = None
    view_378: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_31, [1024, 1, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_379: "f32[1024, 768]" = torch.ops.aten.view.default(permute_315, [1024, 768]);  permute_315 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_32: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg85_1, view_379, permute_318);  arg85_1 = view_379 = permute_318 = None
    view_380: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_32, [1024, 1, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_50: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_376, 8.0);  view_376 = None
    view_381: "f32[1024, 768]" = torch.ops.aten.view.default(div_50, [1024, 768]);  div_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_384: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_378, [1024, 1, 12, 64]);  view_378 = None
    permute_320: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_384, [1, 0, 2, 3]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_322: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_320, [0, 2, 1, 3]);  permute_320 = None
    view_386: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_322, [12, 1024, 64]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_388: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_386, [12, 2, 512, 64]);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_31: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_388, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_96: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_31, 4);  as_strided_31 = None
    permute_324: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_96, [0, 1, 4, 2, 3]);  unsqueeze_96 = None
    view_389: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_381, [1024, 1, 768]);  view_381 = None
    view_390: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_389, [1024, 1, 12, 64]);  view_389 = None
    permute_326: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_390, [1, 0, 2, 3]);  view_390 = None
    permute_327: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_326, [0, 2, 1, 3]);  permute_326 = None
    view_391: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_327, [12, 1024, 64]);  permute_327 = None
    view_392: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_391, [12, 2, 512, 64]);  view_391 = None
    as_strided_32: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_392, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_392 = None
    unsqueeze_97: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_32, 4);  as_strided_32 = None
    permute_328: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_97, [0, 1, 2, 4, 3]);  unsqueeze_97 = None
    permute_329: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_328, [0, 1, 2, 4, 3]);  permute_328 = None
    clone_40: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
    view_393: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_40, [36, 512, 64]);  clone_40 = None
    permute_330: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_324, [0, 1, 4, 3, 2]);  permute_324 = None
    clone_41: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
    view_394: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_41, [36, 64, 512]);  clone_41 = None
    bmm_10: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_393, view_394);  view_393 = view_394 = None
    view_395: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_10, [12, 3, 512, 1, 512]);  bmm_10 = None
    permute_331: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_395, [0, 1, 2, 4, 3]);  view_395 = None
    view_396: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_331, [12, 3, 512, 512]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_20: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_396, [0, 0, 0, 1], 0.0);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_397: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_20, [12, 3, 512, 513]);  constant_pad_nd_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_45: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1156: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_397, 0, 0, 9223372036854775807)
    slice_1157: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1156, 1, 0, 9223372036854775807);  slice_1156 = None
    slice_1158: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1157, 2, 0, 256);  slice_1157 = None
    slice_1159: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1158, 3, 0, 257);  slice_1158 = None
    slice_1160: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_45, 0, 0, 9223372036854775807)
    slice_1161: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1160, 1, 0, -1);  slice_1160 = None
    slice_1162: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1161, 2, 0, 9223372036854775807);  slice_1161 = None
    slice_1163: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1162, 3, 256, 9223372036854775807);  slice_1162 = None
    copy_60: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1163, slice_1159);  slice_1163 = slice_1159 = None
    slice_1164: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_45, 0, 0, 9223372036854775807)
    slice_1165: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1164, 1, 0, -1)
    slice_1166: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1165, 2, 0, 9223372036854775807)
    slice_scatter_220: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1166, copy_60, 3, 256, 9223372036854775807);  slice_1166 = copy_60 = None
    slice_scatter_221: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1165, slice_scatter_220, 2, 0, 9223372036854775807);  slice_1165 = slice_scatter_220 = None
    slice_scatter_222: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1164, slice_scatter_221, 1, 0, -1);  slice_1164 = slice_scatter_221 = None
    slice_scatter_223: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_45, slice_scatter_222, 0, 0, 9223372036854775807);  full_45 = slice_scatter_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1171: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_397, 0, 0, 9223372036854775807)
    select_100: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1171, 1, -1);  slice_1171 = None
    slice_1172: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_100, 1, 256, 9223372036854775807);  select_100 = None
    slice_1173: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1172, 2, 0, 257);  slice_1172 = None
    slice_1177: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_223, 0, 0, 9223372036854775807)
    select_102: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1177, 1, -1);  slice_1177 = None
    slice_1178: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_102, 1, 0, 9223372036854775807);  select_102 = None
    slice_1179: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1178, 2, 256, 9223372036854775807);  slice_1178 = None
    copy_61: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_1179, slice_1173);  slice_1179 = slice_1173 = None
    slice_1180: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_223, 0, 0, 9223372036854775807)
    select_103: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1180, 1, -1)
    slice_1181: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_103, 1, 0, 9223372036854775807)
    slice_scatter_224: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1181, copy_61, 2, 256, 9223372036854775807);  slice_1181 = copy_61 = None
    slice_scatter_225: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_103, slice_scatter_224, 1, 0, 9223372036854775807);  select_103 = slice_scatter_224 = None
    select_scatter_20: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1180, slice_scatter_225, 1, -1);  slice_1180 = slice_scatter_225 = None
    slice_scatter_226: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_223, select_scatter_20, 0, 0, 9223372036854775807);  slice_scatter_223 = select_scatter_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1185: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_397, 0, 0, 9223372036854775807)
    slice_1186: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1185, 1, 0, 9223372036854775807);  slice_1185 = None
    slice_1187: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1186, 2, -257, -1);  slice_1186 = None
    slice_1188: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1187, 3, 257, 9223372036854775807);  slice_1187 = None
    slice_1193: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_226, 0, 0, 9223372036854775807)
    slice_1194: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1193, 1, 1, 9223372036854775807);  slice_1193 = None
    slice_1195: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1194, 2, 0, 9223372036854775807);  slice_1194 = None
    slice_1196: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1195, 3, 0, 256);  slice_1195 = None
    copy_62: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1196, slice_1188);  slice_1196 = slice_1188 = None
    slice_1197: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_226, 0, 0, 9223372036854775807)
    slice_1198: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1197, 1, 1, 9223372036854775807)
    slice_1199: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1198, 2, 0, 9223372036854775807)
    slice_scatter_227: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1199, copy_62, 3, 0, 256);  slice_1199 = copy_62 = None
    slice_scatter_228: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1198, slice_scatter_227, 2, 0, 9223372036854775807);  slice_1198 = slice_scatter_227 = None
    slice_scatter_229: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1197, slice_scatter_228, 1, 1, 9223372036854775807);  slice_1197 = slice_scatter_228 = None
    slice_scatter_230: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_226, slice_scatter_229, 0, 0, 9223372036854775807);  slice_scatter_226 = slice_scatter_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_1204: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_397, 0, 0, 9223372036854775807);  view_397 = None
    select_105: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1204, 1, 0);  slice_1204 = None
    slice_1205: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_105, 1, 0, 255);  select_105 = None
    slice_1206: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1205, 2, -255, 9223372036854775807);  slice_1205 = None
    slice_1210: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_230, 0, 0, 9223372036854775807)
    select_107: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1210, 1, 0);  slice_1210 = None
    slice_1211: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_107, 1, 1, 256);  select_107 = None
    slice_1212: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1211, 2, 1, 256);  slice_1211 = None
    copy_63: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_1212, slice_1206);  slice_1212 = slice_1206 = None
    slice_1213: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_230, 0, 0, 9223372036854775807)
    select_108: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1213, 1, 0)
    slice_1214: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_108, 1, 1, 256)
    slice_scatter_231: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1214, copy_63, 2, 1, 256);  slice_1214 = copy_63 = None
    slice_scatter_232: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_108, slice_scatter_231, 1, 1, 256);  select_108 = slice_scatter_231 = None
    select_scatter_21: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1213, slice_scatter_232, 1, 0);  slice_1213 = slice_scatter_232 = None
    slice_scatter_233: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_230, select_scatter_21, 0, 0, 9223372036854775807);  slice_scatter_230 = select_scatter_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_46: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_20: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_98: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_20, -2);  iota_20 = None
    iota_21: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_99: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
    sub_41: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_98, unsqueeze_99);  unsqueeze_98 = unsqueeze_99 = None
    le_10: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_41, 0);  sub_41 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_40: "f32[256, 257]" = torch.ops.aten.where.self(le_10, full_46, scalar_tensor_20);  le_10 = full_46 = scalar_tensor_20 = None
    rev_20: "f32[256, 257]" = torch.ops.prims.rev.default(where_40, [0]);  where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_100: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_20, 0);  rev_20 = None
    slice_1218: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_100, 1, 0, 9223372036854775807);  unsqueeze_100 = None
    unsqueeze_101: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_1218, 2);  slice_1218 = None
    slice_1219: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_101, 3, 0, 9223372036854775807);  unsqueeze_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_21: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_1219, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_20: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_1219, [1, 256, 12, 257]);  slice_1219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_400: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_233, [1, 12, 1024, 513])
    permute_334: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
    slice_1224: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_334, 0, 0, 9223372036854775807);  permute_334 = None
    slice_1225: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1224, 1, 0, 256);  slice_1224 = None
    slice_1226: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1225, 2, 0, 9223372036854775807);  slice_1225 = None
    slice_1227: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1226, 3, 0, 257);  slice_1226 = None
    full_47: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_25: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_20, torch.bool);  expand_20 = None
    where_41: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_25, full_47, slice_1227);  convert_element_type_25 = full_47 = slice_1227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_401: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_233, [1, 12, 1024, 513])
    permute_335: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None
    slice_1232: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_335, 0, 0, 9223372036854775807);  permute_335 = None
    slice_1233: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1232, 1, 0, 256);  slice_1232 = None
    slice_1234: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1233, 2, 0, 9223372036854775807);  slice_1233 = None
    slice_1235: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1234, 3, 0, 257);  slice_1234 = None
    copy_64: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1235, where_41);  slice_1235 = where_41 = None
    view_402: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_233, [1, 12, 1024, 513]);  slice_scatter_233 = None
    permute_336: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    slice_1236: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_336, 0, 0, 9223372036854775807)
    slice_1237: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1236, 1, 0, 256)
    slice_1238: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1237, 2, 0, 9223372036854775807)
    slice_scatter_234: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1238, copy_64, 3, 0, 257);  slice_1238 = copy_64 = None
    slice_scatter_235: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1237, slice_scatter_234, 2, 0, 9223372036854775807);  slice_1237 = slice_scatter_234 = None
    slice_scatter_236: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1236, slice_scatter_235, 1, 0, 256);  slice_1236 = slice_scatter_235 = None
    slice_scatter_237: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_336, slice_scatter_236, 0, 0, 9223372036854775807);  permute_336 = slice_scatter_236 = None
    permute_337: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_237, [0, 2, 1, 3]);  slice_scatter_237 = None
    view_403: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_337, [12, 4, 256, 513]);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_21: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_21, [1, 256, 12, 257]);  rev_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_405: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_403, [1, 12, 1024, 513])
    permute_339: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
    slice_1247: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_339, 0, 0, 9223372036854775807);  permute_339 = None
    slice_1248: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1247, 1, -256, 9223372036854775807);  slice_1247 = None
    slice_1249: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1248, 2, 0, 9223372036854775807);  slice_1248 = None
    slice_1250: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1249, 3, -257, 9223372036854775807);  slice_1249 = None
    full_48: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_26: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_21, torch.bool);  expand_21 = None
    where_42: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_26, full_48, slice_1250);  convert_element_type_26 = full_48 = slice_1250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_406: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_403, [1, 12, 1024, 513])
    permute_340: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
    slice_1255: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_340, 0, 0, 9223372036854775807);  permute_340 = None
    slice_1256: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1255, 1, -256, 9223372036854775807);  slice_1255 = None
    slice_1257: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1256, 2, 0, 9223372036854775807);  slice_1256 = None
    slice_1258: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1257, 3, -257, 9223372036854775807);  slice_1257 = None
    copy_65: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1258, where_42);  slice_1258 = where_42 = None
    view_407: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_403, [1, 12, 1024, 513]);  view_403 = None
    permute_341: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
    slice_1259: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_341, 0, 0, 9223372036854775807)
    slice_1260: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1259, 1, -256, 9223372036854775807)
    slice_1261: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1260, 2, 0, 9223372036854775807)
    slice_scatter_238: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1261, copy_65, 3, -257, 9223372036854775807);  slice_1261 = copy_65 = None
    slice_scatter_239: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1260, slice_scatter_238, 2, 0, 9223372036854775807);  slice_1260 = slice_scatter_238 = None
    slice_scatter_240: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1259, slice_scatter_239, 1, -256, 9223372036854775807);  slice_1259 = slice_scatter_239 = None
    slice_scatter_241: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_341, slice_scatter_240, 0, 0, 9223372036854775807);  permute_341 = slice_scatter_240 = None
    permute_342: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_241, [0, 2, 1, 3]);  slice_scatter_241 = None
    view_408: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_342, [12, 4, 256, 513]);  permute_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_5: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_1266: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_5, 0, 0, 9223372036854775807);  ne_5 = None
    slice_1267: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_1266, 1, 0, 9223372036854775807);  slice_1266 = None
    unsqueeze_102: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_1267, 2);  slice_1267 = None
    unsqueeze_103: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 3);  unsqueeze_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_27: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_103, torch.float32)
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_43: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_103, scalar_tensor_21, convert_element_type_27);  unsqueeze_103 = scalar_tensor_21 = convert_element_type_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_49: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_344: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_49, [0, 2, 1, 3]);  full_49 = None
    view_410: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_344, [1, 1024, 1]);  permute_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_345: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_43, [0, 2, 1, 3]);  where_43 = None
    view_411: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_345, [1, 1024, 1]);  permute_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_412: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_410, [1, 2, 512, 1]);  view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_33: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_412, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_413: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_411, [1, 2, 512, 1]);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_34: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_413, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_104: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_33, 4);  as_strided_33 = None
    permute_346: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_104, [0, 1, 2, 4, 3]);  unsqueeze_104 = None
    unsqueeze_105: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_34, 4);  as_strided_34 = None
    permute_347: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_105, [0, 1, 4, 2, 3]);  unsqueeze_105 = None
    mul_40: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_346, permute_347);  permute_346 = permute_347 = None
    view_414: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_40, [1, 3, 512, 512]);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_21: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_414, [0, 0, 0, 1], 0.0);  view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_415: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_21, [1, 3, 512, 513]);  constant_pad_nd_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_50: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1268: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_415, 0, 0, 9223372036854775807)
    slice_1269: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1268, 1, 0, 9223372036854775807);  slice_1268 = None
    slice_1270: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1269, 2, 0, 256);  slice_1269 = None
    slice_1271: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1270, 3, 0, 257);  slice_1270 = None
    slice_1272: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_50, 0, 0, 9223372036854775807)
    slice_1273: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1272, 1, 0, -1);  slice_1272 = None
    slice_1274: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1273, 2, 0, 9223372036854775807);  slice_1273 = None
    slice_1275: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1274, 3, 256, 9223372036854775807);  slice_1274 = None
    copy_66: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1275, slice_1271);  slice_1275 = slice_1271 = None
    slice_1276: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_50, 0, 0, 9223372036854775807)
    slice_1277: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1276, 1, 0, -1)
    slice_1278: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1277, 2, 0, 9223372036854775807)
    slice_scatter_242: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1278, copy_66, 3, 256, 9223372036854775807);  slice_1278 = copy_66 = None
    slice_scatter_243: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1277, slice_scatter_242, 2, 0, 9223372036854775807);  slice_1277 = slice_scatter_242 = None
    slice_scatter_244: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1276, slice_scatter_243, 1, 0, -1);  slice_1276 = slice_scatter_243 = None
    slice_scatter_245: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_50, slice_scatter_244, 0, 0, 9223372036854775807);  full_50 = slice_scatter_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1283: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_415, 0, 0, 9223372036854775807)
    select_110: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_1283, 1, -1);  slice_1283 = None
    slice_1284: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_110, 1, 256, 9223372036854775807);  select_110 = None
    slice_1285: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1284, 2, 0, 257);  slice_1284 = None
    slice_1289: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_245, 0, 0, 9223372036854775807)
    select_112: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1289, 1, -1);  slice_1289 = None
    slice_1290: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_112, 1, 0, 9223372036854775807);  select_112 = None
    slice_1291: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1290, 2, 256, 9223372036854775807);  slice_1290 = None
    copy_67: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_1291, slice_1285);  slice_1291 = slice_1285 = None
    slice_1292: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_245, 0, 0, 9223372036854775807)
    select_113: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1292, 1, -1)
    slice_1293: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_113, 1, 0, 9223372036854775807)
    slice_scatter_246: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1293, copy_67, 2, 256, 9223372036854775807);  slice_1293 = copy_67 = None
    slice_scatter_247: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_113, slice_scatter_246, 1, 0, 9223372036854775807);  select_113 = slice_scatter_246 = None
    select_scatter_22: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1292, slice_scatter_247, 1, -1);  slice_1292 = slice_scatter_247 = None
    slice_scatter_248: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_245, select_scatter_22, 0, 0, 9223372036854775807);  slice_scatter_245 = select_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1297: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_415, 0, 0, 9223372036854775807)
    slice_1298: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1297, 1, 0, 9223372036854775807);  slice_1297 = None
    slice_1299: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1298, 2, -257, -1);  slice_1298 = None
    slice_1300: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1299, 3, 257, 9223372036854775807);  slice_1299 = None
    slice_1305: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_248, 0, 0, 9223372036854775807)
    slice_1306: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1305, 1, 1, 9223372036854775807);  slice_1305 = None
    slice_1307: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1306, 2, 0, 9223372036854775807);  slice_1306 = None
    slice_1308: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1307, 3, 0, 256);  slice_1307 = None
    copy_68: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1308, slice_1300);  slice_1308 = slice_1300 = None
    slice_1309: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_248, 0, 0, 9223372036854775807)
    slice_1310: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1309, 1, 1, 9223372036854775807)
    slice_1311: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1310, 2, 0, 9223372036854775807)
    slice_scatter_249: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1311, copy_68, 3, 0, 256);  slice_1311 = copy_68 = None
    slice_scatter_250: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1310, slice_scatter_249, 2, 0, 9223372036854775807);  slice_1310 = slice_scatter_249 = None
    slice_scatter_251: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1309, slice_scatter_250, 1, 1, 9223372036854775807);  slice_1309 = slice_scatter_250 = None
    slice_scatter_252: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_248, slice_scatter_251, 0, 0, 9223372036854775807);  slice_scatter_248 = slice_scatter_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_1316: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_415, 0, 0, 9223372036854775807);  view_415 = None
    select_115: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_1316, 1, 0);  slice_1316 = None
    slice_1317: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_115, 1, 0, 255);  select_115 = None
    slice_1318: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1317, 2, -255, 9223372036854775807);  slice_1317 = None
    slice_1322: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_252, 0, 0, 9223372036854775807)
    select_117: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1322, 1, 0);  slice_1322 = None
    slice_1323: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_117, 1, 1, 256);  select_117 = None
    slice_1324: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1323, 2, 1, 256);  slice_1323 = None
    copy_69: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_1324, slice_1318);  slice_1324 = slice_1318 = None
    slice_1325: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_252, 0, 0, 9223372036854775807)
    select_118: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1325, 1, 0)
    slice_1326: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_118, 1, 1, 256)
    slice_scatter_253: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1326, copy_69, 2, 1, 256);  slice_1326 = copy_69 = None
    slice_scatter_254: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_118, slice_scatter_253, 1, 1, 256);  select_118 = slice_scatter_253 = None
    select_scatter_23: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1325, slice_scatter_254, 1, 0);  slice_1325 = slice_scatter_254 = None
    slice_scatter_255: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_252, select_scatter_23, 0, 0, 9223372036854775807);  slice_scatter_252 = select_scatter_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_51: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_22: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_106: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_22, -2);  iota_22 = None
    iota_23: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_107: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
    sub_43: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_106, unsqueeze_107);  unsqueeze_106 = unsqueeze_107 = None
    le_11: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_43, 0);  sub_43 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_44: "f32[256, 257]" = torch.ops.aten.where.self(le_11, full_51, scalar_tensor_22);  le_11 = full_51 = scalar_tensor_22 = None
    rev_22: "f32[256, 257]" = torch.ops.prims.rev.default(where_44, [0]);  where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_108: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_22, 0);  rev_22 = None
    slice_1330: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_108, 1, 0, 9223372036854775807);  unsqueeze_108 = None
    unsqueeze_109: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_1330, 2);  slice_1330 = None
    slice_1331: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_109, 3, 0, 9223372036854775807);  unsqueeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_23: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_1331, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_22: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_1331, [1, 256, 1, 257]);  slice_1331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_418: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_255, [1, 1, 1024, 513])
    permute_350: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
    slice_1336: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_350, 0, 0, 9223372036854775807);  permute_350 = None
    slice_1337: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1336, 1, 0, 256);  slice_1336 = None
    slice_1338: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1337, 2, 0, 9223372036854775807);  slice_1337 = None
    slice_1339: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1338, 3, 0, 257);  slice_1338 = None
    full_52: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_28: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_22, torch.bool);  expand_22 = None
    where_45: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_28, full_52, slice_1339);  convert_element_type_28 = full_52 = slice_1339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_419: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_255, [1, 1, 1024, 513])
    permute_351: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None
    slice_1344: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_351, 0, 0, 9223372036854775807);  permute_351 = None
    slice_1345: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1344, 1, 0, 256);  slice_1344 = None
    slice_1346: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1345, 2, 0, 9223372036854775807);  slice_1345 = None
    slice_1347: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1346, 3, 0, 257);  slice_1346 = None
    copy_70: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1347, where_45);  slice_1347 = where_45 = None
    view_420: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_255, [1, 1, 1024, 513]);  slice_scatter_255 = None
    permute_352: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    slice_1348: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_352, 0, 0, 9223372036854775807)
    slice_1349: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1348, 1, 0, 256)
    slice_1350: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1349, 2, 0, 9223372036854775807)
    slice_scatter_256: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1350, copy_70, 3, 0, 257);  slice_1350 = copy_70 = None
    slice_scatter_257: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1349, slice_scatter_256, 2, 0, 9223372036854775807);  slice_1349 = slice_scatter_256 = None
    slice_scatter_258: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1348, slice_scatter_257, 1, 0, 256);  slice_1348 = slice_scatter_257 = None
    slice_scatter_259: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_352, slice_scatter_258, 0, 0, 9223372036854775807);  permute_352 = slice_scatter_258 = None
    permute_353: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_259, [0, 2, 1, 3]);  slice_scatter_259 = None
    view_421: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_353, [1, 4, 256, 513]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_23: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_23, [1, 256, 1, 257]);  rev_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_423: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_421, [1, 1, 1024, 513])
    permute_355: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None
    slice_1359: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_355, 0, 0, 9223372036854775807);  permute_355 = None
    slice_1360: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1359, 1, -256, 9223372036854775807);  slice_1359 = None
    slice_1361: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1360, 2, 0, 9223372036854775807);  slice_1360 = None
    slice_1362: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1361, 3, -257, 9223372036854775807);  slice_1361 = None
    full_53: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_29: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_23, torch.bool);  expand_23 = None
    where_46: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_29, full_53, slice_1362);  convert_element_type_29 = full_53 = slice_1362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_424: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_421, [1, 1, 1024, 513])
    permute_356: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
    slice_1367: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_356, 0, 0, 9223372036854775807);  permute_356 = None
    slice_1368: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1367, 1, -256, 9223372036854775807);  slice_1367 = None
    slice_1369: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1368, 2, 0, 9223372036854775807);  slice_1368 = None
    slice_1370: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1369, 3, -257, 9223372036854775807);  slice_1369 = None
    copy_71: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1370, where_46);  slice_1370 = where_46 = None
    view_425: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_421, [1, 1, 1024, 513]);  view_421 = None
    permute_357: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    slice_1371: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_357, 0, 0, 9223372036854775807)
    slice_1372: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1371, 1, -256, 9223372036854775807)
    slice_1373: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1372, 2, 0, 9223372036854775807)
    slice_scatter_260: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1373, copy_71, 3, -257, 9223372036854775807);  slice_1373 = copy_71 = None
    slice_scatter_261: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1372, slice_scatter_260, 2, 0, 9223372036854775807);  slice_1372 = slice_scatter_260 = None
    slice_scatter_262: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1371, slice_scatter_261, 1, -256, 9223372036854775807);  slice_1371 = slice_scatter_261 = None
    slice_scatter_263: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_357, slice_scatter_262, 0, 0, 9223372036854775807);  permute_357 = slice_scatter_262 = None
    permute_358: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_263, [0, 2, 1, 3]);  slice_scatter_263 = None
    view_426: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_358, [1, 4, 256, 513]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_428: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_408, [1, 12, 1024, 513]);  view_408 = None
    permute_360: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
    view_429: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_426, [1, 1, 1024, 513]);  view_426 = None
    permute_361: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_429, [0, 2, 1, 3]);  view_429 = None
    add_57: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_360, permute_361);  permute_360 = permute_361 = None
    permute_362: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_57, [0, 2, 1, 3]);  add_57 = None
    view_431: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_362, [12, 4, 256, 513]);  permute_362 = None
    view_432: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_431, [1, 12, 1024, 513]);  view_431 = None
    permute_363: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_42: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
    amax_5: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_42, [-1], True)
    sub_44: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_42, amax_5);  clone_42 = amax_5 = None
    exp_5: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_6: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_57: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_1378: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_1379: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_1378, 1, 0, 9223372036854775807);  slice_1378 = None
    unsqueeze_110: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_1379, 2);  slice_1379 = None
    unsqueeze_111: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 3);  unsqueeze_110 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_47: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_111, scalar_tensor_23, div_57);  unsqueeze_111 = scalar_tensor_23 = div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_43: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_47);  where_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_433: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_380, [1024, 1, 12, 64]);  view_380 = None
    permute_364: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_433, [1, 0, 2, 3]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_365: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_43, [0, 2, 1, 3]);  clone_43 = None
    view_434: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_365, [12, 4, 256, 513]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_366: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_364, [0, 2, 1, 3]);  permute_364 = None
    view_435: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_366, [12, 1024, 64]);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_22: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_435, [0, 0, 256, 256], -1.0);  view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_35: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_22, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_23: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_434, [0, 257], 0.0);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_436: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_23, [12, 4, -1]);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_1380: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_436, 0, 0, 9223372036854775807);  view_436 = None
    slice_1381: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_1380, 1, 0, 9223372036854775807);  slice_1380 = None
    slice_1382: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_1381, 2, 0, -256);  slice_1381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_437: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_1382, [12, 4, 256, 769]);  slice_1382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_1383: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_437, 0, 0, 9223372036854775807);  view_437 = None
    slice_1384: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1383, 1, 0, 9223372036854775807);  slice_1383 = None
    slice_1385: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1384, 2, 0, 9223372036854775807);  slice_1384 = None
    slice_1386: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_1385, 3, 0, -1);  slice_1385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_112: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1386, 4);  slice_1386 = None
    permute_367: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_112, [0, 1, 2, 4, 3]);  unsqueeze_112 = None
    unsqueeze_113: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_35, 4);  as_strided_35 = None
    permute_368: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_113, [0, 1, 4, 3, 2]);  unsqueeze_113 = None
    permute_369: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_367, [0, 1, 2, 4, 3]);  permute_367 = None
    view_438: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_369, [48, 256, 768]);  permute_369 = None
    permute_370: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_368, [0, 1, 4, 3, 2]);  permute_368 = None
    clone_44: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_439: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_44, [48, 768, 64]);  clone_44 = None
    bmm_11: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_438, view_439);  view_438 = view_439 = None
    view_440: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_11, [12, 4, 256, 1, 64]);  bmm_11 = None
    permute_371: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_440, [0, 1, 2, 4, 3]);  view_440 = None
    view_441: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_371, [12, 4, 256, 64]);  permute_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_442: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_441, [1, 12, 1024, 64]);  view_441 = None
    permute_372: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_373: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_372, [1, 0, 2, 3]);  permute_372 = None
    clone_45: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_373, memory_format = torch.contiguous_format);  permute_373 = None
    view_443: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_45, [1024, 1, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_374: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_443, [1, 0, 2]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_444: "f32[1024, 768]" = torch.ops.aten.view.default(permute_374, [1024, 768]);  permute_374 = None
    permute_375: "f32[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_33: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg87_1, view_444, permute_375);  arg87_1 = view_444 = permute_375 = None
    view_445: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_33, [1, 1024, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_46: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_445);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_46, add_54);  clone_46 = add_54 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    add_60: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_46: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_21);  add_59 = getitem_21 = None
    mul_41: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_10);  sub_46 = rsqrt_10 = None
    mul_42: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg88_1);  mul_41 = arg88_1 = None
    add_61: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_42, arg89_1);  mul_42 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_446: "f32[1024, 768]" = torch.ops.aten.view.default(add_61, [1024, 768])
    permute_376: "f32[768, 3072]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_34: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg91_1, view_446, permute_376);  arg91_1 = view_446 = permute_376 = None
    view_447: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 1024, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_43: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_447, 0.5)
    mul_44: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_447, 0.7071067811865476);  view_447 = None
    erf_5: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
    add_62: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_45: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_43, add_62);  mul_43 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_448: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_45, [1024, 3072]);  mul_45 = None
    permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    addmm_35: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg93_1, view_448, permute_377);  arg93_1 = view_448 = permute_377 = None
    view_449: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_35, [1, 1024, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_47: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_449);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_47, add_61);  clone_47 = add_61 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    add_64: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_47: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_23);  add_63 = getitem_23 = None
    mul_46: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_11);  sub_47 = rsqrt_11 = None
    mul_47: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_46, arg94_1);  mul_46 = arg94_1 = None
    add_65: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_47, arg95_1);  mul_47 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_378: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_65, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_450: "f32[1024, 768]" = torch.ops.aten.view.default(permute_378, [1024, 768])
    permute_379: "f32[768, 768]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_36: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg97_1, view_450, permute_379);  arg97_1 = view_450 = permute_379 = None
    view_451: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_36, [1024, 1, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_452: "f32[1024, 768]" = torch.ops.aten.view.default(permute_378, [1024, 768])
    permute_380: "f32[768, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_37: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg99_1, view_452, permute_380);  arg99_1 = view_452 = permute_380 = None
    view_453: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_37, [1024, 1, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_454: "f32[1024, 768]" = torch.ops.aten.view.default(permute_378, [1024, 768]);  permute_378 = None
    permute_381: "f32[768, 768]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_38: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg101_1, view_454, permute_381);  arg101_1 = view_454 = permute_381 = None
    view_455: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_38, [1024, 1, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_60: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_451, 8.0);  view_451 = None
    view_456: "f32[1024, 768]" = torch.ops.aten.view.default(div_60, [1024, 768]);  div_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_459: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_453, [1024, 1, 12, 64]);  view_453 = None
    permute_383: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_459, [1, 0, 2, 3]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_385: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_383, [0, 2, 1, 3]);  permute_383 = None
    view_461: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_385, [12, 1024, 64]);  permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_463: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_461, [12, 2, 512, 64]);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_37: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_463, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_115: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_37, 4);  as_strided_37 = None
    permute_387: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_115, [0, 1, 4, 2, 3]);  unsqueeze_115 = None
    view_464: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_456, [1024, 1, 768]);  view_456 = None
    view_465: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_464, [1024, 1, 12, 64]);  view_464 = None
    permute_389: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_465, [1, 0, 2, 3]);  view_465 = None
    permute_390: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_389, [0, 2, 1, 3]);  permute_389 = None
    view_466: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_390, [12, 1024, 64]);  permute_390 = None
    view_467: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_466, [12, 2, 512, 64]);  view_466 = None
    as_strided_38: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_467, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_467 = None
    unsqueeze_116: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_38, 4);  as_strided_38 = None
    permute_391: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_116, [0, 1, 2, 4, 3]);  unsqueeze_116 = None
    permute_392: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_391, [0, 1, 2, 4, 3]);  permute_391 = None
    clone_48: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_468: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_48, [36, 512, 64]);  clone_48 = None
    permute_393: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_387, [0, 1, 4, 3, 2]);  permute_387 = None
    clone_49: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
    view_469: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_49, [36, 64, 512]);  clone_49 = None
    bmm_12: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_468, view_469);  view_468 = view_469 = None
    view_470: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_12, [12, 3, 512, 1, 512]);  bmm_12 = None
    permute_394: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_470, [0, 1, 2, 4, 3]);  view_470 = None
    view_471: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_394, [12, 3, 512, 512]);  permute_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_24: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_471, [0, 0, 0, 1], 0.0);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_472: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_24, [12, 3, 512, 513]);  constant_pad_nd_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_54: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1387: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_472, 0, 0, 9223372036854775807)
    slice_1388: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1387, 1, 0, 9223372036854775807);  slice_1387 = None
    slice_1389: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1388, 2, 0, 256);  slice_1388 = None
    slice_1390: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1389, 3, 0, 257);  slice_1389 = None
    slice_1391: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_54, 0, 0, 9223372036854775807)
    slice_1392: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1391, 1, 0, -1);  slice_1391 = None
    slice_1393: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1392, 2, 0, 9223372036854775807);  slice_1392 = None
    slice_1394: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1393, 3, 256, 9223372036854775807);  slice_1393 = None
    copy_72: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1394, slice_1390);  slice_1394 = slice_1390 = None
    slice_1395: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_54, 0, 0, 9223372036854775807)
    slice_1396: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1395, 1, 0, -1)
    slice_1397: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1396, 2, 0, 9223372036854775807)
    slice_scatter_264: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1397, copy_72, 3, 256, 9223372036854775807);  slice_1397 = copy_72 = None
    slice_scatter_265: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1396, slice_scatter_264, 2, 0, 9223372036854775807);  slice_1396 = slice_scatter_264 = None
    slice_scatter_266: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1395, slice_scatter_265, 1, 0, -1);  slice_1395 = slice_scatter_265 = None
    slice_scatter_267: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_54, slice_scatter_266, 0, 0, 9223372036854775807);  full_54 = slice_scatter_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1402: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_472, 0, 0, 9223372036854775807)
    select_120: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1402, 1, -1);  slice_1402 = None
    slice_1403: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_120, 1, 256, 9223372036854775807);  select_120 = None
    slice_1404: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1403, 2, 0, 257);  slice_1403 = None
    slice_1408: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_267, 0, 0, 9223372036854775807)
    select_122: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1408, 1, -1);  slice_1408 = None
    slice_1409: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_122, 1, 0, 9223372036854775807);  select_122 = None
    slice_1410: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1409, 2, 256, 9223372036854775807);  slice_1409 = None
    copy_73: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_1410, slice_1404);  slice_1410 = slice_1404 = None
    slice_1411: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_267, 0, 0, 9223372036854775807)
    select_123: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1411, 1, -1)
    slice_1412: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_123, 1, 0, 9223372036854775807)
    slice_scatter_268: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1412, copy_73, 2, 256, 9223372036854775807);  slice_1412 = copy_73 = None
    slice_scatter_269: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_123, slice_scatter_268, 1, 0, 9223372036854775807);  select_123 = slice_scatter_268 = None
    select_scatter_24: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1411, slice_scatter_269, 1, -1);  slice_1411 = slice_scatter_269 = None
    slice_scatter_270: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_267, select_scatter_24, 0, 0, 9223372036854775807);  slice_scatter_267 = select_scatter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1416: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_472, 0, 0, 9223372036854775807)
    slice_1417: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1416, 1, 0, 9223372036854775807);  slice_1416 = None
    slice_1418: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1417, 2, -257, -1);  slice_1417 = None
    slice_1419: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1418, 3, 257, 9223372036854775807);  slice_1418 = None
    slice_1424: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_270, 0, 0, 9223372036854775807)
    slice_1425: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1424, 1, 1, 9223372036854775807);  slice_1424 = None
    slice_1426: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1425, 2, 0, 9223372036854775807);  slice_1425 = None
    slice_1427: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1426, 3, 0, 256);  slice_1426 = None
    copy_74: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1427, slice_1419);  slice_1427 = slice_1419 = None
    slice_1428: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_270, 0, 0, 9223372036854775807)
    slice_1429: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1428, 1, 1, 9223372036854775807)
    slice_1430: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1429, 2, 0, 9223372036854775807)
    slice_scatter_271: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1430, copy_74, 3, 0, 256);  slice_1430 = copy_74 = None
    slice_scatter_272: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1429, slice_scatter_271, 2, 0, 9223372036854775807);  slice_1429 = slice_scatter_271 = None
    slice_scatter_273: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1428, slice_scatter_272, 1, 1, 9223372036854775807);  slice_1428 = slice_scatter_272 = None
    slice_scatter_274: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_270, slice_scatter_273, 0, 0, 9223372036854775807);  slice_scatter_270 = slice_scatter_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_1435: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_472, 0, 0, 9223372036854775807);  view_472 = None
    select_125: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1435, 1, 0);  slice_1435 = None
    slice_1436: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_125, 1, 0, 255);  select_125 = None
    slice_1437: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1436, 2, -255, 9223372036854775807);  slice_1436 = None
    slice_1441: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_274, 0, 0, 9223372036854775807)
    select_127: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1441, 1, 0);  slice_1441 = None
    slice_1442: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_127, 1, 1, 256);  select_127 = None
    slice_1443: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1442, 2, 1, 256);  slice_1442 = None
    copy_75: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_1443, slice_1437);  slice_1443 = slice_1437 = None
    slice_1444: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_274, 0, 0, 9223372036854775807)
    select_128: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1444, 1, 0)
    slice_1445: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_128, 1, 1, 256)
    slice_scatter_275: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1445, copy_75, 2, 1, 256);  slice_1445 = copy_75 = None
    slice_scatter_276: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_128, slice_scatter_275, 1, 1, 256);  select_128 = slice_scatter_275 = None
    select_scatter_25: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1444, slice_scatter_276, 1, 0);  slice_1444 = slice_scatter_276 = None
    slice_scatter_277: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_274, select_scatter_25, 0, 0, 9223372036854775807);  slice_scatter_274 = select_scatter_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_55: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_24: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_117: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_24, -2);  iota_24 = None
    iota_25: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_118: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
    sub_49: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_117, unsqueeze_118);  unsqueeze_117 = unsqueeze_118 = None
    le_12: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_49, 0);  sub_49 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_48: "f32[256, 257]" = torch.ops.aten.where.self(le_12, full_55, scalar_tensor_24);  le_12 = full_55 = scalar_tensor_24 = None
    rev_24: "f32[256, 257]" = torch.ops.prims.rev.default(where_48, [0]);  where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_119: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_24, 0);  rev_24 = None
    slice_1449: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_119, 1, 0, 9223372036854775807);  unsqueeze_119 = None
    unsqueeze_120: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_1449, 2);  slice_1449 = None
    slice_1450: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_120, 3, 0, 9223372036854775807);  unsqueeze_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_25: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_1450, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_24: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_1450, [1, 256, 12, 257]);  slice_1450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_475: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_277, [1, 12, 1024, 513])
    permute_397: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
    slice_1455: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_397, 0, 0, 9223372036854775807);  permute_397 = None
    slice_1456: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1455, 1, 0, 256);  slice_1455 = None
    slice_1457: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1456, 2, 0, 9223372036854775807);  slice_1456 = None
    slice_1458: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1457, 3, 0, 257);  slice_1457 = None
    full_56: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_30: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_24, torch.bool);  expand_24 = None
    where_49: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_30, full_56, slice_1458);  convert_element_type_30 = full_56 = slice_1458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_476: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_277, [1, 12, 1024, 513])
    permute_398: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    slice_1463: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_398, 0, 0, 9223372036854775807);  permute_398 = None
    slice_1464: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1463, 1, 0, 256);  slice_1463 = None
    slice_1465: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1464, 2, 0, 9223372036854775807);  slice_1464 = None
    slice_1466: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1465, 3, 0, 257);  slice_1465 = None
    copy_76: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1466, where_49);  slice_1466 = where_49 = None
    view_477: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_277, [1, 12, 1024, 513]);  slice_scatter_277 = None
    permute_399: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
    slice_1467: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_399, 0, 0, 9223372036854775807)
    slice_1468: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1467, 1, 0, 256)
    slice_1469: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1468, 2, 0, 9223372036854775807)
    slice_scatter_278: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1469, copy_76, 3, 0, 257);  slice_1469 = copy_76 = None
    slice_scatter_279: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1468, slice_scatter_278, 2, 0, 9223372036854775807);  slice_1468 = slice_scatter_278 = None
    slice_scatter_280: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1467, slice_scatter_279, 1, 0, 256);  slice_1467 = slice_scatter_279 = None
    slice_scatter_281: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_399, slice_scatter_280, 0, 0, 9223372036854775807);  permute_399 = slice_scatter_280 = None
    permute_400: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_281, [0, 2, 1, 3]);  slice_scatter_281 = None
    view_478: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_400, [12, 4, 256, 513]);  permute_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_25: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_25, [1, 256, 12, 257]);  rev_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_480: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_478, [1, 12, 1024, 513])
    permute_402: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None
    slice_1478: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_402, 0, 0, 9223372036854775807);  permute_402 = None
    slice_1479: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1478, 1, -256, 9223372036854775807);  slice_1478 = None
    slice_1480: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1479, 2, 0, 9223372036854775807);  slice_1479 = None
    slice_1481: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1480, 3, -257, 9223372036854775807);  slice_1480 = None
    full_57: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_31: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_25, torch.bool);  expand_25 = None
    where_50: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_31, full_57, slice_1481);  convert_element_type_31 = full_57 = slice_1481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_481: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_478, [1, 12, 1024, 513])
    permute_403: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
    slice_1486: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_403, 0, 0, 9223372036854775807);  permute_403 = None
    slice_1487: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1486, 1, -256, 9223372036854775807);  slice_1486 = None
    slice_1488: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1487, 2, 0, 9223372036854775807);  slice_1487 = None
    slice_1489: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1488, 3, -257, 9223372036854775807);  slice_1488 = None
    copy_77: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1489, where_50);  slice_1489 = where_50 = None
    view_482: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_478, [1, 12, 1024, 513]);  view_478 = None
    permute_404: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
    slice_1490: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_404, 0, 0, 9223372036854775807)
    slice_1491: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1490, 1, -256, 9223372036854775807)
    slice_1492: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1491, 2, 0, 9223372036854775807)
    slice_scatter_282: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1492, copy_77, 3, -257, 9223372036854775807);  slice_1492 = copy_77 = None
    slice_scatter_283: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1491, slice_scatter_282, 2, 0, 9223372036854775807);  slice_1491 = slice_scatter_282 = None
    slice_scatter_284: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1490, slice_scatter_283, 1, -256, 9223372036854775807);  slice_1490 = slice_scatter_283 = None
    slice_scatter_285: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_404, slice_scatter_284, 0, 0, 9223372036854775807);  permute_404 = slice_scatter_284 = None
    permute_405: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_285, [0, 2, 1, 3]);  slice_scatter_285 = None
    view_483: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_405, [12, 4, 256, 513]);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_6: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_1497: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_6, 0, 0, 9223372036854775807);  ne_6 = None
    slice_1498: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_1497, 1, 0, 9223372036854775807);  slice_1497 = None
    unsqueeze_121: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_1498, 2);  slice_1498 = None
    unsqueeze_122: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_32: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_122, torch.float32)
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_51: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_122, scalar_tensor_25, convert_element_type_32);  unsqueeze_122 = scalar_tensor_25 = convert_element_type_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_58: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_407: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_58, [0, 2, 1, 3]);  full_58 = None
    view_485: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_407, [1, 1024, 1]);  permute_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_408: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_51, [0, 2, 1, 3]);  where_51 = None
    view_486: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_408, [1, 1024, 1]);  permute_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_487: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_485, [1, 2, 512, 1]);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_39: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_487, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_488: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_486, [1, 2, 512, 1]);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_40: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_488, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_123: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_39, 4);  as_strided_39 = None
    permute_409: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_123, [0, 1, 2, 4, 3]);  unsqueeze_123 = None
    unsqueeze_124: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_40, 4);  as_strided_40 = None
    permute_410: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_124, [0, 1, 4, 2, 3]);  unsqueeze_124 = None
    mul_48: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_409, permute_410);  permute_409 = permute_410 = None
    view_489: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_48, [1, 3, 512, 512]);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_25: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_489, [0, 0, 0, 1], 0.0);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_490: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_25, [1, 3, 512, 513]);  constant_pad_nd_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_59: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1499: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_490, 0, 0, 9223372036854775807)
    slice_1500: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1499, 1, 0, 9223372036854775807);  slice_1499 = None
    slice_1501: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1500, 2, 0, 256);  slice_1500 = None
    slice_1502: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1501, 3, 0, 257);  slice_1501 = None
    slice_1503: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_59, 0, 0, 9223372036854775807)
    slice_1504: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1503, 1, 0, -1);  slice_1503 = None
    slice_1505: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1504, 2, 0, 9223372036854775807);  slice_1504 = None
    slice_1506: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1505, 3, 256, 9223372036854775807);  slice_1505 = None
    copy_78: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1506, slice_1502);  slice_1506 = slice_1502 = None
    slice_1507: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_59, 0, 0, 9223372036854775807)
    slice_1508: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1507, 1, 0, -1)
    slice_1509: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1508, 2, 0, 9223372036854775807)
    slice_scatter_286: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1509, copy_78, 3, 256, 9223372036854775807);  slice_1509 = copy_78 = None
    slice_scatter_287: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1508, slice_scatter_286, 2, 0, 9223372036854775807);  slice_1508 = slice_scatter_286 = None
    slice_scatter_288: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1507, slice_scatter_287, 1, 0, -1);  slice_1507 = slice_scatter_287 = None
    slice_scatter_289: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_59, slice_scatter_288, 0, 0, 9223372036854775807);  full_59 = slice_scatter_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1514: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_490, 0, 0, 9223372036854775807)
    select_130: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_1514, 1, -1);  slice_1514 = None
    slice_1515: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_130, 1, 256, 9223372036854775807);  select_130 = None
    slice_1516: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1515, 2, 0, 257);  slice_1515 = None
    slice_1520: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_289, 0, 0, 9223372036854775807)
    select_132: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1520, 1, -1);  slice_1520 = None
    slice_1521: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_132, 1, 0, 9223372036854775807);  select_132 = None
    slice_1522: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1521, 2, 256, 9223372036854775807);  slice_1521 = None
    copy_79: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_1522, slice_1516);  slice_1522 = slice_1516 = None
    slice_1523: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_289, 0, 0, 9223372036854775807)
    select_133: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1523, 1, -1)
    slice_1524: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_133, 1, 0, 9223372036854775807)
    slice_scatter_290: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1524, copy_79, 2, 256, 9223372036854775807);  slice_1524 = copy_79 = None
    slice_scatter_291: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_133, slice_scatter_290, 1, 0, 9223372036854775807);  select_133 = slice_scatter_290 = None
    select_scatter_26: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1523, slice_scatter_291, 1, -1);  slice_1523 = slice_scatter_291 = None
    slice_scatter_292: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_289, select_scatter_26, 0, 0, 9223372036854775807);  slice_scatter_289 = select_scatter_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1528: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_490, 0, 0, 9223372036854775807)
    slice_1529: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1528, 1, 0, 9223372036854775807);  slice_1528 = None
    slice_1530: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1529, 2, -257, -1);  slice_1529 = None
    slice_1531: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1530, 3, 257, 9223372036854775807);  slice_1530 = None
    slice_1536: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_292, 0, 0, 9223372036854775807)
    slice_1537: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1536, 1, 1, 9223372036854775807);  slice_1536 = None
    slice_1538: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1537, 2, 0, 9223372036854775807);  slice_1537 = None
    slice_1539: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1538, 3, 0, 256);  slice_1538 = None
    copy_80: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1539, slice_1531);  slice_1539 = slice_1531 = None
    slice_1540: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_292, 0, 0, 9223372036854775807)
    slice_1541: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1540, 1, 1, 9223372036854775807)
    slice_1542: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1541, 2, 0, 9223372036854775807)
    slice_scatter_293: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1542, copy_80, 3, 0, 256);  slice_1542 = copy_80 = None
    slice_scatter_294: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1541, slice_scatter_293, 2, 0, 9223372036854775807);  slice_1541 = slice_scatter_293 = None
    slice_scatter_295: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1540, slice_scatter_294, 1, 1, 9223372036854775807);  slice_1540 = slice_scatter_294 = None
    slice_scatter_296: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_292, slice_scatter_295, 0, 0, 9223372036854775807);  slice_scatter_292 = slice_scatter_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_1547: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_490, 0, 0, 9223372036854775807);  view_490 = None
    select_135: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_1547, 1, 0);  slice_1547 = None
    slice_1548: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_135, 1, 0, 255);  select_135 = None
    slice_1549: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1548, 2, -255, 9223372036854775807);  slice_1548 = None
    slice_1553: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_296, 0, 0, 9223372036854775807)
    select_137: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1553, 1, 0);  slice_1553 = None
    slice_1554: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_137, 1, 1, 256);  select_137 = None
    slice_1555: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1554, 2, 1, 256);  slice_1554 = None
    copy_81: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_1555, slice_1549);  slice_1555 = slice_1549 = None
    slice_1556: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_296, 0, 0, 9223372036854775807)
    select_138: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1556, 1, 0)
    slice_1557: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_138, 1, 1, 256)
    slice_scatter_297: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1557, copy_81, 2, 1, 256);  slice_1557 = copy_81 = None
    slice_scatter_298: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_138, slice_scatter_297, 1, 1, 256);  select_138 = slice_scatter_297 = None
    select_scatter_27: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1556, slice_scatter_298, 1, 0);  slice_1556 = slice_scatter_298 = None
    slice_scatter_299: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_296, select_scatter_27, 0, 0, 9223372036854775807);  slice_scatter_296 = select_scatter_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_60: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_26: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_125: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_26, -2);  iota_26 = None
    iota_27: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_126: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
    sub_51: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_125, unsqueeze_126);  unsqueeze_125 = unsqueeze_126 = None
    le_13: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_51, 0);  sub_51 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_52: "f32[256, 257]" = torch.ops.aten.where.self(le_13, full_60, scalar_tensor_26);  le_13 = full_60 = scalar_tensor_26 = None
    rev_26: "f32[256, 257]" = torch.ops.prims.rev.default(where_52, [0]);  where_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_127: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_26, 0);  rev_26 = None
    slice_1561: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_127, 1, 0, 9223372036854775807);  unsqueeze_127 = None
    unsqueeze_128: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_1561, 2);  slice_1561 = None
    slice_1562: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_128, 3, 0, 9223372036854775807);  unsqueeze_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_27: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_1562, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_26: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_1562, [1, 256, 1, 257]);  slice_1562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_493: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_299, [1, 1, 1024, 513])
    permute_413: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    slice_1567: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_413, 0, 0, 9223372036854775807);  permute_413 = None
    slice_1568: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1567, 1, 0, 256);  slice_1567 = None
    slice_1569: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1568, 2, 0, 9223372036854775807);  slice_1568 = None
    slice_1570: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1569, 3, 0, 257);  slice_1569 = None
    full_61: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_33: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_26, torch.bool);  expand_26 = None
    where_53: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_33, full_61, slice_1570);  convert_element_type_33 = full_61 = slice_1570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_494: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_299, [1, 1, 1024, 513])
    permute_414: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    slice_1575: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_414, 0, 0, 9223372036854775807);  permute_414 = None
    slice_1576: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1575, 1, 0, 256);  slice_1575 = None
    slice_1577: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1576, 2, 0, 9223372036854775807);  slice_1576 = None
    slice_1578: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1577, 3, 0, 257);  slice_1577 = None
    copy_82: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1578, where_53);  slice_1578 = where_53 = None
    view_495: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_299, [1, 1, 1024, 513]);  slice_scatter_299 = None
    permute_415: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None
    slice_1579: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_415, 0, 0, 9223372036854775807)
    slice_1580: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1579, 1, 0, 256)
    slice_1581: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1580, 2, 0, 9223372036854775807)
    slice_scatter_300: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1581, copy_82, 3, 0, 257);  slice_1581 = copy_82 = None
    slice_scatter_301: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1580, slice_scatter_300, 2, 0, 9223372036854775807);  slice_1580 = slice_scatter_300 = None
    slice_scatter_302: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1579, slice_scatter_301, 1, 0, 256);  slice_1579 = slice_scatter_301 = None
    slice_scatter_303: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_415, slice_scatter_302, 0, 0, 9223372036854775807);  permute_415 = slice_scatter_302 = None
    permute_416: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_303, [0, 2, 1, 3]);  slice_scatter_303 = None
    view_496: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_416, [1, 4, 256, 513]);  permute_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_27: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_27, [1, 256, 1, 257]);  rev_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_498: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_496, [1, 1, 1024, 513])
    permute_418: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
    slice_1590: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_418, 0, 0, 9223372036854775807);  permute_418 = None
    slice_1591: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1590, 1, -256, 9223372036854775807);  slice_1590 = None
    slice_1592: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1591, 2, 0, 9223372036854775807);  slice_1591 = None
    slice_1593: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1592, 3, -257, 9223372036854775807);  slice_1592 = None
    full_62: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_34: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_27, torch.bool);  expand_27 = None
    where_54: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_34, full_62, slice_1593);  convert_element_type_34 = full_62 = slice_1593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_499: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_496, [1, 1, 1024, 513])
    permute_419: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
    slice_1598: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_419, 0, 0, 9223372036854775807);  permute_419 = None
    slice_1599: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1598, 1, -256, 9223372036854775807);  slice_1598 = None
    slice_1600: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1599, 2, 0, 9223372036854775807);  slice_1599 = None
    slice_1601: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1600, 3, -257, 9223372036854775807);  slice_1600 = None
    copy_83: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1601, where_54);  slice_1601 = where_54 = None
    view_500: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_496, [1, 1, 1024, 513]);  view_496 = None
    permute_420: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
    slice_1602: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_420, 0, 0, 9223372036854775807)
    slice_1603: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1602, 1, -256, 9223372036854775807)
    slice_1604: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1603, 2, 0, 9223372036854775807)
    slice_scatter_304: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1604, copy_83, 3, -257, 9223372036854775807);  slice_1604 = copy_83 = None
    slice_scatter_305: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1603, slice_scatter_304, 2, 0, 9223372036854775807);  slice_1603 = slice_scatter_304 = None
    slice_scatter_306: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1602, slice_scatter_305, 1, -256, 9223372036854775807);  slice_1602 = slice_scatter_305 = None
    slice_scatter_307: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_420, slice_scatter_306, 0, 0, 9223372036854775807);  permute_420 = slice_scatter_306 = None
    permute_421: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_307, [0, 2, 1, 3]);  slice_scatter_307 = None
    view_501: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_421, [1, 4, 256, 513]);  permute_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_503: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_483, [1, 12, 1024, 513]);  view_483 = None
    permute_423: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_503, [0, 2, 1, 3]);  view_503 = None
    view_504: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_501, [1, 1, 1024, 513]);  view_501 = None
    permute_424: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
    add_68: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_423, permute_424);  permute_423 = permute_424 = None
    permute_425: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_68, [0, 2, 1, 3]);  add_68 = None
    view_506: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_425, [12, 4, 256, 513]);  permute_425 = None
    view_507: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_506, [1, 12, 1024, 513]);  view_506 = None
    permute_426: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_50: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    amax_6: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_50, [-1], True)
    sub_52: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_50, amax_6);  clone_50 = amax_6 = None
    exp_6: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_7: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_67: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_1609: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_1610: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_1609, 1, 0, 9223372036854775807);  slice_1609 = None
    unsqueeze_129: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_1610, 2);  slice_1610 = None
    unsqueeze_130: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_55: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_130, scalar_tensor_27, div_67);  unsqueeze_130 = scalar_tensor_27 = div_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_51: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_55);  where_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_508: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_455, [1024, 1, 12, 64]);  view_455 = None
    permute_427: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_508, [1, 0, 2, 3]);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_428: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_51, [0, 2, 1, 3]);  clone_51 = None
    view_509: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_428, [12, 4, 256, 513]);  permute_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_429: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_427, [0, 2, 1, 3]);  permute_427 = None
    view_510: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_429, [12, 1024, 64]);  permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_26: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_510, [0, 0, 256, 256], -1.0);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_41: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_26, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_27: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_509, [0, 257], 0.0);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_511: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_27, [12, 4, -1]);  constant_pad_nd_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_1611: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_511, 0, 0, 9223372036854775807);  view_511 = None
    slice_1612: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_1611, 1, 0, 9223372036854775807);  slice_1611 = None
    slice_1613: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_1612, 2, 0, -256);  slice_1612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_512: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_1613, [12, 4, 256, 769]);  slice_1613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_1614: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_512, 0, 0, 9223372036854775807);  view_512 = None
    slice_1615: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1614, 1, 0, 9223372036854775807);  slice_1614 = None
    slice_1616: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1615, 2, 0, 9223372036854775807);  slice_1615 = None
    slice_1617: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_1616, 3, 0, -1);  slice_1616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_131: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1617, 4);  slice_1617 = None
    permute_430: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_131, [0, 1, 2, 4, 3]);  unsqueeze_131 = None
    unsqueeze_132: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_41, 4);  as_strided_41 = None
    permute_431: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_132, [0, 1, 4, 3, 2]);  unsqueeze_132 = None
    permute_432: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_430, [0, 1, 2, 4, 3]);  permute_430 = None
    view_513: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_432, [48, 256, 768]);  permute_432 = None
    permute_433: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_431, [0, 1, 4, 3, 2]);  permute_431 = None
    clone_52: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    view_514: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_52, [48, 768, 64]);  clone_52 = None
    bmm_13: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_513, view_514);  view_513 = view_514 = None
    view_515: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_13, [12, 4, 256, 1, 64]);  bmm_13 = None
    permute_434: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_515, [0, 1, 2, 4, 3]);  view_515 = None
    view_516: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_434, [12, 4, 256, 64]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_517: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_516, [1, 12, 1024, 64]);  view_516 = None
    permute_435: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_436: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_435, [1, 0, 2, 3]);  permute_435 = None
    clone_53: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
    view_518: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_53, [1024, 1, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_437: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_518, [1, 0, 2]);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_519: "f32[1024, 768]" = torch.ops.aten.view.default(permute_437, [1024, 768]);  permute_437 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_39: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg103_1, view_519, permute_438);  arg103_1 = view_519 = permute_438 = None
    view_520: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_39, [1, 1024, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_54: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_520);  view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_70: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_54, add_65);  clone_54 = add_65 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    add_71: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_54: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_25);  add_70 = getitem_25 = None
    mul_49: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_12);  sub_54 = rsqrt_12 = None
    mul_50: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg104_1);  mul_49 = arg104_1 = None
    add_72: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_50, arg105_1);  mul_50 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_521: "f32[1024, 768]" = torch.ops.aten.view.default(add_72, [1024, 768])
    permute_439: "f32[768, 3072]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_40: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg107_1, view_521, permute_439);  arg107_1 = view_521 = permute_439 = None
    view_522: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 1024, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_51: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_522, 0.5)
    mul_52: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_522, 0.7071067811865476);  view_522 = None
    erf_6: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
    add_73: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_53: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_73);  mul_51 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_523: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_53, [1024, 3072]);  mul_53 = None
    permute_440: "f32[3072, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_41: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg109_1, view_523, permute_440);  arg109_1 = view_523 = permute_440 = None
    view_524: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_41, [1, 1024, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_55: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_524);  view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_74: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_55, add_72);  clone_55 = add_72 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 1024, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    add_75: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_55: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_27);  add_74 = getitem_27 = None
    mul_54: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_13);  sub_55 = rsqrt_13 = None
    mul_55: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg110_1);  mul_54 = arg110_1 = None
    add_76: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_55, arg111_1);  mul_55 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_441: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_76, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_525: "f32[1024, 768]" = torch.ops.aten.view.default(permute_441, [1024, 768])
    permute_442: "f32[768, 768]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_42: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg113_1, view_525, permute_442);  arg113_1 = view_525 = permute_442 = None
    view_526: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_42, [1024, 1, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_527: "f32[1024, 768]" = torch.ops.aten.view.default(permute_441, [1024, 768])
    permute_443: "f32[768, 768]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_43: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg115_1, view_527, permute_443);  arg115_1 = view_527 = permute_443 = None
    view_528: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_43, [1024, 1, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_529: "f32[1024, 768]" = torch.ops.aten.view.default(permute_441, [1024, 768]);  permute_441 = None
    permute_444: "f32[768, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_44: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg117_1, view_529, permute_444);  arg117_1 = view_529 = permute_444 = None
    view_530: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_44, [1024, 1, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_70: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_526, 8.0);  view_526 = None
    view_531: "f32[1024, 768]" = torch.ops.aten.view.default(div_70, [1024, 768]);  div_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_534: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_528, [1024, 1, 12, 64]);  view_528 = None
    permute_446: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_534, [1, 0, 2, 3]);  view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_448: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_446, [0, 2, 1, 3]);  permute_446 = None
    view_536: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_448, [12, 1024, 64]);  permute_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_538: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_536, [12, 2, 512, 64]);  view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_43: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_538, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_134: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_43, 4);  as_strided_43 = None
    permute_450: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_134, [0, 1, 4, 2, 3]);  unsqueeze_134 = None
    view_539: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_531, [1024, 1, 768]);  view_531 = None
    view_540: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_539, [1024, 1, 12, 64]);  view_539 = None
    permute_452: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_540, [1, 0, 2, 3]);  view_540 = None
    permute_453: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_452, [0, 2, 1, 3]);  permute_452 = None
    view_541: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_453, [12, 1024, 64]);  permute_453 = None
    view_542: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_541, [12, 2, 512, 64]);  view_541 = None
    as_strided_44: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_542, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_542 = None
    unsqueeze_135: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_44, 4);  as_strided_44 = None
    permute_454: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_135, [0, 1, 2, 4, 3]);  unsqueeze_135 = None
    permute_455: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_454, [0, 1, 2, 4, 3]);  permute_454 = None
    clone_56: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
    view_543: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_56, [36, 512, 64]);  clone_56 = None
    permute_456: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_450, [0, 1, 4, 3, 2]);  permute_450 = None
    clone_57: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_456, memory_format = torch.contiguous_format);  permute_456 = None
    view_544: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_57, [36, 64, 512]);  clone_57 = None
    bmm_14: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_543, view_544);  view_543 = view_544 = None
    view_545: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_14, [12, 3, 512, 1, 512]);  bmm_14 = None
    permute_457: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_545, [0, 1, 2, 4, 3]);  view_545 = None
    view_546: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_457, [12, 3, 512, 512]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_28: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_546, [0, 0, 0, 1], 0.0);  view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_547: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_28, [12, 3, 512, 513]);  constant_pad_nd_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_63: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1618: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_547, 0, 0, 9223372036854775807)
    slice_1619: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1618, 1, 0, 9223372036854775807);  slice_1618 = None
    slice_1620: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1619, 2, 0, 256);  slice_1619 = None
    slice_1621: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1620, 3, 0, 257);  slice_1620 = None
    slice_1622: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_63, 0, 0, 9223372036854775807)
    slice_1623: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1622, 1, 0, -1);  slice_1622 = None
    slice_1624: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1623, 2, 0, 9223372036854775807);  slice_1623 = None
    slice_1625: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1624, 3, 256, 9223372036854775807);  slice_1624 = None
    copy_84: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1625, slice_1621);  slice_1625 = slice_1621 = None
    slice_1626: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_63, 0, 0, 9223372036854775807)
    slice_1627: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1626, 1, 0, -1)
    slice_1628: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1627, 2, 0, 9223372036854775807)
    slice_scatter_308: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1628, copy_84, 3, 256, 9223372036854775807);  slice_1628 = copy_84 = None
    slice_scatter_309: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1627, slice_scatter_308, 2, 0, 9223372036854775807);  slice_1627 = slice_scatter_308 = None
    slice_scatter_310: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1626, slice_scatter_309, 1, 0, -1);  slice_1626 = slice_scatter_309 = None
    slice_scatter_311: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_63, slice_scatter_310, 0, 0, 9223372036854775807);  full_63 = slice_scatter_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1633: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_547, 0, 0, 9223372036854775807)
    select_140: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1633, 1, -1);  slice_1633 = None
    slice_1634: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_140, 1, 256, 9223372036854775807);  select_140 = None
    slice_1635: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1634, 2, 0, 257);  slice_1634 = None
    slice_1639: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_311, 0, 0, 9223372036854775807)
    select_142: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1639, 1, -1);  slice_1639 = None
    slice_1640: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_142, 1, 0, 9223372036854775807);  select_142 = None
    slice_1641: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1640, 2, 256, 9223372036854775807);  slice_1640 = None
    copy_85: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_1641, slice_1635);  slice_1641 = slice_1635 = None
    slice_1642: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_311, 0, 0, 9223372036854775807)
    select_143: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1642, 1, -1)
    slice_1643: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_143, 1, 0, 9223372036854775807)
    slice_scatter_312: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1643, copy_85, 2, 256, 9223372036854775807);  slice_1643 = copy_85 = None
    slice_scatter_313: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_143, slice_scatter_312, 1, 0, 9223372036854775807);  select_143 = slice_scatter_312 = None
    select_scatter_28: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1642, slice_scatter_313, 1, -1);  slice_1642 = slice_scatter_313 = None
    slice_scatter_314: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_311, select_scatter_28, 0, 0, 9223372036854775807);  slice_scatter_311 = select_scatter_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1647: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_547, 0, 0, 9223372036854775807)
    slice_1648: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1647, 1, 0, 9223372036854775807);  slice_1647 = None
    slice_1649: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1648, 2, -257, -1);  slice_1648 = None
    slice_1650: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1649, 3, 257, 9223372036854775807);  slice_1649 = None
    slice_1655: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_314, 0, 0, 9223372036854775807)
    slice_1656: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1655, 1, 1, 9223372036854775807);  slice_1655 = None
    slice_1657: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1656, 2, 0, 9223372036854775807);  slice_1656 = None
    slice_1658: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1657, 3, 0, 256);  slice_1657 = None
    copy_86: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1658, slice_1650);  slice_1658 = slice_1650 = None
    slice_1659: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_314, 0, 0, 9223372036854775807)
    slice_1660: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1659, 1, 1, 9223372036854775807)
    slice_1661: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1660, 2, 0, 9223372036854775807)
    slice_scatter_315: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1661, copy_86, 3, 0, 256);  slice_1661 = copy_86 = None
    slice_scatter_316: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1660, slice_scatter_315, 2, 0, 9223372036854775807);  slice_1660 = slice_scatter_315 = None
    slice_scatter_317: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1659, slice_scatter_316, 1, 1, 9223372036854775807);  slice_1659 = slice_scatter_316 = None
    slice_scatter_318: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_314, slice_scatter_317, 0, 0, 9223372036854775807);  slice_scatter_314 = slice_scatter_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_1666: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_547, 0, 0, 9223372036854775807);  view_547 = None
    select_145: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1666, 1, 0);  slice_1666 = None
    slice_1667: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_145, 1, 0, 255);  select_145 = None
    slice_1668: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1667, 2, -255, 9223372036854775807);  slice_1667 = None
    slice_1672: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_318, 0, 0, 9223372036854775807)
    select_147: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1672, 1, 0);  slice_1672 = None
    slice_1673: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_147, 1, 1, 256);  select_147 = None
    slice_1674: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1673, 2, 1, 256);  slice_1673 = None
    copy_87: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_1674, slice_1668);  slice_1674 = slice_1668 = None
    slice_1675: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_318, 0, 0, 9223372036854775807)
    select_148: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1675, 1, 0)
    slice_1676: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_148, 1, 1, 256)
    slice_scatter_319: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1676, copy_87, 2, 1, 256);  slice_1676 = copy_87 = None
    slice_scatter_320: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_148, slice_scatter_319, 1, 1, 256);  select_148 = slice_scatter_319 = None
    select_scatter_29: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1675, slice_scatter_320, 1, 0);  slice_1675 = slice_scatter_320 = None
    slice_scatter_321: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_318, select_scatter_29, 0, 0, 9223372036854775807);  slice_scatter_318 = select_scatter_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_64: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_28: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_136: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_28, -2);  iota_28 = None
    iota_29: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_137: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
    sub_57: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_136, unsqueeze_137);  unsqueeze_136 = unsqueeze_137 = None
    le_14: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_57, 0);  sub_57 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_56: "f32[256, 257]" = torch.ops.aten.where.self(le_14, full_64, scalar_tensor_28);  le_14 = full_64 = scalar_tensor_28 = None
    rev_28: "f32[256, 257]" = torch.ops.prims.rev.default(where_56, [0]);  where_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_138: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_28, 0);  rev_28 = None
    slice_1680: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_138, 1, 0, 9223372036854775807);  unsqueeze_138 = None
    unsqueeze_139: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_1680, 2);  slice_1680 = None
    slice_1681: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_139, 3, 0, 9223372036854775807);  unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_29: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_1681, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_28: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_1681, [1, 256, 12, 257]);  slice_1681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_550: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_321, [1, 12, 1024, 513])
    permute_460: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
    slice_1686: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_460, 0, 0, 9223372036854775807);  permute_460 = None
    slice_1687: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1686, 1, 0, 256);  slice_1686 = None
    slice_1688: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1687, 2, 0, 9223372036854775807);  slice_1687 = None
    slice_1689: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1688, 3, 0, 257);  slice_1688 = None
    full_65: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_35: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_28, torch.bool);  expand_28 = None
    where_57: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_35, full_65, slice_1689);  convert_element_type_35 = full_65 = slice_1689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_551: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_321, [1, 12, 1024, 513])
    permute_461: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
    slice_1694: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_461, 0, 0, 9223372036854775807);  permute_461 = None
    slice_1695: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1694, 1, 0, 256);  slice_1694 = None
    slice_1696: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1695, 2, 0, 9223372036854775807);  slice_1695 = None
    slice_1697: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1696, 3, 0, 257);  slice_1696 = None
    copy_88: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1697, where_57);  slice_1697 = where_57 = None
    view_552: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_321, [1, 12, 1024, 513]);  slice_scatter_321 = None
    permute_462: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None
    slice_1698: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_462, 0, 0, 9223372036854775807)
    slice_1699: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1698, 1, 0, 256)
    slice_1700: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1699, 2, 0, 9223372036854775807)
    slice_scatter_322: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1700, copy_88, 3, 0, 257);  slice_1700 = copy_88 = None
    slice_scatter_323: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1699, slice_scatter_322, 2, 0, 9223372036854775807);  slice_1699 = slice_scatter_322 = None
    slice_scatter_324: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1698, slice_scatter_323, 1, 0, 256);  slice_1698 = slice_scatter_323 = None
    slice_scatter_325: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_462, slice_scatter_324, 0, 0, 9223372036854775807);  permute_462 = slice_scatter_324 = None
    permute_463: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_325, [0, 2, 1, 3]);  slice_scatter_325 = None
    view_553: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_463, [12, 4, 256, 513]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_29: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_29, [1, 256, 12, 257]);  rev_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_555: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_553, [1, 12, 1024, 513])
    permute_465: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
    slice_1709: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_465, 0, 0, 9223372036854775807);  permute_465 = None
    slice_1710: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1709, 1, -256, 9223372036854775807);  slice_1709 = None
    slice_1711: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1710, 2, 0, 9223372036854775807);  slice_1710 = None
    slice_1712: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1711, 3, -257, 9223372036854775807);  slice_1711 = None
    full_66: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_36: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_29, torch.bool);  expand_29 = None
    where_58: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_36, full_66, slice_1712);  convert_element_type_36 = full_66 = slice_1712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_556: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_553, [1, 12, 1024, 513])
    permute_466: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
    slice_1717: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_466, 0, 0, 9223372036854775807);  permute_466 = None
    slice_1718: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1717, 1, -256, 9223372036854775807);  slice_1717 = None
    slice_1719: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1718, 2, 0, 9223372036854775807);  slice_1718 = None
    slice_1720: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1719, 3, -257, 9223372036854775807);  slice_1719 = None
    copy_89: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1720, where_58);  slice_1720 = where_58 = None
    view_557: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_553, [1, 12, 1024, 513]);  view_553 = None
    permute_467: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None
    slice_1721: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_467, 0, 0, 9223372036854775807)
    slice_1722: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1721, 1, -256, 9223372036854775807)
    slice_1723: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1722, 2, 0, 9223372036854775807)
    slice_scatter_326: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1723, copy_89, 3, -257, 9223372036854775807);  slice_1723 = copy_89 = None
    slice_scatter_327: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1722, slice_scatter_326, 2, 0, 9223372036854775807);  slice_1722 = slice_scatter_326 = None
    slice_scatter_328: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1721, slice_scatter_327, 1, -256, 9223372036854775807);  slice_1721 = slice_scatter_327 = None
    slice_scatter_329: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_467, slice_scatter_328, 0, 0, 9223372036854775807);  permute_467 = slice_scatter_328 = None
    permute_468: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_329, [0, 2, 1, 3]);  slice_scatter_329 = None
    view_558: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_468, [12, 4, 256, 513]);  permute_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_7: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_1728: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_7, 0, 0, 9223372036854775807);  ne_7 = None
    slice_1729: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_1728, 1, 0, 9223372036854775807);  slice_1728 = None
    unsqueeze_140: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_1729, 2);  slice_1729 = None
    unsqueeze_141: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_37: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_141, torch.float32)
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_59: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_141, scalar_tensor_29, convert_element_type_37);  unsqueeze_141 = scalar_tensor_29 = convert_element_type_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_67: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_470: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_67, [0, 2, 1, 3]);  full_67 = None
    view_560: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_470, [1, 1024, 1]);  permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_471: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_59, [0, 2, 1, 3]);  where_59 = None
    view_561: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_471, [1, 1024, 1]);  permute_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_562: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_560, [1, 2, 512, 1]);  view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_45: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_562, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_563: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_561, [1, 2, 512, 1]);  view_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_46: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_563, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_142: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_45, 4);  as_strided_45 = None
    permute_472: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_142, [0, 1, 2, 4, 3]);  unsqueeze_142 = None
    unsqueeze_143: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_46, 4);  as_strided_46 = None
    permute_473: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_143, [0, 1, 4, 2, 3]);  unsqueeze_143 = None
    mul_56: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_472, permute_473);  permute_472 = permute_473 = None
    view_564: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_56, [1, 3, 512, 512]);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_29: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_564, [0, 0, 0, 1], 0.0);  view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_565: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_29, [1, 3, 512, 513]);  constant_pad_nd_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_68: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1730: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_565, 0, 0, 9223372036854775807)
    slice_1731: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1730, 1, 0, 9223372036854775807);  slice_1730 = None
    slice_1732: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1731, 2, 0, 256);  slice_1731 = None
    slice_1733: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1732, 3, 0, 257);  slice_1732 = None
    slice_1734: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_68, 0, 0, 9223372036854775807)
    slice_1735: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1734, 1, 0, -1);  slice_1734 = None
    slice_1736: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1735, 2, 0, 9223372036854775807);  slice_1735 = None
    slice_1737: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1736, 3, 256, 9223372036854775807);  slice_1736 = None
    copy_90: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1737, slice_1733);  slice_1737 = slice_1733 = None
    slice_1738: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_68, 0, 0, 9223372036854775807)
    slice_1739: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1738, 1, 0, -1)
    slice_1740: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1739, 2, 0, 9223372036854775807)
    slice_scatter_330: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1740, copy_90, 3, 256, 9223372036854775807);  slice_1740 = copy_90 = None
    slice_scatter_331: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1739, slice_scatter_330, 2, 0, 9223372036854775807);  slice_1739 = slice_scatter_330 = None
    slice_scatter_332: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1738, slice_scatter_331, 1, 0, -1);  slice_1738 = slice_scatter_331 = None
    slice_scatter_333: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_68, slice_scatter_332, 0, 0, 9223372036854775807);  full_68 = slice_scatter_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1745: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_565, 0, 0, 9223372036854775807)
    select_150: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_1745, 1, -1);  slice_1745 = None
    slice_1746: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_150, 1, 256, 9223372036854775807);  select_150 = None
    slice_1747: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1746, 2, 0, 257);  slice_1746 = None
    slice_1751: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_333, 0, 0, 9223372036854775807)
    select_152: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1751, 1, -1);  slice_1751 = None
    slice_1752: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_152, 1, 0, 9223372036854775807);  select_152 = None
    slice_1753: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1752, 2, 256, 9223372036854775807);  slice_1752 = None
    copy_91: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_1753, slice_1747);  slice_1753 = slice_1747 = None
    slice_1754: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_333, 0, 0, 9223372036854775807)
    select_153: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1754, 1, -1)
    slice_1755: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_153, 1, 0, 9223372036854775807)
    slice_scatter_334: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1755, copy_91, 2, 256, 9223372036854775807);  slice_1755 = copy_91 = None
    slice_scatter_335: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_153, slice_scatter_334, 1, 0, 9223372036854775807);  select_153 = slice_scatter_334 = None
    select_scatter_30: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1754, slice_scatter_335, 1, -1);  slice_1754 = slice_scatter_335 = None
    slice_scatter_336: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_333, select_scatter_30, 0, 0, 9223372036854775807);  slice_scatter_333 = select_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1759: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_565, 0, 0, 9223372036854775807)
    slice_1760: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1759, 1, 0, 9223372036854775807);  slice_1759 = None
    slice_1761: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1760, 2, -257, -1);  slice_1760 = None
    slice_1762: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1761, 3, 257, 9223372036854775807);  slice_1761 = None
    slice_1767: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_336, 0, 0, 9223372036854775807)
    slice_1768: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1767, 1, 1, 9223372036854775807);  slice_1767 = None
    slice_1769: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1768, 2, 0, 9223372036854775807);  slice_1768 = None
    slice_1770: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1769, 3, 0, 256);  slice_1769 = None
    copy_92: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1770, slice_1762);  slice_1770 = slice_1762 = None
    slice_1771: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_336, 0, 0, 9223372036854775807)
    slice_1772: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1771, 1, 1, 9223372036854775807)
    slice_1773: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1772, 2, 0, 9223372036854775807)
    slice_scatter_337: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1773, copy_92, 3, 0, 256);  slice_1773 = copy_92 = None
    slice_scatter_338: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1772, slice_scatter_337, 2, 0, 9223372036854775807);  slice_1772 = slice_scatter_337 = None
    slice_scatter_339: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1771, slice_scatter_338, 1, 1, 9223372036854775807);  slice_1771 = slice_scatter_338 = None
    slice_scatter_340: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_336, slice_scatter_339, 0, 0, 9223372036854775807);  slice_scatter_336 = slice_scatter_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_1778: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_565, 0, 0, 9223372036854775807);  view_565 = None
    select_155: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_1778, 1, 0);  slice_1778 = None
    slice_1779: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_155, 1, 0, 255);  select_155 = None
    slice_1780: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1779, 2, -255, 9223372036854775807);  slice_1779 = None
    slice_1784: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_340, 0, 0, 9223372036854775807)
    select_157: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1784, 1, 0);  slice_1784 = None
    slice_1785: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_157, 1, 1, 256);  select_157 = None
    slice_1786: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1785, 2, 1, 256);  slice_1785 = None
    copy_93: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_1786, slice_1780);  slice_1786 = slice_1780 = None
    slice_1787: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_340, 0, 0, 9223372036854775807)
    select_158: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1787, 1, 0)
    slice_1788: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_158, 1, 1, 256)
    slice_scatter_341: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1788, copy_93, 2, 1, 256);  slice_1788 = copy_93 = None
    slice_scatter_342: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_158, slice_scatter_341, 1, 1, 256);  select_158 = slice_scatter_341 = None
    select_scatter_31: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1787, slice_scatter_342, 1, 0);  slice_1787 = slice_scatter_342 = None
    slice_scatter_343: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_340, select_scatter_31, 0, 0, 9223372036854775807);  slice_scatter_340 = select_scatter_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_69: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_30: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_144: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_30, -2);  iota_30 = None
    iota_31: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_145: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
    sub_59: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_144, unsqueeze_145);  unsqueeze_144 = unsqueeze_145 = None
    le_15: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_59, 0);  sub_59 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_60: "f32[256, 257]" = torch.ops.aten.where.self(le_15, full_69, scalar_tensor_30);  le_15 = full_69 = scalar_tensor_30 = None
    rev_30: "f32[256, 257]" = torch.ops.prims.rev.default(where_60, [0]);  where_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_146: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_30, 0);  rev_30 = None
    slice_1792: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_146, 1, 0, 9223372036854775807);  unsqueeze_146 = None
    unsqueeze_147: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_1792, 2);  slice_1792 = None
    slice_1793: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_147, 3, 0, 9223372036854775807);  unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_31: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_1793, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_30: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_1793, [1, 256, 1, 257]);  slice_1793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_568: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_343, [1, 1, 1024, 513])
    permute_476: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
    slice_1798: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_476, 0, 0, 9223372036854775807);  permute_476 = None
    slice_1799: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1798, 1, 0, 256);  slice_1798 = None
    slice_1800: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1799, 2, 0, 9223372036854775807);  slice_1799 = None
    slice_1801: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1800, 3, 0, 257);  slice_1800 = None
    full_70: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_38: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_30, torch.bool);  expand_30 = None
    where_61: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_38, full_70, slice_1801);  convert_element_type_38 = full_70 = slice_1801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_569: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_343, [1, 1, 1024, 513])
    permute_477: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None
    slice_1806: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_477, 0, 0, 9223372036854775807);  permute_477 = None
    slice_1807: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1806, 1, 0, 256);  slice_1806 = None
    slice_1808: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1807, 2, 0, 9223372036854775807);  slice_1807 = None
    slice_1809: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1808, 3, 0, 257);  slice_1808 = None
    copy_94: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1809, where_61);  slice_1809 = where_61 = None
    view_570: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_343, [1, 1, 1024, 513]);  slice_scatter_343 = None
    permute_478: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    slice_1810: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_478, 0, 0, 9223372036854775807)
    slice_1811: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1810, 1, 0, 256)
    slice_1812: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1811, 2, 0, 9223372036854775807)
    slice_scatter_344: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1812, copy_94, 3, 0, 257);  slice_1812 = copy_94 = None
    slice_scatter_345: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1811, slice_scatter_344, 2, 0, 9223372036854775807);  slice_1811 = slice_scatter_344 = None
    slice_scatter_346: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1810, slice_scatter_345, 1, 0, 256);  slice_1810 = slice_scatter_345 = None
    slice_scatter_347: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_478, slice_scatter_346, 0, 0, 9223372036854775807);  permute_478 = slice_scatter_346 = None
    permute_479: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_347, [0, 2, 1, 3]);  slice_scatter_347 = None
    view_571: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_479, [1, 4, 256, 513]);  permute_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_31: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_31, [1, 256, 1, 257]);  rev_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_573: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_571, [1, 1, 1024, 513])
    permute_481: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    slice_1821: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_481, 0, 0, 9223372036854775807);  permute_481 = None
    slice_1822: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1821, 1, -256, 9223372036854775807);  slice_1821 = None
    slice_1823: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1822, 2, 0, 9223372036854775807);  slice_1822 = None
    slice_1824: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1823, 3, -257, 9223372036854775807);  slice_1823 = None
    full_71: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_39: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_31, torch.bool);  expand_31 = None
    where_62: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_39, full_71, slice_1824);  convert_element_type_39 = full_71 = slice_1824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_574: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_571, [1, 1, 1024, 513])
    permute_482: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
    slice_1829: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_482, 0, 0, 9223372036854775807);  permute_482 = None
    slice_1830: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1829, 1, -256, 9223372036854775807);  slice_1829 = None
    slice_1831: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1830, 2, 0, 9223372036854775807);  slice_1830 = None
    slice_1832: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1831, 3, -257, 9223372036854775807);  slice_1831 = None
    copy_95: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1832, where_62);  slice_1832 = where_62 = None
    view_575: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_571, [1, 1, 1024, 513]);  view_571 = None
    permute_483: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_575, [0, 2, 1, 3]);  view_575 = None
    slice_1833: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_483, 0, 0, 9223372036854775807)
    slice_1834: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1833, 1, -256, 9223372036854775807)
    slice_1835: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_1834, 2, 0, 9223372036854775807)
    slice_scatter_348: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1835, copy_95, 3, -257, 9223372036854775807);  slice_1835 = copy_95 = None
    slice_scatter_349: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1834, slice_scatter_348, 2, 0, 9223372036854775807);  slice_1834 = slice_scatter_348 = None
    slice_scatter_350: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_1833, slice_scatter_349, 1, -256, 9223372036854775807);  slice_1833 = slice_scatter_349 = None
    slice_scatter_351: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_483, slice_scatter_350, 0, 0, 9223372036854775807);  permute_483 = slice_scatter_350 = None
    permute_484: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_351, [0, 2, 1, 3]);  slice_scatter_351 = None
    view_576: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_484, [1, 4, 256, 513]);  permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_578: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_558, [1, 12, 1024, 513]);  view_558 = None
    permute_486: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
    view_579: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_576, [1, 1, 1024, 513]);  view_576 = None
    permute_487: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_579, [0, 2, 1, 3]);  view_579 = None
    add_79: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_486, permute_487);  permute_486 = permute_487 = None
    permute_488: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_79, [0, 2, 1, 3]);  add_79 = None
    view_581: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_488, [12, 4, 256, 513]);  permute_488 = None
    view_582: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_581, [1, 12, 1024, 513]);  view_581 = None
    permute_489: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_58: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
    amax_7: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_58, [-1], True)
    sub_60: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_58, amax_7);  clone_58 = amax_7 = None
    exp_7: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_8: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_77: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_1840: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_1841: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_1840, 1, 0, 9223372036854775807);  slice_1840 = None
    unsqueeze_148: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_1841, 2);  slice_1841 = None
    unsqueeze_149: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_63: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_149, scalar_tensor_31, div_77);  unsqueeze_149 = scalar_tensor_31 = div_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_59: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_63);  where_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_583: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_530, [1024, 1, 12, 64]);  view_530 = None
    permute_490: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_583, [1, 0, 2, 3]);  view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_491: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_59, [0, 2, 1, 3]);  clone_59 = None
    view_584: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_491, [12, 4, 256, 513]);  permute_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_492: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
    view_585: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_492, [12, 1024, 64]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_30: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_585, [0, 0, 256, 256], -1.0);  view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_47: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_30, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_31: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_584, [0, 257], 0.0);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_586: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_31, [12, 4, -1]);  constant_pad_nd_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_1842: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_586, 0, 0, 9223372036854775807);  view_586 = None
    slice_1843: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_1842, 1, 0, 9223372036854775807);  slice_1842 = None
    slice_1844: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_1843, 2, 0, -256);  slice_1843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_587: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_1844, [12, 4, 256, 769]);  slice_1844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_1845: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_587, 0, 0, 9223372036854775807);  view_587 = None
    slice_1846: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1845, 1, 0, 9223372036854775807);  slice_1845 = None
    slice_1847: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_1846, 2, 0, 9223372036854775807);  slice_1846 = None
    slice_1848: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_1847, 3, 0, -1);  slice_1847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_150: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1848, 4);  slice_1848 = None
    permute_493: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_150, [0, 1, 2, 4, 3]);  unsqueeze_150 = None
    unsqueeze_151: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_47, 4);  as_strided_47 = None
    permute_494: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_151, [0, 1, 4, 3, 2]);  unsqueeze_151 = None
    permute_495: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_493, [0, 1, 2, 4, 3]);  permute_493 = None
    view_588: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_495, [48, 256, 768]);  permute_495 = None
    permute_496: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_494, [0, 1, 4, 3, 2]);  permute_494 = None
    clone_60: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
    view_589: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_60, [48, 768, 64]);  clone_60 = None
    bmm_15: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_588, view_589);  view_588 = view_589 = None
    view_590: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_15, [12, 4, 256, 1, 64]);  bmm_15 = None
    permute_497: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_590, [0, 1, 2, 4, 3]);  view_590 = None
    view_591: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_497, [12, 4, 256, 64]);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_592: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_591, [1, 12, 1024, 64]);  view_591 = None
    permute_498: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_499: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_498, [1, 0, 2, 3]);  permute_498 = None
    clone_61: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
    view_593: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_61, [1024, 1, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_500: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_593, [1, 0, 2]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_594: "f32[1024, 768]" = torch.ops.aten.view.default(permute_500, [1024, 768]);  permute_500 = None
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_45: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg119_1, view_594, permute_501);  arg119_1 = view_594 = permute_501 = None
    view_595: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_45, [1, 1024, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_62: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_595);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_62, add_76);  clone_62 = add_76 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 1024, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    add_82: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_62: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_29);  add_81 = getitem_29 = None
    mul_57: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = rsqrt_14 = None
    mul_58: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg120_1);  mul_57 = arg120_1 = None
    add_83: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_58, arg121_1);  mul_58 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_596: "f32[1024, 768]" = torch.ops.aten.view.default(add_83, [1024, 768])
    permute_502: "f32[768, 3072]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_46: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg123_1, view_596, permute_502);  arg123_1 = view_596 = permute_502 = None
    view_597: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 1024, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_59: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_597, 0.5)
    mul_60: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_597, 0.7071067811865476);  view_597 = None
    erf_7: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_84: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_61: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_59, add_84);  mul_59 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_598: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_61, [1024, 3072]);  mul_61 = None
    permute_503: "f32[3072, 768]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_47: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg125_1, view_598, permute_503);  arg125_1 = view_598 = permute_503 = None
    view_599: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_47, [1, 1024, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_63: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_599);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_63, add_83);  clone_63 = add_83 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 1024, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    add_86: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_63: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_31);  add_85 = getitem_31 = None
    mul_62: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = rsqrt_15 = None
    mul_63: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_62, arg126_1);  mul_62 = arg126_1 = None
    add_87: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_63, arg127_1);  mul_63 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_504: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_87, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_600: "f32[1024, 768]" = torch.ops.aten.view.default(permute_504, [1024, 768])
    permute_505: "f32[768, 768]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_48: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg129_1, view_600, permute_505);  arg129_1 = view_600 = permute_505 = None
    view_601: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_48, [1024, 1, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_602: "f32[1024, 768]" = torch.ops.aten.view.default(permute_504, [1024, 768])
    permute_506: "f32[768, 768]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_49: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg131_1, view_602, permute_506);  arg131_1 = view_602 = permute_506 = None
    view_603: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_49, [1024, 1, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_604: "f32[1024, 768]" = torch.ops.aten.view.default(permute_504, [1024, 768]);  permute_504 = None
    permute_507: "f32[768, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_50: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg133_1, view_604, permute_507);  arg133_1 = view_604 = permute_507 = None
    view_605: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_50, [1024, 1, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_80: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_601, 8.0);  view_601 = None
    view_606: "f32[1024, 768]" = torch.ops.aten.view.default(div_80, [1024, 768]);  div_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_609: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_603, [1024, 1, 12, 64]);  view_603 = None
    permute_509: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_609, [1, 0, 2, 3]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_511: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_509, [0, 2, 1, 3]);  permute_509 = None
    view_611: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_511, [12, 1024, 64]);  permute_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_613: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_611, [12, 2, 512, 64]);  view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_49: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_613, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_153: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_49, 4);  as_strided_49 = None
    permute_513: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_153, [0, 1, 4, 2, 3]);  unsqueeze_153 = None
    view_614: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_606, [1024, 1, 768]);  view_606 = None
    view_615: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_614, [1024, 1, 12, 64]);  view_614 = None
    permute_515: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_615, [1, 0, 2, 3]);  view_615 = None
    permute_516: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_515, [0, 2, 1, 3]);  permute_515 = None
    view_616: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_516, [12, 1024, 64]);  permute_516 = None
    view_617: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_616, [12, 2, 512, 64]);  view_616 = None
    as_strided_50: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_617, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_617 = None
    unsqueeze_154: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_50, 4);  as_strided_50 = None
    permute_517: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_154, [0, 1, 2, 4, 3]);  unsqueeze_154 = None
    permute_518: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_517, [0, 1, 2, 4, 3]);  permute_517 = None
    clone_64: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_618: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_64, [36, 512, 64]);  clone_64 = None
    permute_519: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_513, [0, 1, 4, 3, 2]);  permute_513 = None
    clone_65: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_619: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_65, [36, 64, 512]);  clone_65 = None
    bmm_16: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_618, view_619);  view_618 = view_619 = None
    view_620: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_16, [12, 3, 512, 1, 512]);  bmm_16 = None
    permute_520: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_620, [0, 1, 2, 4, 3]);  view_620 = None
    view_621: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_520, [12, 3, 512, 512]);  permute_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_32: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_621, [0, 0, 0, 1], 0.0);  view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_622: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_32, [12, 3, 512, 513]);  constant_pad_nd_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_72: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1849: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_622, 0, 0, 9223372036854775807)
    slice_1850: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1849, 1, 0, 9223372036854775807);  slice_1849 = None
    slice_1851: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1850, 2, 0, 256);  slice_1850 = None
    slice_1852: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1851, 3, 0, 257);  slice_1851 = None
    slice_1853: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_72, 0, 0, 9223372036854775807)
    slice_1854: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1853, 1, 0, -1);  slice_1853 = None
    slice_1855: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1854, 2, 0, 9223372036854775807);  slice_1854 = None
    slice_1856: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1855, 3, 256, 9223372036854775807);  slice_1855 = None
    copy_96: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1856, slice_1852);  slice_1856 = slice_1852 = None
    slice_1857: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_72, 0, 0, 9223372036854775807)
    slice_1858: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1857, 1, 0, -1)
    slice_1859: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1858, 2, 0, 9223372036854775807)
    slice_scatter_352: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1859, copy_96, 3, 256, 9223372036854775807);  slice_1859 = copy_96 = None
    slice_scatter_353: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1858, slice_scatter_352, 2, 0, 9223372036854775807);  slice_1858 = slice_scatter_352 = None
    slice_scatter_354: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1857, slice_scatter_353, 1, 0, -1);  slice_1857 = slice_scatter_353 = None
    slice_scatter_355: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_72, slice_scatter_354, 0, 0, 9223372036854775807);  full_72 = slice_scatter_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1864: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_622, 0, 0, 9223372036854775807)
    select_160: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1864, 1, -1);  slice_1864 = None
    slice_1865: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_160, 1, 256, 9223372036854775807);  select_160 = None
    slice_1866: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1865, 2, 0, 257);  slice_1865 = None
    slice_1870: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_355, 0, 0, 9223372036854775807)
    select_162: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1870, 1, -1);  slice_1870 = None
    slice_1871: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_162, 1, 0, 9223372036854775807);  select_162 = None
    slice_1872: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1871, 2, 256, 9223372036854775807);  slice_1871 = None
    copy_97: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_1872, slice_1866);  slice_1872 = slice_1866 = None
    slice_1873: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_355, 0, 0, 9223372036854775807)
    select_163: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1873, 1, -1)
    slice_1874: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_163, 1, 0, 9223372036854775807)
    slice_scatter_356: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1874, copy_97, 2, 256, 9223372036854775807);  slice_1874 = copy_97 = None
    slice_scatter_357: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_163, slice_scatter_356, 1, 0, 9223372036854775807);  select_163 = slice_scatter_356 = None
    select_scatter_32: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1873, slice_scatter_357, 1, -1);  slice_1873 = slice_scatter_357 = None
    slice_scatter_358: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_355, select_scatter_32, 0, 0, 9223372036854775807);  slice_scatter_355 = select_scatter_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1878: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_622, 0, 0, 9223372036854775807)
    slice_1879: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1878, 1, 0, 9223372036854775807);  slice_1878 = None
    slice_1880: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1879, 2, -257, -1);  slice_1879 = None
    slice_1881: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1880, 3, 257, 9223372036854775807);  slice_1880 = None
    slice_1886: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_358, 0, 0, 9223372036854775807)
    slice_1887: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1886, 1, 1, 9223372036854775807);  slice_1886 = None
    slice_1888: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1887, 2, 0, 9223372036854775807);  slice_1887 = None
    slice_1889: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1888, 3, 0, 256);  slice_1888 = None
    copy_98: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1889, slice_1881);  slice_1889 = slice_1881 = None
    slice_1890: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_358, 0, 0, 9223372036854775807)
    slice_1891: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1890, 1, 1, 9223372036854775807)
    slice_1892: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1891, 2, 0, 9223372036854775807)
    slice_scatter_359: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1892, copy_98, 3, 0, 256);  slice_1892 = copy_98 = None
    slice_scatter_360: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1891, slice_scatter_359, 2, 0, 9223372036854775807);  slice_1891 = slice_scatter_359 = None
    slice_scatter_361: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1890, slice_scatter_360, 1, 1, 9223372036854775807);  slice_1890 = slice_scatter_360 = None
    slice_scatter_362: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_358, slice_scatter_361, 0, 0, 9223372036854775807);  slice_scatter_358 = slice_scatter_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_1897: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_622, 0, 0, 9223372036854775807);  view_622 = None
    select_165: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_1897, 1, 0);  slice_1897 = None
    slice_1898: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_165, 1, 0, 255);  select_165 = None
    slice_1899: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1898, 2, -255, 9223372036854775807);  slice_1898 = None
    slice_1903: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_362, 0, 0, 9223372036854775807)
    select_167: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1903, 1, 0);  slice_1903 = None
    slice_1904: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_167, 1, 1, 256);  select_167 = None
    slice_1905: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1904, 2, 1, 256);  slice_1904 = None
    copy_99: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_1905, slice_1899);  slice_1905 = slice_1899 = None
    slice_1906: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_362, 0, 0, 9223372036854775807)
    select_168: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_1906, 1, 0)
    slice_1907: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_168, 1, 1, 256)
    slice_scatter_363: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1907, copy_99, 2, 1, 256);  slice_1907 = copy_99 = None
    slice_scatter_364: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_168, slice_scatter_363, 1, 1, 256);  select_168 = slice_scatter_363 = None
    select_scatter_33: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1906, slice_scatter_364, 1, 0);  slice_1906 = slice_scatter_364 = None
    slice_scatter_365: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_362, select_scatter_33, 0, 0, 9223372036854775807);  slice_scatter_362 = select_scatter_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_73: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_32: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_155: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_32, -2);  iota_32 = None
    iota_33: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_156: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
    sub_65: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_155, unsqueeze_156);  unsqueeze_155 = unsqueeze_156 = None
    le_16: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_65, 0);  sub_65 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_64: "f32[256, 257]" = torch.ops.aten.where.self(le_16, full_73, scalar_tensor_32);  le_16 = full_73 = scalar_tensor_32 = None
    rev_32: "f32[256, 257]" = torch.ops.prims.rev.default(where_64, [0]);  where_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_157: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_32, 0);  rev_32 = None
    slice_1911: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_157, 1, 0, 9223372036854775807);  unsqueeze_157 = None
    unsqueeze_158: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_1911, 2);  slice_1911 = None
    slice_1912: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_158, 3, 0, 9223372036854775807);  unsqueeze_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_33: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_1912, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_32: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_1912, [1, 256, 12, 257]);  slice_1912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_625: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_365, [1, 12, 1024, 513])
    permute_523: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_625, [0, 2, 1, 3]);  view_625 = None
    slice_1917: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_523, 0, 0, 9223372036854775807);  permute_523 = None
    slice_1918: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1917, 1, 0, 256);  slice_1917 = None
    slice_1919: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1918, 2, 0, 9223372036854775807);  slice_1918 = None
    slice_1920: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1919, 3, 0, 257);  slice_1919 = None
    full_74: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_40: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_32, torch.bool);  expand_32 = None
    where_65: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_40, full_74, slice_1920);  convert_element_type_40 = full_74 = slice_1920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_626: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_365, [1, 12, 1024, 513])
    permute_524: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None
    slice_1925: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_524, 0, 0, 9223372036854775807);  permute_524 = None
    slice_1926: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1925, 1, 0, 256);  slice_1925 = None
    slice_1927: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1926, 2, 0, 9223372036854775807);  slice_1926 = None
    slice_1928: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1927, 3, 0, 257);  slice_1927 = None
    copy_100: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1928, where_65);  slice_1928 = where_65 = None
    view_627: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_365, [1, 12, 1024, 513]);  slice_scatter_365 = None
    permute_525: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
    slice_1929: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_525, 0, 0, 9223372036854775807)
    slice_1930: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1929, 1, 0, 256)
    slice_1931: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1930, 2, 0, 9223372036854775807)
    slice_scatter_366: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1931, copy_100, 3, 0, 257);  slice_1931 = copy_100 = None
    slice_scatter_367: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1930, slice_scatter_366, 2, 0, 9223372036854775807);  slice_1930 = slice_scatter_366 = None
    slice_scatter_368: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1929, slice_scatter_367, 1, 0, 256);  slice_1929 = slice_scatter_367 = None
    slice_scatter_369: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_525, slice_scatter_368, 0, 0, 9223372036854775807);  permute_525 = slice_scatter_368 = None
    permute_526: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_369, [0, 2, 1, 3]);  slice_scatter_369 = None
    view_628: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_526, [12, 4, 256, 513]);  permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_33: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_33, [1, 256, 12, 257]);  rev_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_630: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_628, [1, 12, 1024, 513])
    permute_528: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
    slice_1940: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_528, 0, 0, 9223372036854775807);  permute_528 = None
    slice_1941: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1940, 1, -256, 9223372036854775807);  slice_1940 = None
    slice_1942: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1941, 2, 0, 9223372036854775807);  slice_1941 = None
    slice_1943: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1942, 3, -257, 9223372036854775807);  slice_1942 = None
    full_75: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_41: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_33, torch.bool);  expand_33 = None
    where_66: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_41, full_75, slice_1943);  convert_element_type_41 = full_75 = slice_1943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_631: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_628, [1, 12, 1024, 513])
    permute_529: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
    slice_1948: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_529, 0, 0, 9223372036854775807);  permute_529 = None
    slice_1949: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1948, 1, -256, 9223372036854775807);  slice_1948 = None
    slice_1950: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1949, 2, 0, 9223372036854775807);  slice_1949 = None
    slice_1951: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1950, 3, -257, 9223372036854775807);  slice_1950 = None
    copy_101: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1951, where_66);  slice_1951 = where_66 = None
    view_632: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_628, [1, 12, 1024, 513]);  view_628 = None
    permute_530: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_632, [0, 2, 1, 3]);  view_632 = None
    slice_1952: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_530, 0, 0, 9223372036854775807)
    slice_1953: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1952, 1, -256, 9223372036854775807)
    slice_1954: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_1953, 2, 0, 9223372036854775807)
    slice_scatter_370: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1954, copy_101, 3, -257, 9223372036854775807);  slice_1954 = copy_101 = None
    slice_scatter_371: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1953, slice_scatter_370, 2, 0, 9223372036854775807);  slice_1953 = slice_scatter_370 = None
    slice_scatter_372: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1952, slice_scatter_371, 1, -256, 9223372036854775807);  slice_1952 = slice_scatter_371 = None
    slice_scatter_373: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_530, slice_scatter_372, 0, 0, 9223372036854775807);  permute_530 = slice_scatter_372 = None
    permute_531: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_373, [0, 2, 1, 3]);  slice_scatter_373 = None
    view_633: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_531, [12, 4, 256, 513]);  permute_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_8: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_1959: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_8, 0, 0, 9223372036854775807);  ne_8 = None
    slice_1960: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_1959, 1, 0, 9223372036854775807);  slice_1959 = None
    unsqueeze_159: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_1960, 2);  slice_1960 = None
    unsqueeze_160: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_42: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_160, torch.float32)
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_67: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_160, scalar_tensor_33, convert_element_type_42);  unsqueeze_160 = scalar_tensor_33 = convert_element_type_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_76: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_533: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_76, [0, 2, 1, 3]);  full_76 = None
    view_635: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_533, [1, 1024, 1]);  permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_534: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_67, [0, 2, 1, 3]);  where_67 = None
    view_636: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_534, [1, 1024, 1]);  permute_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_637: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_635, [1, 2, 512, 1]);  view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_51: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_637, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_638: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_636, [1, 2, 512, 1]);  view_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_52: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_638, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_161: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_51, 4);  as_strided_51 = None
    permute_535: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_161, [0, 1, 2, 4, 3]);  unsqueeze_161 = None
    unsqueeze_162: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_52, 4);  as_strided_52 = None
    permute_536: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_162, [0, 1, 4, 2, 3]);  unsqueeze_162 = None
    mul_64: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_535, permute_536);  permute_535 = permute_536 = None
    view_639: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_64, [1, 3, 512, 512]);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_33: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_639, [0, 0, 0, 1], 0.0);  view_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_640: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_33, [1, 3, 512, 513]);  constant_pad_nd_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_77: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1961: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_640, 0, 0, 9223372036854775807)
    slice_1962: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1961, 1, 0, 9223372036854775807);  slice_1961 = None
    slice_1963: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1962, 2, 0, 256);  slice_1962 = None
    slice_1964: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1963, 3, 0, 257);  slice_1963 = None
    slice_1965: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_77, 0, 0, 9223372036854775807)
    slice_1966: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1965, 1, 0, -1);  slice_1965 = None
    slice_1967: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1966, 2, 0, 9223372036854775807);  slice_1966 = None
    slice_1968: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1967, 3, 256, 9223372036854775807);  slice_1967 = None
    copy_102: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1968, slice_1964);  slice_1968 = slice_1964 = None
    slice_1969: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_77, 0, 0, 9223372036854775807)
    slice_1970: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1969, 1, 0, -1)
    slice_1971: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1970, 2, 0, 9223372036854775807)
    slice_scatter_374: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1971, copy_102, 3, 256, 9223372036854775807);  slice_1971 = copy_102 = None
    slice_scatter_375: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1970, slice_scatter_374, 2, 0, 9223372036854775807);  slice_1970 = slice_scatter_374 = None
    slice_scatter_376: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1969, slice_scatter_375, 1, 0, -1);  slice_1969 = slice_scatter_375 = None
    slice_scatter_377: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_77, slice_scatter_376, 0, 0, 9223372036854775807);  full_77 = slice_scatter_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_1976: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_640, 0, 0, 9223372036854775807)
    select_170: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_1976, 1, -1);  slice_1976 = None
    slice_1977: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_170, 1, 256, 9223372036854775807);  select_170 = None
    slice_1978: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1977, 2, 0, 257);  slice_1977 = None
    slice_1982: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_377, 0, 0, 9223372036854775807)
    select_172: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1982, 1, -1);  slice_1982 = None
    slice_1983: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_172, 1, 0, 9223372036854775807);  select_172 = None
    slice_1984: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1983, 2, 256, 9223372036854775807);  slice_1983 = None
    copy_103: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_1984, slice_1978);  slice_1984 = slice_1978 = None
    slice_1985: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_377, 0, 0, 9223372036854775807)
    select_173: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_1985, 1, -1)
    slice_1986: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_173, 1, 0, 9223372036854775807)
    slice_scatter_378: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1986, copy_103, 2, 256, 9223372036854775807);  slice_1986 = copy_103 = None
    slice_scatter_379: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_173, slice_scatter_378, 1, 0, 9223372036854775807);  select_173 = slice_scatter_378 = None
    select_scatter_34: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_1985, slice_scatter_379, 1, -1);  slice_1985 = slice_scatter_379 = None
    slice_scatter_380: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_377, select_scatter_34, 0, 0, 9223372036854775807);  slice_scatter_377 = select_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_1990: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_640, 0, 0, 9223372036854775807)
    slice_1991: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_1990, 1, 0, 9223372036854775807);  slice_1990 = None
    slice_1992: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1991, 2, -257, -1);  slice_1991 = None
    slice_1993: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1992, 3, 257, 9223372036854775807);  slice_1992 = None
    slice_1998: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_380, 0, 0, 9223372036854775807)
    slice_1999: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1998, 1, 1, 9223372036854775807);  slice_1998 = None
    slice_2000: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_1999, 2, 0, 9223372036854775807);  slice_1999 = None
    slice_2001: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2000, 3, 0, 256);  slice_2000 = None
    copy_104: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2001, slice_1993);  slice_2001 = slice_1993 = None
    slice_2002: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_380, 0, 0, 9223372036854775807)
    slice_2003: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2002, 1, 1, 9223372036854775807)
    slice_2004: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2003, 2, 0, 9223372036854775807)
    slice_scatter_381: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2004, copy_104, 3, 0, 256);  slice_2004 = copy_104 = None
    slice_scatter_382: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2003, slice_scatter_381, 2, 0, 9223372036854775807);  slice_2003 = slice_scatter_381 = None
    slice_scatter_383: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2002, slice_scatter_382, 1, 1, 9223372036854775807);  slice_2002 = slice_scatter_382 = None
    slice_scatter_384: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_380, slice_scatter_383, 0, 0, 9223372036854775807);  slice_scatter_380 = slice_scatter_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_2009: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_640, 0, 0, 9223372036854775807);  view_640 = None
    select_175: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_2009, 1, 0);  slice_2009 = None
    slice_2010: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_175, 1, 0, 255);  select_175 = None
    slice_2011: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2010, 2, -255, 9223372036854775807);  slice_2010 = None
    slice_2015: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_384, 0, 0, 9223372036854775807)
    select_177: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2015, 1, 0);  slice_2015 = None
    slice_2016: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_177, 1, 1, 256);  select_177 = None
    slice_2017: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2016, 2, 1, 256);  slice_2016 = None
    copy_105: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_2017, slice_2011);  slice_2017 = slice_2011 = None
    slice_2018: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_384, 0, 0, 9223372036854775807)
    select_178: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2018, 1, 0)
    slice_2019: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_178, 1, 1, 256)
    slice_scatter_385: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2019, copy_105, 2, 1, 256);  slice_2019 = copy_105 = None
    slice_scatter_386: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_178, slice_scatter_385, 1, 1, 256);  select_178 = slice_scatter_385 = None
    select_scatter_35: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2018, slice_scatter_386, 1, 0);  slice_2018 = slice_scatter_386 = None
    slice_scatter_387: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_384, select_scatter_35, 0, 0, 9223372036854775807);  slice_scatter_384 = select_scatter_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_78: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_34: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_163: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_34, -2);  iota_34 = None
    iota_35: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_164: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
    sub_67: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_163, unsqueeze_164);  unsqueeze_163 = unsqueeze_164 = None
    le_17: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_67, 0);  sub_67 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_68: "f32[256, 257]" = torch.ops.aten.where.self(le_17, full_78, scalar_tensor_34);  le_17 = full_78 = scalar_tensor_34 = None
    rev_34: "f32[256, 257]" = torch.ops.prims.rev.default(where_68, [0]);  where_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_165: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_34, 0);  rev_34 = None
    slice_2023: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_165, 1, 0, 9223372036854775807);  unsqueeze_165 = None
    unsqueeze_166: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_2023, 2);  slice_2023 = None
    slice_2024: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_166, 3, 0, 9223372036854775807);  unsqueeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_35: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_2024, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_34: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_2024, [1, 256, 1, 257]);  slice_2024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_643: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_387, [1, 1, 1024, 513])
    permute_539: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_643, [0, 2, 1, 3]);  view_643 = None
    slice_2029: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_539, 0, 0, 9223372036854775807);  permute_539 = None
    slice_2030: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2029, 1, 0, 256);  slice_2029 = None
    slice_2031: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2030, 2, 0, 9223372036854775807);  slice_2030 = None
    slice_2032: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2031, 3, 0, 257);  slice_2031 = None
    full_79: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_43: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_34, torch.bool);  expand_34 = None
    where_69: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_43, full_79, slice_2032);  convert_element_type_43 = full_79 = slice_2032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_644: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_387, [1, 1, 1024, 513])
    permute_540: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_644, [0, 2, 1, 3]);  view_644 = None
    slice_2037: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_540, 0, 0, 9223372036854775807);  permute_540 = None
    slice_2038: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2037, 1, 0, 256);  slice_2037 = None
    slice_2039: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2038, 2, 0, 9223372036854775807);  slice_2038 = None
    slice_2040: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2039, 3, 0, 257);  slice_2039 = None
    copy_106: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2040, where_69);  slice_2040 = where_69 = None
    view_645: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_387, [1, 1, 1024, 513]);  slice_scatter_387 = None
    permute_541: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
    slice_2041: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_541, 0, 0, 9223372036854775807)
    slice_2042: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2041, 1, 0, 256)
    slice_2043: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2042, 2, 0, 9223372036854775807)
    slice_scatter_388: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2043, copy_106, 3, 0, 257);  slice_2043 = copy_106 = None
    slice_scatter_389: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2042, slice_scatter_388, 2, 0, 9223372036854775807);  slice_2042 = slice_scatter_388 = None
    slice_scatter_390: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2041, slice_scatter_389, 1, 0, 256);  slice_2041 = slice_scatter_389 = None
    slice_scatter_391: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_541, slice_scatter_390, 0, 0, 9223372036854775807);  permute_541 = slice_scatter_390 = None
    permute_542: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_391, [0, 2, 1, 3]);  slice_scatter_391 = None
    view_646: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_542, [1, 4, 256, 513]);  permute_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_35: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_35, [1, 256, 1, 257]);  rev_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_648: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_646, [1, 1, 1024, 513])
    permute_544: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None
    slice_2052: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_544, 0, 0, 9223372036854775807);  permute_544 = None
    slice_2053: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2052, 1, -256, 9223372036854775807);  slice_2052 = None
    slice_2054: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2053, 2, 0, 9223372036854775807);  slice_2053 = None
    slice_2055: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2054, 3, -257, 9223372036854775807);  slice_2054 = None
    full_80: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_44: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_35, torch.bool);  expand_35 = None
    where_70: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_44, full_80, slice_2055);  convert_element_type_44 = full_80 = slice_2055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_649: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_646, [1, 1, 1024, 513])
    permute_545: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None
    slice_2060: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_545, 0, 0, 9223372036854775807);  permute_545 = None
    slice_2061: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2060, 1, -256, 9223372036854775807);  slice_2060 = None
    slice_2062: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2061, 2, 0, 9223372036854775807);  slice_2061 = None
    slice_2063: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2062, 3, -257, 9223372036854775807);  slice_2062 = None
    copy_107: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2063, where_70);  slice_2063 = where_70 = None
    view_650: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_646, [1, 1, 1024, 513]);  view_646 = None
    permute_546: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_650, [0, 2, 1, 3]);  view_650 = None
    slice_2064: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_546, 0, 0, 9223372036854775807)
    slice_2065: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2064, 1, -256, 9223372036854775807)
    slice_2066: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2065, 2, 0, 9223372036854775807)
    slice_scatter_392: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2066, copy_107, 3, -257, 9223372036854775807);  slice_2066 = copy_107 = None
    slice_scatter_393: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2065, slice_scatter_392, 2, 0, 9223372036854775807);  slice_2065 = slice_scatter_392 = None
    slice_scatter_394: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2064, slice_scatter_393, 1, -256, 9223372036854775807);  slice_2064 = slice_scatter_393 = None
    slice_scatter_395: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_546, slice_scatter_394, 0, 0, 9223372036854775807);  permute_546 = slice_scatter_394 = None
    permute_547: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_395, [0, 2, 1, 3]);  slice_scatter_395 = None
    view_651: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_547, [1, 4, 256, 513]);  permute_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_653: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_633, [1, 12, 1024, 513]);  view_633 = None
    permute_549: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    view_654: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_651, [1, 1, 1024, 513]);  view_651 = None
    permute_550: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    add_90: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_549, permute_550);  permute_549 = permute_550 = None
    permute_551: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_90, [0, 2, 1, 3]);  add_90 = None
    view_656: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_551, [12, 4, 256, 513]);  permute_551 = None
    view_657: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_656, [1, 12, 1024, 513]);  view_656 = None
    permute_552: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_657, [0, 2, 1, 3]);  view_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_66: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
    amax_8: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_66, [-1], True)
    sub_68: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_66, amax_8);  clone_66 = amax_8 = None
    exp_8: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
    sum_9: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_87: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_2071: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_2072: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_2071, 1, 0, 9223372036854775807);  slice_2071 = None
    unsqueeze_167: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_2072, 2);  slice_2072 = None
    unsqueeze_168: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_71: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_168, scalar_tensor_35, div_87);  unsqueeze_168 = scalar_tensor_35 = div_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_67: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_71);  where_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_658: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_605, [1024, 1, 12, 64]);  view_605 = None
    permute_553: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_658, [1, 0, 2, 3]);  view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_554: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_67, [0, 2, 1, 3]);  clone_67 = None
    view_659: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_554, [12, 4, 256, 513]);  permute_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_555: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_553, [0, 2, 1, 3]);  permute_553 = None
    view_660: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_555, [12, 1024, 64]);  permute_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_34: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_660, [0, 0, 256, 256], -1.0);  view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_53: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_34, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_35: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_659, [0, 257], 0.0);  view_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_661: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_35, [12, 4, -1]);  constant_pad_nd_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_2073: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_661, 0, 0, 9223372036854775807);  view_661 = None
    slice_2074: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_2073, 1, 0, 9223372036854775807);  slice_2073 = None
    slice_2075: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_2074, 2, 0, -256);  slice_2074 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_662: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_2075, [12, 4, 256, 769]);  slice_2075 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_2076: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_662, 0, 0, 9223372036854775807);  view_662 = None
    slice_2077: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2076, 1, 0, 9223372036854775807);  slice_2076 = None
    slice_2078: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2077, 2, 0, 9223372036854775807);  slice_2077 = None
    slice_2079: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_2078, 3, 0, -1);  slice_2078 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_169: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2079, 4);  slice_2079 = None
    permute_556: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_169, [0, 1, 2, 4, 3]);  unsqueeze_169 = None
    unsqueeze_170: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_53, 4);  as_strided_53 = None
    permute_557: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_170, [0, 1, 4, 3, 2]);  unsqueeze_170 = None
    permute_558: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_556, [0, 1, 2, 4, 3]);  permute_556 = None
    view_663: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_558, [48, 256, 768]);  permute_558 = None
    permute_559: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_557, [0, 1, 4, 3, 2]);  permute_557 = None
    clone_68: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
    view_664: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_68, [48, 768, 64]);  clone_68 = None
    bmm_17: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_663, view_664);  view_663 = view_664 = None
    view_665: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_17, [12, 4, 256, 1, 64]);  bmm_17 = None
    permute_560: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_665, [0, 1, 2, 4, 3]);  view_665 = None
    view_666: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_560, [12, 4, 256, 64]);  permute_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_667: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_666, [1, 12, 1024, 64]);  view_666 = None
    permute_561: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_667, [0, 2, 1, 3]);  view_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_562: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_561, [1, 0, 2, 3]);  permute_561 = None
    clone_69: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_562, memory_format = torch.contiguous_format);  permute_562 = None
    view_668: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_69, [1024, 1, 768]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_563: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_668, [1, 0, 2]);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_669: "f32[1024, 768]" = torch.ops.aten.view.default(permute_563, [1024, 768]);  permute_563 = None
    permute_564: "f32[768, 768]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_51: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg135_1, view_669, permute_564);  arg135_1 = view_669 = permute_564 = None
    view_670: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_51, [1, 1024, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_70: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_670);  view_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_92: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_70, add_87);  clone_70 = add_87 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 1024, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    add_93: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_70: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_33);  add_92 = getitem_33 = None
    mul_65: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_16);  sub_70 = rsqrt_16 = None
    mul_66: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg136_1);  mul_65 = arg136_1 = None
    add_94: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_66, arg137_1);  mul_66 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_671: "f32[1024, 768]" = torch.ops.aten.view.default(add_94, [1024, 768])
    permute_565: "f32[768, 3072]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_52: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg139_1, view_671, permute_565);  arg139_1 = view_671 = permute_565 = None
    view_672: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_52, [1, 1024, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_67: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_672, 0.5)
    mul_68: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_672, 0.7071067811865476);  view_672 = None
    erf_8: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_95: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_69: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_95);  mul_67 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_673: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_69, [1024, 3072]);  mul_69 = None
    permute_566: "f32[3072, 768]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_53: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg141_1, view_673, permute_566);  arg141_1 = view_673 = permute_566 = None
    view_674: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_53, [1, 1024, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_71: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_674);  view_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_96: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_71, add_94);  clone_71 = add_94 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 1024, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    add_97: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_71: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_35);  add_96 = getitem_35 = None
    mul_70: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_17);  sub_71 = rsqrt_17 = None
    mul_71: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg142_1);  mul_70 = arg142_1 = None
    add_98: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_71, arg143_1);  mul_71 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_567: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_98, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_675: "f32[1024, 768]" = torch.ops.aten.view.default(permute_567, [1024, 768])
    permute_568: "f32[768, 768]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_54: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg145_1, view_675, permute_568);  arg145_1 = view_675 = permute_568 = None
    view_676: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_54, [1024, 1, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_677: "f32[1024, 768]" = torch.ops.aten.view.default(permute_567, [1024, 768])
    permute_569: "f32[768, 768]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    addmm_55: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg147_1, view_677, permute_569);  arg147_1 = view_677 = permute_569 = None
    view_678: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_55, [1024, 1, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_679: "f32[1024, 768]" = torch.ops.aten.view.default(permute_567, [1024, 768]);  permute_567 = None
    permute_570: "f32[768, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_56: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg149_1, view_679, permute_570);  arg149_1 = view_679 = permute_570 = None
    view_680: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_56, [1024, 1, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_90: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_676, 8.0);  view_676 = None
    view_681: "f32[1024, 768]" = torch.ops.aten.view.default(div_90, [1024, 768]);  div_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_684: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_678, [1024, 1, 12, 64]);  view_678 = None
    permute_572: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_684, [1, 0, 2, 3]);  view_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_574: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_572, [0, 2, 1, 3]);  permute_572 = None
    view_686: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_574, [12, 1024, 64]);  permute_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_688: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_686, [12, 2, 512, 64]);  view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_55: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_688, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_172: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_55, 4);  as_strided_55 = None
    permute_576: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_172, [0, 1, 4, 2, 3]);  unsqueeze_172 = None
    view_689: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_681, [1024, 1, 768]);  view_681 = None
    view_690: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_689, [1024, 1, 12, 64]);  view_689 = None
    permute_578: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_690, [1, 0, 2, 3]);  view_690 = None
    permute_579: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_578, [0, 2, 1, 3]);  permute_578 = None
    view_691: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_579, [12, 1024, 64]);  permute_579 = None
    view_692: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_691, [12, 2, 512, 64]);  view_691 = None
    as_strided_56: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_692, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_692 = None
    unsqueeze_173: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_56, 4);  as_strided_56 = None
    permute_580: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_173, [0, 1, 2, 4, 3]);  unsqueeze_173 = None
    permute_581: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_580, [0, 1, 2, 4, 3]);  permute_580 = None
    clone_72: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_581, memory_format = torch.contiguous_format);  permute_581 = None
    view_693: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_72, [36, 512, 64]);  clone_72 = None
    permute_582: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_576, [0, 1, 4, 3, 2]);  permute_576 = None
    clone_73: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_582, memory_format = torch.contiguous_format);  permute_582 = None
    view_694: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_73, [36, 64, 512]);  clone_73 = None
    bmm_18: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_693, view_694);  view_693 = view_694 = None
    view_695: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_18, [12, 3, 512, 1, 512]);  bmm_18 = None
    permute_583: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_695, [0, 1, 2, 4, 3]);  view_695 = None
    view_696: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_583, [12, 3, 512, 512]);  permute_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_36: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_696, [0, 0, 0, 1], 0.0);  view_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_697: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_36, [12, 3, 512, 513]);  constant_pad_nd_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_81: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2080: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_697, 0, 0, 9223372036854775807)
    slice_2081: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2080, 1, 0, 9223372036854775807);  slice_2080 = None
    slice_2082: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2081, 2, 0, 256);  slice_2081 = None
    slice_2083: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2082, 3, 0, 257);  slice_2082 = None
    slice_2084: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_81, 0, 0, 9223372036854775807)
    slice_2085: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2084, 1, 0, -1);  slice_2084 = None
    slice_2086: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2085, 2, 0, 9223372036854775807);  slice_2085 = None
    slice_2087: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2086, 3, 256, 9223372036854775807);  slice_2086 = None
    copy_108: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2087, slice_2083);  slice_2087 = slice_2083 = None
    slice_2088: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_81, 0, 0, 9223372036854775807)
    slice_2089: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2088, 1, 0, -1)
    slice_2090: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2089, 2, 0, 9223372036854775807)
    slice_scatter_396: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2090, copy_108, 3, 256, 9223372036854775807);  slice_2090 = copy_108 = None
    slice_scatter_397: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2089, slice_scatter_396, 2, 0, 9223372036854775807);  slice_2089 = slice_scatter_396 = None
    slice_scatter_398: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2088, slice_scatter_397, 1, 0, -1);  slice_2088 = slice_scatter_397 = None
    slice_scatter_399: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_81, slice_scatter_398, 0, 0, 9223372036854775807);  full_81 = slice_scatter_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2095: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_697, 0, 0, 9223372036854775807)
    select_180: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2095, 1, -1);  slice_2095 = None
    slice_2096: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_180, 1, 256, 9223372036854775807);  select_180 = None
    slice_2097: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2096, 2, 0, 257);  slice_2096 = None
    slice_2101: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_399, 0, 0, 9223372036854775807)
    select_182: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2101, 1, -1);  slice_2101 = None
    slice_2102: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_182, 1, 0, 9223372036854775807);  select_182 = None
    slice_2103: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2102, 2, 256, 9223372036854775807);  slice_2102 = None
    copy_109: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_2103, slice_2097);  slice_2103 = slice_2097 = None
    slice_2104: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_399, 0, 0, 9223372036854775807)
    select_183: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2104, 1, -1)
    slice_2105: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_183, 1, 0, 9223372036854775807)
    slice_scatter_400: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2105, copy_109, 2, 256, 9223372036854775807);  slice_2105 = copy_109 = None
    slice_scatter_401: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_183, slice_scatter_400, 1, 0, 9223372036854775807);  select_183 = slice_scatter_400 = None
    select_scatter_36: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2104, slice_scatter_401, 1, -1);  slice_2104 = slice_scatter_401 = None
    slice_scatter_402: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_399, select_scatter_36, 0, 0, 9223372036854775807);  slice_scatter_399 = select_scatter_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_2109: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_697, 0, 0, 9223372036854775807)
    slice_2110: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2109, 1, 0, 9223372036854775807);  slice_2109 = None
    slice_2111: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2110, 2, -257, -1);  slice_2110 = None
    slice_2112: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2111, 3, 257, 9223372036854775807);  slice_2111 = None
    slice_2117: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_402, 0, 0, 9223372036854775807)
    slice_2118: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2117, 1, 1, 9223372036854775807);  slice_2117 = None
    slice_2119: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2118, 2, 0, 9223372036854775807);  slice_2118 = None
    slice_2120: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2119, 3, 0, 256);  slice_2119 = None
    copy_110: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2120, slice_2112);  slice_2120 = slice_2112 = None
    slice_2121: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_402, 0, 0, 9223372036854775807)
    slice_2122: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2121, 1, 1, 9223372036854775807)
    slice_2123: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2122, 2, 0, 9223372036854775807)
    slice_scatter_403: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2123, copy_110, 3, 0, 256);  slice_2123 = copy_110 = None
    slice_scatter_404: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2122, slice_scatter_403, 2, 0, 9223372036854775807);  slice_2122 = slice_scatter_403 = None
    slice_scatter_405: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2121, slice_scatter_404, 1, 1, 9223372036854775807);  slice_2121 = slice_scatter_404 = None
    slice_scatter_406: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_402, slice_scatter_405, 0, 0, 9223372036854775807);  slice_scatter_402 = slice_scatter_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_2128: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_697, 0, 0, 9223372036854775807);  view_697 = None
    select_185: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2128, 1, 0);  slice_2128 = None
    slice_2129: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_185, 1, 0, 255);  select_185 = None
    slice_2130: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2129, 2, -255, 9223372036854775807);  slice_2129 = None
    slice_2134: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_406, 0, 0, 9223372036854775807)
    select_187: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2134, 1, 0);  slice_2134 = None
    slice_2135: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_187, 1, 1, 256);  select_187 = None
    slice_2136: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2135, 2, 1, 256);  slice_2135 = None
    copy_111: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_2136, slice_2130);  slice_2136 = slice_2130 = None
    slice_2137: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_406, 0, 0, 9223372036854775807)
    select_188: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2137, 1, 0)
    slice_2138: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_188, 1, 1, 256)
    slice_scatter_407: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2138, copy_111, 2, 1, 256);  slice_2138 = copy_111 = None
    slice_scatter_408: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_188, slice_scatter_407, 1, 1, 256);  select_188 = slice_scatter_407 = None
    select_scatter_37: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2137, slice_scatter_408, 1, 0);  slice_2137 = slice_scatter_408 = None
    slice_scatter_409: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_406, select_scatter_37, 0, 0, 9223372036854775807);  slice_scatter_406 = select_scatter_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_82: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_36: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_174: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_36, -2);  iota_36 = None
    iota_37: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_175: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
    sub_73: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_174, unsqueeze_175);  unsqueeze_174 = unsqueeze_175 = None
    le_18: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_73, 0);  sub_73 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_72: "f32[256, 257]" = torch.ops.aten.where.self(le_18, full_82, scalar_tensor_36);  le_18 = full_82 = scalar_tensor_36 = None
    rev_36: "f32[256, 257]" = torch.ops.prims.rev.default(where_72, [0]);  where_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_176: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_36, 0);  rev_36 = None
    slice_2142: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_176, 1, 0, 9223372036854775807);  unsqueeze_176 = None
    unsqueeze_177: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_2142, 2);  slice_2142 = None
    slice_2143: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_177, 3, 0, 9223372036854775807);  unsqueeze_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_37: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_2143, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_36: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_2143, [1, 256, 12, 257]);  slice_2143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_700: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_409, [1, 12, 1024, 513])
    permute_586: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
    slice_2148: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_586, 0, 0, 9223372036854775807);  permute_586 = None
    slice_2149: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2148, 1, 0, 256);  slice_2148 = None
    slice_2150: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2149, 2, 0, 9223372036854775807);  slice_2149 = None
    slice_2151: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2150, 3, 0, 257);  slice_2150 = None
    full_83: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_45: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_36, torch.bool);  expand_36 = None
    where_73: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_45, full_83, slice_2151);  convert_element_type_45 = full_83 = slice_2151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_701: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_409, [1, 12, 1024, 513])
    permute_587: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_701, [0, 2, 1, 3]);  view_701 = None
    slice_2156: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_587, 0, 0, 9223372036854775807);  permute_587 = None
    slice_2157: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2156, 1, 0, 256);  slice_2156 = None
    slice_2158: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2157, 2, 0, 9223372036854775807);  slice_2157 = None
    slice_2159: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2158, 3, 0, 257);  slice_2158 = None
    copy_112: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2159, where_73);  slice_2159 = where_73 = None
    view_702: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_409, [1, 12, 1024, 513]);  slice_scatter_409 = None
    permute_588: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_702, [0, 2, 1, 3]);  view_702 = None
    slice_2160: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_588, 0, 0, 9223372036854775807)
    slice_2161: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2160, 1, 0, 256)
    slice_2162: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2161, 2, 0, 9223372036854775807)
    slice_scatter_410: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2162, copy_112, 3, 0, 257);  slice_2162 = copy_112 = None
    slice_scatter_411: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2161, slice_scatter_410, 2, 0, 9223372036854775807);  slice_2161 = slice_scatter_410 = None
    slice_scatter_412: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2160, slice_scatter_411, 1, 0, 256);  slice_2160 = slice_scatter_411 = None
    slice_scatter_413: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_588, slice_scatter_412, 0, 0, 9223372036854775807);  permute_588 = slice_scatter_412 = None
    permute_589: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_413, [0, 2, 1, 3]);  slice_scatter_413 = None
    view_703: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_589, [12, 4, 256, 513]);  permute_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_37: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_37, [1, 256, 12, 257]);  rev_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_705: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_703, [1, 12, 1024, 513])
    permute_591: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_705, [0, 2, 1, 3]);  view_705 = None
    slice_2171: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_591, 0, 0, 9223372036854775807);  permute_591 = None
    slice_2172: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2171, 1, -256, 9223372036854775807);  slice_2171 = None
    slice_2173: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2172, 2, 0, 9223372036854775807);  slice_2172 = None
    slice_2174: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2173, 3, -257, 9223372036854775807);  slice_2173 = None
    full_84: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_46: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_37, torch.bool);  expand_37 = None
    where_74: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_46, full_84, slice_2174);  convert_element_type_46 = full_84 = slice_2174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_706: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_703, [1, 12, 1024, 513])
    permute_592: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_706, [0, 2, 1, 3]);  view_706 = None
    slice_2179: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_592, 0, 0, 9223372036854775807);  permute_592 = None
    slice_2180: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2179, 1, -256, 9223372036854775807);  slice_2179 = None
    slice_2181: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2180, 2, 0, 9223372036854775807);  slice_2180 = None
    slice_2182: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2181, 3, -257, 9223372036854775807);  slice_2181 = None
    copy_113: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2182, where_74);  slice_2182 = where_74 = None
    view_707: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_703, [1, 12, 1024, 513]);  view_703 = None
    permute_593: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_707, [0, 2, 1, 3]);  view_707 = None
    slice_2183: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_593, 0, 0, 9223372036854775807)
    slice_2184: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2183, 1, -256, 9223372036854775807)
    slice_2185: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2184, 2, 0, 9223372036854775807)
    slice_scatter_414: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2185, copy_113, 3, -257, 9223372036854775807);  slice_2185 = copy_113 = None
    slice_scatter_415: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2184, slice_scatter_414, 2, 0, 9223372036854775807);  slice_2184 = slice_scatter_414 = None
    slice_scatter_416: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2183, slice_scatter_415, 1, -256, 9223372036854775807);  slice_2183 = slice_scatter_415 = None
    slice_scatter_417: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_593, slice_scatter_416, 0, 0, 9223372036854775807);  permute_593 = slice_scatter_416 = None
    permute_594: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_417, [0, 2, 1, 3]);  slice_scatter_417 = None
    view_708: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_594, [12, 4, 256, 513]);  permute_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_9: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_2190: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_9, 0, 0, 9223372036854775807);  ne_9 = None
    slice_2191: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_2190, 1, 0, 9223372036854775807);  slice_2190 = None
    unsqueeze_178: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_2191, 2);  slice_2191 = None
    unsqueeze_179: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_47: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_179, torch.float32)
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_75: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_179, scalar_tensor_37, convert_element_type_47);  unsqueeze_179 = scalar_tensor_37 = convert_element_type_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_85: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_596: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_85, [0, 2, 1, 3]);  full_85 = None
    view_710: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_596, [1, 1024, 1]);  permute_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_597: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_75, [0, 2, 1, 3]);  where_75 = None
    view_711: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_597, [1, 1024, 1]);  permute_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_712: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_710, [1, 2, 512, 1]);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_57: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_712, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_713: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_711, [1, 2, 512, 1]);  view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_58: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_713, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_180: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_57, 4);  as_strided_57 = None
    permute_598: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_180, [0, 1, 2, 4, 3]);  unsqueeze_180 = None
    unsqueeze_181: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_58, 4);  as_strided_58 = None
    permute_599: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_181, [0, 1, 4, 2, 3]);  unsqueeze_181 = None
    mul_72: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_598, permute_599);  permute_598 = permute_599 = None
    view_714: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_72, [1, 3, 512, 512]);  mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_37: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_714, [0, 0, 0, 1], 0.0);  view_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_715: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_37, [1, 3, 512, 513]);  constant_pad_nd_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_86: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2192: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_715, 0, 0, 9223372036854775807)
    slice_2193: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2192, 1, 0, 9223372036854775807);  slice_2192 = None
    slice_2194: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2193, 2, 0, 256);  slice_2193 = None
    slice_2195: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2194, 3, 0, 257);  slice_2194 = None
    slice_2196: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_86, 0, 0, 9223372036854775807)
    slice_2197: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2196, 1, 0, -1);  slice_2196 = None
    slice_2198: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2197, 2, 0, 9223372036854775807);  slice_2197 = None
    slice_2199: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2198, 3, 256, 9223372036854775807);  slice_2198 = None
    copy_114: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2199, slice_2195);  slice_2199 = slice_2195 = None
    slice_2200: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_86, 0, 0, 9223372036854775807)
    slice_2201: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2200, 1, 0, -1)
    slice_2202: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2201, 2, 0, 9223372036854775807)
    slice_scatter_418: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2202, copy_114, 3, 256, 9223372036854775807);  slice_2202 = copy_114 = None
    slice_scatter_419: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2201, slice_scatter_418, 2, 0, 9223372036854775807);  slice_2201 = slice_scatter_418 = None
    slice_scatter_420: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2200, slice_scatter_419, 1, 0, -1);  slice_2200 = slice_scatter_419 = None
    slice_scatter_421: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_86, slice_scatter_420, 0, 0, 9223372036854775807);  full_86 = slice_scatter_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2207: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_715, 0, 0, 9223372036854775807)
    select_190: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_2207, 1, -1);  slice_2207 = None
    slice_2208: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_190, 1, 256, 9223372036854775807);  select_190 = None
    slice_2209: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2208, 2, 0, 257);  slice_2208 = None
    slice_2213: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_421, 0, 0, 9223372036854775807)
    select_192: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2213, 1, -1);  slice_2213 = None
    slice_2214: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_192, 1, 0, 9223372036854775807);  select_192 = None
    slice_2215: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2214, 2, 256, 9223372036854775807);  slice_2214 = None
    copy_115: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_2215, slice_2209);  slice_2215 = slice_2209 = None
    slice_2216: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_421, 0, 0, 9223372036854775807)
    select_193: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2216, 1, -1)
    slice_2217: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_193, 1, 0, 9223372036854775807)
    slice_scatter_422: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2217, copy_115, 2, 256, 9223372036854775807);  slice_2217 = copy_115 = None
    slice_scatter_423: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_193, slice_scatter_422, 1, 0, 9223372036854775807);  select_193 = slice_scatter_422 = None
    select_scatter_38: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2216, slice_scatter_423, 1, -1);  slice_2216 = slice_scatter_423 = None
    slice_scatter_424: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_421, select_scatter_38, 0, 0, 9223372036854775807);  slice_scatter_421 = select_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_2221: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_715, 0, 0, 9223372036854775807)
    slice_2222: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2221, 1, 0, 9223372036854775807);  slice_2221 = None
    slice_2223: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2222, 2, -257, -1);  slice_2222 = None
    slice_2224: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2223, 3, 257, 9223372036854775807);  slice_2223 = None
    slice_2229: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_424, 0, 0, 9223372036854775807)
    slice_2230: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2229, 1, 1, 9223372036854775807);  slice_2229 = None
    slice_2231: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2230, 2, 0, 9223372036854775807);  slice_2230 = None
    slice_2232: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2231, 3, 0, 256);  slice_2231 = None
    copy_116: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2232, slice_2224);  slice_2232 = slice_2224 = None
    slice_2233: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_424, 0, 0, 9223372036854775807)
    slice_2234: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2233, 1, 1, 9223372036854775807)
    slice_2235: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2234, 2, 0, 9223372036854775807)
    slice_scatter_425: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2235, copy_116, 3, 0, 256);  slice_2235 = copy_116 = None
    slice_scatter_426: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2234, slice_scatter_425, 2, 0, 9223372036854775807);  slice_2234 = slice_scatter_425 = None
    slice_scatter_427: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2233, slice_scatter_426, 1, 1, 9223372036854775807);  slice_2233 = slice_scatter_426 = None
    slice_scatter_428: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_424, slice_scatter_427, 0, 0, 9223372036854775807);  slice_scatter_424 = slice_scatter_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_2240: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_715, 0, 0, 9223372036854775807);  view_715 = None
    select_195: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_2240, 1, 0);  slice_2240 = None
    slice_2241: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_195, 1, 0, 255);  select_195 = None
    slice_2242: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2241, 2, -255, 9223372036854775807);  slice_2241 = None
    slice_2246: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_428, 0, 0, 9223372036854775807)
    select_197: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2246, 1, 0);  slice_2246 = None
    slice_2247: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_197, 1, 1, 256);  select_197 = None
    slice_2248: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2247, 2, 1, 256);  slice_2247 = None
    copy_117: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_2248, slice_2242);  slice_2248 = slice_2242 = None
    slice_2249: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_428, 0, 0, 9223372036854775807)
    select_198: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2249, 1, 0)
    slice_2250: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_198, 1, 1, 256)
    slice_scatter_429: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2250, copy_117, 2, 1, 256);  slice_2250 = copy_117 = None
    slice_scatter_430: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_198, slice_scatter_429, 1, 1, 256);  select_198 = slice_scatter_429 = None
    select_scatter_39: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2249, slice_scatter_430, 1, 0);  slice_2249 = slice_scatter_430 = None
    slice_scatter_431: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_428, select_scatter_39, 0, 0, 9223372036854775807);  slice_scatter_428 = select_scatter_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_87: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_38: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_182: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_38, -2);  iota_38 = None
    iota_39: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_183: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
    sub_75: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_182, unsqueeze_183);  unsqueeze_182 = unsqueeze_183 = None
    le_19: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_75, 0);  sub_75 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_76: "f32[256, 257]" = torch.ops.aten.where.self(le_19, full_87, scalar_tensor_38);  le_19 = full_87 = scalar_tensor_38 = None
    rev_38: "f32[256, 257]" = torch.ops.prims.rev.default(where_76, [0]);  where_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_184: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_38, 0);  rev_38 = None
    slice_2254: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_184, 1, 0, 9223372036854775807);  unsqueeze_184 = None
    unsqueeze_185: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_2254, 2);  slice_2254 = None
    slice_2255: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_185, 3, 0, 9223372036854775807);  unsqueeze_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_39: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_2255, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_38: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_2255, [1, 256, 1, 257]);  slice_2255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_718: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_431, [1, 1, 1024, 513])
    permute_602: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None
    slice_2260: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_602, 0, 0, 9223372036854775807);  permute_602 = None
    slice_2261: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2260, 1, 0, 256);  slice_2260 = None
    slice_2262: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2261, 2, 0, 9223372036854775807);  slice_2261 = None
    slice_2263: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2262, 3, 0, 257);  slice_2262 = None
    full_88: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_48: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_38, torch.bool);  expand_38 = None
    where_77: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_48, full_88, slice_2263);  convert_element_type_48 = full_88 = slice_2263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_719: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_431, [1, 1, 1024, 513])
    permute_603: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_719, [0, 2, 1, 3]);  view_719 = None
    slice_2268: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_603, 0, 0, 9223372036854775807);  permute_603 = None
    slice_2269: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2268, 1, 0, 256);  slice_2268 = None
    slice_2270: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2269, 2, 0, 9223372036854775807);  slice_2269 = None
    slice_2271: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2270, 3, 0, 257);  slice_2270 = None
    copy_118: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2271, where_77);  slice_2271 = where_77 = None
    view_720: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_431, [1, 1, 1024, 513]);  slice_scatter_431 = None
    permute_604: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
    slice_2272: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_604, 0, 0, 9223372036854775807)
    slice_2273: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2272, 1, 0, 256)
    slice_2274: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2273, 2, 0, 9223372036854775807)
    slice_scatter_432: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2274, copy_118, 3, 0, 257);  slice_2274 = copy_118 = None
    slice_scatter_433: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2273, slice_scatter_432, 2, 0, 9223372036854775807);  slice_2273 = slice_scatter_432 = None
    slice_scatter_434: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2272, slice_scatter_433, 1, 0, 256);  slice_2272 = slice_scatter_433 = None
    slice_scatter_435: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_604, slice_scatter_434, 0, 0, 9223372036854775807);  permute_604 = slice_scatter_434 = None
    permute_605: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_435, [0, 2, 1, 3]);  slice_scatter_435 = None
    view_721: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_605, [1, 4, 256, 513]);  permute_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_39: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_39, [1, 256, 1, 257]);  rev_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_723: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_721, [1, 1, 1024, 513])
    permute_607: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None
    slice_2283: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_607, 0, 0, 9223372036854775807);  permute_607 = None
    slice_2284: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2283, 1, -256, 9223372036854775807);  slice_2283 = None
    slice_2285: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2284, 2, 0, 9223372036854775807);  slice_2284 = None
    slice_2286: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2285, 3, -257, 9223372036854775807);  slice_2285 = None
    full_89: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_49: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_39, torch.bool);  expand_39 = None
    where_78: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_49, full_89, slice_2286);  convert_element_type_49 = full_89 = slice_2286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_724: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_721, [1, 1, 1024, 513])
    permute_608: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_724, [0, 2, 1, 3]);  view_724 = None
    slice_2291: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_608, 0, 0, 9223372036854775807);  permute_608 = None
    slice_2292: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2291, 1, -256, 9223372036854775807);  slice_2291 = None
    slice_2293: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2292, 2, 0, 9223372036854775807);  slice_2292 = None
    slice_2294: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2293, 3, -257, 9223372036854775807);  slice_2293 = None
    copy_119: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2294, where_78);  slice_2294 = where_78 = None
    view_725: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_721, [1, 1, 1024, 513]);  view_721 = None
    permute_609: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_725, [0, 2, 1, 3]);  view_725 = None
    slice_2295: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_609, 0, 0, 9223372036854775807)
    slice_2296: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2295, 1, -256, 9223372036854775807)
    slice_2297: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2296, 2, 0, 9223372036854775807)
    slice_scatter_436: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2297, copy_119, 3, -257, 9223372036854775807);  slice_2297 = copy_119 = None
    slice_scatter_437: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2296, slice_scatter_436, 2, 0, 9223372036854775807);  slice_2296 = slice_scatter_436 = None
    slice_scatter_438: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2295, slice_scatter_437, 1, -256, 9223372036854775807);  slice_2295 = slice_scatter_437 = None
    slice_scatter_439: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_609, slice_scatter_438, 0, 0, 9223372036854775807);  permute_609 = slice_scatter_438 = None
    permute_610: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_439, [0, 2, 1, 3]);  slice_scatter_439 = None
    view_726: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_610, [1, 4, 256, 513]);  permute_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_728: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_708, [1, 12, 1024, 513]);  view_708 = None
    permute_612: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_728, [0, 2, 1, 3]);  view_728 = None
    view_729: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_726, [1, 1, 1024, 513]);  view_726 = None
    permute_613: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_729, [0, 2, 1, 3]);  view_729 = None
    add_101: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_612, permute_613);  permute_612 = permute_613 = None
    permute_614: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_101, [0, 2, 1, 3]);  add_101 = None
    view_731: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_614, [12, 4, 256, 513]);  permute_614 = None
    view_732: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_731, [1, 12, 1024, 513]);  view_731 = None
    permute_615: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_74: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
    amax_9: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_74, [-1], True)
    sub_76: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_74, amax_9);  clone_74 = amax_9 = None
    exp_9: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
    sum_10: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_97: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_2302: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_2303: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_2302, 1, 0, 9223372036854775807);  slice_2302 = None
    unsqueeze_186: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_2303, 2);  slice_2303 = None
    unsqueeze_187: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_79: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_187, scalar_tensor_39, div_97);  unsqueeze_187 = scalar_tensor_39 = div_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_75: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_79);  where_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_733: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_680, [1024, 1, 12, 64]);  view_680 = None
    permute_616: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_733, [1, 0, 2, 3]);  view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_617: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_75, [0, 2, 1, 3]);  clone_75 = None
    view_734: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_617, [12, 4, 256, 513]);  permute_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_618: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_616, [0, 2, 1, 3]);  permute_616 = None
    view_735: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_618, [12, 1024, 64]);  permute_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_38: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_735, [0, 0, 256, 256], -1.0);  view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_59: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_38, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_39: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_734, [0, 257], 0.0);  view_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_736: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_39, [12, 4, -1]);  constant_pad_nd_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_2304: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_736, 0, 0, 9223372036854775807);  view_736 = None
    slice_2305: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_2304, 1, 0, 9223372036854775807);  slice_2304 = None
    slice_2306: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_2305, 2, 0, -256);  slice_2305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_737: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_2306, [12, 4, 256, 769]);  slice_2306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_2307: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_737, 0, 0, 9223372036854775807);  view_737 = None
    slice_2308: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2307, 1, 0, 9223372036854775807);  slice_2307 = None
    slice_2309: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2308, 2, 0, 9223372036854775807);  slice_2308 = None
    slice_2310: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_2309, 3, 0, -1);  slice_2309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_188: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2310, 4);  slice_2310 = None
    permute_619: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_188, [0, 1, 2, 4, 3]);  unsqueeze_188 = None
    unsqueeze_189: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_59, 4);  as_strided_59 = None
    permute_620: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_189, [0, 1, 4, 3, 2]);  unsqueeze_189 = None
    permute_621: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_619, [0, 1, 2, 4, 3]);  permute_619 = None
    view_738: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_621, [48, 256, 768]);  permute_621 = None
    permute_622: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_620, [0, 1, 4, 3, 2]);  permute_620 = None
    clone_76: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
    view_739: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_76, [48, 768, 64]);  clone_76 = None
    bmm_19: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_738, view_739);  view_738 = view_739 = None
    view_740: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_19, [12, 4, 256, 1, 64]);  bmm_19 = None
    permute_623: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_740, [0, 1, 2, 4, 3]);  view_740 = None
    view_741: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_623, [12, 4, 256, 64]);  permute_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_742: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_741, [1, 12, 1024, 64]);  view_741 = None
    permute_624: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_742, [0, 2, 1, 3]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_625: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_624, [1, 0, 2, 3]);  permute_624 = None
    clone_77: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_625, memory_format = torch.contiguous_format);  permute_625 = None
    view_743: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_77, [1024, 1, 768]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_626: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_743, [1, 0, 2]);  view_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_744: "f32[1024, 768]" = torch.ops.aten.view.default(permute_626, [1024, 768]);  permute_626 = None
    permute_627: "f32[768, 768]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_57: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg151_1, view_744, permute_627);  arg151_1 = view_744 = permute_627 = None
    view_745: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_57, [1, 1024, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_78: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_745);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_103: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_78, add_98);  clone_78 = add_98 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 1024, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    add_104: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_78: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_37);  add_103 = getitem_37 = None
    mul_73: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_18);  sub_78 = rsqrt_18 = None
    mul_74: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg152_1);  mul_73 = arg152_1 = None
    add_105: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_74, arg153_1);  mul_74 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_746: "f32[1024, 768]" = torch.ops.aten.view.default(add_105, [1024, 768])
    permute_628: "f32[768, 3072]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_58: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg155_1, view_746, permute_628);  arg155_1 = view_746 = permute_628 = None
    view_747: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_58, [1, 1024, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_747, 0.5)
    mul_76: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_747, 0.7071067811865476);  view_747 = None
    erf_9: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_106: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_77: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_106);  mul_75 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_748: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_77, [1024, 3072]);  mul_77 = None
    permute_629: "f32[3072, 768]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_59: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg157_1, view_748, permute_629);  arg157_1 = view_748 = permute_629 = None
    view_749: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_59, [1, 1024, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_79: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_749);  view_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_107: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_79, add_105);  clone_79 = add_105 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 1024, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    add_108: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_79: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_39);  add_107 = getitem_39 = None
    mul_78: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_19);  sub_79 = rsqrt_19 = None
    mul_79: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg158_1);  mul_78 = arg158_1 = None
    add_109: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_79, arg159_1);  mul_79 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_630: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_109, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_750: "f32[1024, 768]" = torch.ops.aten.view.default(permute_630, [1024, 768])
    permute_631: "f32[768, 768]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_60: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg161_1, view_750, permute_631);  arg161_1 = view_750 = permute_631 = None
    view_751: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_60, [1024, 1, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_752: "f32[1024, 768]" = torch.ops.aten.view.default(permute_630, [1024, 768])
    permute_632: "f32[768, 768]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    addmm_61: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg163_1, view_752, permute_632);  arg163_1 = view_752 = permute_632 = None
    view_753: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_61, [1024, 1, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_754: "f32[1024, 768]" = torch.ops.aten.view.default(permute_630, [1024, 768]);  permute_630 = None
    permute_633: "f32[768, 768]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_62: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg165_1, view_754, permute_633);  arg165_1 = view_754 = permute_633 = None
    view_755: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_62, [1024, 1, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_100: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_751, 8.0);  view_751 = None
    view_756: "f32[1024, 768]" = torch.ops.aten.view.default(div_100, [1024, 768]);  div_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_759: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_753, [1024, 1, 12, 64]);  view_753 = None
    permute_635: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_759, [1, 0, 2, 3]);  view_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_637: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_635, [0, 2, 1, 3]);  permute_635 = None
    view_761: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_637, [12, 1024, 64]);  permute_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_763: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_761, [12, 2, 512, 64]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_61: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_763, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_191: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_61, 4);  as_strided_61 = None
    permute_639: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 4, 2, 3]);  unsqueeze_191 = None
    view_764: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_756, [1024, 1, 768]);  view_756 = None
    view_765: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_764, [1024, 1, 12, 64]);  view_764 = None
    permute_641: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_765, [1, 0, 2, 3]);  view_765 = None
    permute_642: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_641, [0, 2, 1, 3]);  permute_641 = None
    view_766: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_642, [12, 1024, 64]);  permute_642 = None
    view_767: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_766, [12, 2, 512, 64]);  view_766 = None
    as_strided_62: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_767, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_767 = None
    unsqueeze_192: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_62, 4);  as_strided_62 = None
    permute_643: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_192, [0, 1, 2, 4, 3]);  unsqueeze_192 = None
    permute_644: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_643, [0, 1, 2, 4, 3]);  permute_643 = None
    clone_80: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
    view_768: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_80, [36, 512, 64]);  clone_80 = None
    permute_645: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_639, [0, 1, 4, 3, 2]);  permute_639 = None
    clone_81: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_645, memory_format = torch.contiguous_format);  permute_645 = None
    view_769: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_81, [36, 64, 512]);  clone_81 = None
    bmm_20: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_768, view_769);  view_768 = view_769 = None
    view_770: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_20, [12, 3, 512, 1, 512]);  bmm_20 = None
    permute_646: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_770, [0, 1, 2, 4, 3]);  view_770 = None
    view_771: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_646, [12, 3, 512, 512]);  permute_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_40: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_771, [0, 0, 0, 1], 0.0);  view_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_772: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_40, [12, 3, 512, 513]);  constant_pad_nd_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_90: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2311: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_772, 0, 0, 9223372036854775807)
    slice_2312: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2311, 1, 0, 9223372036854775807);  slice_2311 = None
    slice_2313: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2312, 2, 0, 256);  slice_2312 = None
    slice_2314: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2313, 3, 0, 257);  slice_2313 = None
    slice_2315: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_90, 0, 0, 9223372036854775807)
    slice_2316: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2315, 1, 0, -1);  slice_2315 = None
    slice_2317: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2316, 2, 0, 9223372036854775807);  slice_2316 = None
    slice_2318: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2317, 3, 256, 9223372036854775807);  slice_2317 = None
    copy_120: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2318, slice_2314);  slice_2318 = slice_2314 = None
    slice_2319: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_90, 0, 0, 9223372036854775807)
    slice_2320: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2319, 1, 0, -1)
    slice_2321: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2320, 2, 0, 9223372036854775807)
    slice_scatter_440: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2321, copy_120, 3, 256, 9223372036854775807);  slice_2321 = copy_120 = None
    slice_scatter_441: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2320, slice_scatter_440, 2, 0, 9223372036854775807);  slice_2320 = slice_scatter_440 = None
    slice_scatter_442: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2319, slice_scatter_441, 1, 0, -1);  slice_2319 = slice_scatter_441 = None
    slice_scatter_443: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_90, slice_scatter_442, 0, 0, 9223372036854775807);  full_90 = slice_scatter_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2326: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_772, 0, 0, 9223372036854775807)
    select_200: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2326, 1, -1);  slice_2326 = None
    slice_2327: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_200, 1, 256, 9223372036854775807);  select_200 = None
    slice_2328: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2327, 2, 0, 257);  slice_2327 = None
    slice_2332: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_443, 0, 0, 9223372036854775807)
    select_202: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2332, 1, -1);  slice_2332 = None
    slice_2333: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_202, 1, 0, 9223372036854775807);  select_202 = None
    slice_2334: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2333, 2, 256, 9223372036854775807);  slice_2333 = None
    copy_121: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_2334, slice_2328);  slice_2334 = slice_2328 = None
    slice_2335: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_443, 0, 0, 9223372036854775807)
    select_203: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2335, 1, -1)
    slice_2336: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_203, 1, 0, 9223372036854775807)
    slice_scatter_444: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2336, copy_121, 2, 256, 9223372036854775807);  slice_2336 = copy_121 = None
    slice_scatter_445: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_203, slice_scatter_444, 1, 0, 9223372036854775807);  select_203 = slice_scatter_444 = None
    select_scatter_40: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2335, slice_scatter_445, 1, -1);  slice_2335 = slice_scatter_445 = None
    slice_scatter_446: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_443, select_scatter_40, 0, 0, 9223372036854775807);  slice_scatter_443 = select_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_2340: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_772, 0, 0, 9223372036854775807)
    slice_2341: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2340, 1, 0, 9223372036854775807);  slice_2340 = None
    slice_2342: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2341, 2, -257, -1);  slice_2341 = None
    slice_2343: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2342, 3, 257, 9223372036854775807);  slice_2342 = None
    slice_2348: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_446, 0, 0, 9223372036854775807)
    slice_2349: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2348, 1, 1, 9223372036854775807);  slice_2348 = None
    slice_2350: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2349, 2, 0, 9223372036854775807);  slice_2349 = None
    slice_2351: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2350, 3, 0, 256);  slice_2350 = None
    copy_122: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2351, slice_2343);  slice_2351 = slice_2343 = None
    slice_2352: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_446, 0, 0, 9223372036854775807)
    slice_2353: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2352, 1, 1, 9223372036854775807)
    slice_2354: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2353, 2, 0, 9223372036854775807)
    slice_scatter_447: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2354, copy_122, 3, 0, 256);  slice_2354 = copy_122 = None
    slice_scatter_448: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2353, slice_scatter_447, 2, 0, 9223372036854775807);  slice_2353 = slice_scatter_447 = None
    slice_scatter_449: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2352, slice_scatter_448, 1, 1, 9223372036854775807);  slice_2352 = slice_scatter_448 = None
    slice_scatter_450: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_446, slice_scatter_449, 0, 0, 9223372036854775807);  slice_scatter_446 = slice_scatter_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_2359: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_772, 0, 0, 9223372036854775807);  view_772 = None
    select_205: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2359, 1, 0);  slice_2359 = None
    slice_2360: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_205, 1, 0, 255);  select_205 = None
    slice_2361: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2360, 2, -255, 9223372036854775807);  slice_2360 = None
    slice_2365: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_450, 0, 0, 9223372036854775807)
    select_207: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2365, 1, 0);  slice_2365 = None
    slice_2366: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_207, 1, 1, 256);  select_207 = None
    slice_2367: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2366, 2, 1, 256);  slice_2366 = None
    copy_123: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_2367, slice_2361);  slice_2367 = slice_2361 = None
    slice_2368: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_450, 0, 0, 9223372036854775807)
    select_208: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2368, 1, 0)
    slice_2369: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_208, 1, 1, 256)
    slice_scatter_451: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2369, copy_123, 2, 1, 256);  slice_2369 = copy_123 = None
    slice_scatter_452: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_208, slice_scatter_451, 1, 1, 256);  select_208 = slice_scatter_451 = None
    select_scatter_41: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2368, slice_scatter_452, 1, 0);  slice_2368 = slice_scatter_452 = None
    slice_scatter_453: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_450, select_scatter_41, 0, 0, 9223372036854775807);  slice_scatter_450 = select_scatter_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_91: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_40: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_193: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_40, -2);  iota_40 = None
    iota_41: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_194: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
    sub_81: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_193, unsqueeze_194);  unsqueeze_193 = unsqueeze_194 = None
    le_20: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_81, 0);  sub_81 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_80: "f32[256, 257]" = torch.ops.aten.where.self(le_20, full_91, scalar_tensor_40);  le_20 = full_91 = scalar_tensor_40 = None
    rev_40: "f32[256, 257]" = torch.ops.prims.rev.default(where_80, [0]);  where_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_195: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_40, 0);  rev_40 = None
    slice_2373: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_195, 1, 0, 9223372036854775807);  unsqueeze_195 = None
    unsqueeze_196: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_2373, 2);  slice_2373 = None
    slice_2374: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_196, 3, 0, 9223372036854775807);  unsqueeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_41: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_2374, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_40: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_2374, [1, 256, 12, 257]);  slice_2374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_775: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_453, [1, 12, 1024, 513])
    permute_649: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_775, [0, 2, 1, 3]);  view_775 = None
    slice_2379: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_649, 0, 0, 9223372036854775807);  permute_649 = None
    slice_2380: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2379, 1, 0, 256);  slice_2379 = None
    slice_2381: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2380, 2, 0, 9223372036854775807);  slice_2380 = None
    slice_2382: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2381, 3, 0, 257);  slice_2381 = None
    full_92: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_50: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_40, torch.bool);  expand_40 = None
    where_81: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_50, full_92, slice_2382);  convert_element_type_50 = full_92 = slice_2382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_776: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_453, [1, 12, 1024, 513])
    permute_650: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None
    slice_2387: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_650, 0, 0, 9223372036854775807);  permute_650 = None
    slice_2388: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2387, 1, 0, 256);  slice_2387 = None
    slice_2389: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2388, 2, 0, 9223372036854775807);  slice_2388 = None
    slice_2390: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2389, 3, 0, 257);  slice_2389 = None
    copy_124: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2390, where_81);  slice_2390 = where_81 = None
    view_777: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_453, [1, 12, 1024, 513]);  slice_scatter_453 = None
    permute_651: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_777, [0, 2, 1, 3]);  view_777 = None
    slice_2391: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_651, 0, 0, 9223372036854775807)
    slice_2392: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2391, 1, 0, 256)
    slice_2393: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2392, 2, 0, 9223372036854775807)
    slice_scatter_454: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2393, copy_124, 3, 0, 257);  slice_2393 = copy_124 = None
    slice_scatter_455: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2392, slice_scatter_454, 2, 0, 9223372036854775807);  slice_2392 = slice_scatter_454 = None
    slice_scatter_456: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2391, slice_scatter_455, 1, 0, 256);  slice_2391 = slice_scatter_455 = None
    slice_scatter_457: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_651, slice_scatter_456, 0, 0, 9223372036854775807);  permute_651 = slice_scatter_456 = None
    permute_652: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_457, [0, 2, 1, 3]);  slice_scatter_457 = None
    view_778: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_652, [12, 4, 256, 513]);  permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_41: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_41, [1, 256, 12, 257]);  rev_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_780: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_778, [1, 12, 1024, 513])
    permute_654: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
    slice_2402: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_654, 0, 0, 9223372036854775807);  permute_654 = None
    slice_2403: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2402, 1, -256, 9223372036854775807);  slice_2402 = None
    slice_2404: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2403, 2, 0, 9223372036854775807);  slice_2403 = None
    slice_2405: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2404, 3, -257, 9223372036854775807);  slice_2404 = None
    full_93: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_51: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_41, torch.bool);  expand_41 = None
    where_82: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_51, full_93, slice_2405);  convert_element_type_51 = full_93 = slice_2405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_781: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_778, [1, 12, 1024, 513])
    permute_655: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_781, [0, 2, 1, 3]);  view_781 = None
    slice_2410: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_655, 0, 0, 9223372036854775807);  permute_655 = None
    slice_2411: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2410, 1, -256, 9223372036854775807);  slice_2410 = None
    slice_2412: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2411, 2, 0, 9223372036854775807);  slice_2411 = None
    slice_2413: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2412, 3, -257, 9223372036854775807);  slice_2412 = None
    copy_125: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2413, where_82);  slice_2413 = where_82 = None
    view_782: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_778, [1, 12, 1024, 513]);  view_778 = None
    permute_656: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_782, [0, 2, 1, 3]);  view_782 = None
    slice_2414: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_656, 0, 0, 9223372036854775807)
    slice_2415: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2414, 1, -256, 9223372036854775807)
    slice_2416: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2415, 2, 0, 9223372036854775807)
    slice_scatter_458: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2416, copy_125, 3, -257, 9223372036854775807);  slice_2416 = copy_125 = None
    slice_scatter_459: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2415, slice_scatter_458, 2, 0, 9223372036854775807);  slice_2415 = slice_scatter_458 = None
    slice_scatter_460: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2414, slice_scatter_459, 1, -256, 9223372036854775807);  slice_2414 = slice_scatter_459 = None
    slice_scatter_461: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_656, slice_scatter_460, 0, 0, 9223372036854775807);  permute_656 = slice_scatter_460 = None
    permute_657: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_461, [0, 2, 1, 3]);  slice_scatter_461 = None
    view_783: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_657, [12, 4, 256, 513]);  permute_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_10: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0)
    slice_2421: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_10, 0, 0, 9223372036854775807);  ne_10 = None
    slice_2422: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_2421, 1, 0, 9223372036854775807);  slice_2421 = None
    unsqueeze_197: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_2422, 2);  slice_2422 = None
    unsqueeze_198: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_52: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_198, torch.float32)
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_83: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_198, scalar_tensor_41, convert_element_type_52);  unsqueeze_198 = scalar_tensor_41 = convert_element_type_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_94: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_659: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_94, [0, 2, 1, 3]);  full_94 = None
    view_785: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_659, [1, 1024, 1]);  permute_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_660: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_83, [0, 2, 1, 3]);  where_83 = None
    view_786: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_660, [1, 1024, 1]);  permute_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_787: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_785, [1, 2, 512, 1]);  view_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_63: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_787, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_788: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_786, [1, 2, 512, 1]);  view_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_64: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_788, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_199: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_63, 4);  as_strided_63 = None
    permute_661: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_199, [0, 1, 2, 4, 3]);  unsqueeze_199 = None
    unsqueeze_200: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_64, 4);  as_strided_64 = None
    permute_662: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 4, 2, 3]);  unsqueeze_200 = None
    mul_80: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_661, permute_662);  permute_661 = permute_662 = None
    view_789: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_80, [1, 3, 512, 512]);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_41: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_789, [0, 0, 0, 1], 0.0);  view_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_790: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_41, [1, 3, 512, 513]);  constant_pad_nd_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_95: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2423: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_790, 0, 0, 9223372036854775807)
    slice_2424: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2423, 1, 0, 9223372036854775807);  slice_2423 = None
    slice_2425: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2424, 2, 0, 256);  slice_2424 = None
    slice_2426: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2425, 3, 0, 257);  slice_2425 = None
    slice_2427: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_95, 0, 0, 9223372036854775807)
    slice_2428: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2427, 1, 0, -1);  slice_2427 = None
    slice_2429: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2428, 2, 0, 9223372036854775807);  slice_2428 = None
    slice_2430: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2429, 3, 256, 9223372036854775807);  slice_2429 = None
    copy_126: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2430, slice_2426);  slice_2430 = slice_2426 = None
    slice_2431: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_95, 0, 0, 9223372036854775807)
    slice_2432: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2431, 1, 0, -1)
    slice_2433: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2432, 2, 0, 9223372036854775807)
    slice_scatter_462: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2433, copy_126, 3, 256, 9223372036854775807);  slice_2433 = copy_126 = None
    slice_scatter_463: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2432, slice_scatter_462, 2, 0, 9223372036854775807);  slice_2432 = slice_scatter_462 = None
    slice_scatter_464: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2431, slice_scatter_463, 1, 0, -1);  slice_2431 = slice_scatter_463 = None
    slice_scatter_465: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_95, slice_scatter_464, 0, 0, 9223372036854775807);  full_95 = slice_scatter_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2438: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_790, 0, 0, 9223372036854775807)
    select_210: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_2438, 1, -1);  slice_2438 = None
    slice_2439: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_210, 1, 256, 9223372036854775807);  select_210 = None
    slice_2440: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2439, 2, 0, 257);  slice_2439 = None
    slice_2444: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_465, 0, 0, 9223372036854775807)
    select_212: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2444, 1, -1);  slice_2444 = None
    slice_2445: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_212, 1, 0, 9223372036854775807);  select_212 = None
    slice_2446: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2445, 2, 256, 9223372036854775807);  slice_2445 = None
    copy_127: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_2446, slice_2440);  slice_2446 = slice_2440 = None
    slice_2447: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_465, 0, 0, 9223372036854775807)
    select_213: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2447, 1, -1)
    slice_2448: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_213, 1, 0, 9223372036854775807)
    slice_scatter_466: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2448, copy_127, 2, 256, 9223372036854775807);  slice_2448 = copy_127 = None
    slice_scatter_467: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_213, slice_scatter_466, 1, 0, 9223372036854775807);  select_213 = slice_scatter_466 = None
    select_scatter_42: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2447, slice_scatter_467, 1, -1);  slice_2447 = slice_scatter_467 = None
    slice_scatter_468: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_465, select_scatter_42, 0, 0, 9223372036854775807);  slice_scatter_465 = select_scatter_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_2452: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_790, 0, 0, 9223372036854775807)
    slice_2453: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2452, 1, 0, 9223372036854775807);  slice_2452 = None
    slice_2454: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2453, 2, -257, -1);  slice_2453 = None
    slice_2455: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2454, 3, 257, 9223372036854775807);  slice_2454 = None
    slice_2460: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_468, 0, 0, 9223372036854775807)
    slice_2461: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2460, 1, 1, 9223372036854775807);  slice_2460 = None
    slice_2462: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2461, 2, 0, 9223372036854775807);  slice_2461 = None
    slice_2463: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2462, 3, 0, 256);  slice_2462 = None
    copy_128: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2463, slice_2455);  slice_2463 = slice_2455 = None
    slice_2464: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_468, 0, 0, 9223372036854775807)
    slice_2465: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2464, 1, 1, 9223372036854775807)
    slice_2466: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2465, 2, 0, 9223372036854775807)
    slice_scatter_469: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2466, copy_128, 3, 0, 256);  slice_2466 = copy_128 = None
    slice_scatter_470: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2465, slice_scatter_469, 2, 0, 9223372036854775807);  slice_2465 = slice_scatter_469 = None
    slice_scatter_471: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2464, slice_scatter_470, 1, 1, 9223372036854775807);  slice_2464 = slice_scatter_470 = None
    slice_scatter_472: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_468, slice_scatter_471, 0, 0, 9223372036854775807);  slice_scatter_468 = slice_scatter_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_2471: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_790, 0, 0, 9223372036854775807);  view_790 = None
    select_215: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_2471, 1, 0);  slice_2471 = None
    slice_2472: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_215, 1, 0, 255);  select_215 = None
    slice_2473: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2472, 2, -255, 9223372036854775807);  slice_2472 = None
    slice_2477: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_472, 0, 0, 9223372036854775807)
    select_217: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2477, 1, 0);  slice_2477 = None
    slice_2478: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_217, 1, 1, 256);  select_217 = None
    slice_2479: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2478, 2, 1, 256);  slice_2478 = None
    copy_129: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_2479, slice_2473);  slice_2479 = slice_2473 = None
    slice_2480: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_472, 0, 0, 9223372036854775807)
    select_218: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2480, 1, 0)
    slice_2481: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_218, 1, 1, 256)
    slice_scatter_473: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2481, copy_129, 2, 1, 256);  slice_2481 = copy_129 = None
    slice_scatter_474: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_218, slice_scatter_473, 1, 1, 256);  select_218 = slice_scatter_473 = None
    select_scatter_43: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2480, slice_scatter_474, 1, 0);  slice_2480 = slice_scatter_474 = None
    slice_scatter_475: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_472, select_scatter_43, 0, 0, 9223372036854775807);  slice_scatter_472 = select_scatter_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_96: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_42: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_201: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_42, -2);  iota_42 = None
    iota_43: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_202: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
    sub_83: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_201, unsqueeze_202);  unsqueeze_201 = unsqueeze_202 = None
    le_21: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_83, 0);  sub_83 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_84: "f32[256, 257]" = torch.ops.aten.where.self(le_21, full_96, scalar_tensor_42);  le_21 = full_96 = scalar_tensor_42 = None
    rev_42: "f32[256, 257]" = torch.ops.prims.rev.default(where_84, [0]);  where_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_203: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_42, 0);  rev_42 = None
    slice_2485: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_203, 1, 0, 9223372036854775807);  unsqueeze_203 = None
    unsqueeze_204: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_2485, 2);  slice_2485 = None
    slice_2486: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_204, 3, 0, 9223372036854775807);  unsqueeze_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_43: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_2486, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_42: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_2486, [1, 256, 1, 257]);  slice_2486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_793: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_475, [1, 1, 1024, 513])
    permute_665: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_793, [0, 2, 1, 3]);  view_793 = None
    slice_2491: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_665, 0, 0, 9223372036854775807);  permute_665 = None
    slice_2492: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2491, 1, 0, 256);  slice_2491 = None
    slice_2493: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2492, 2, 0, 9223372036854775807);  slice_2492 = None
    slice_2494: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2493, 3, 0, 257);  slice_2493 = None
    full_97: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_53: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_42, torch.bool);  expand_42 = None
    where_85: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_53, full_97, slice_2494);  convert_element_type_53 = full_97 = slice_2494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_794: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_475, [1, 1, 1024, 513])
    permute_666: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_794, [0, 2, 1, 3]);  view_794 = None
    slice_2499: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_666, 0, 0, 9223372036854775807);  permute_666 = None
    slice_2500: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2499, 1, 0, 256);  slice_2499 = None
    slice_2501: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2500, 2, 0, 9223372036854775807);  slice_2500 = None
    slice_2502: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2501, 3, 0, 257);  slice_2501 = None
    copy_130: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2502, where_85);  slice_2502 = where_85 = None
    view_795: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_475, [1, 1, 1024, 513]);  slice_scatter_475 = None
    permute_667: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_795, [0, 2, 1, 3]);  view_795 = None
    slice_2503: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_667, 0, 0, 9223372036854775807)
    slice_2504: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2503, 1, 0, 256)
    slice_2505: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2504, 2, 0, 9223372036854775807)
    slice_scatter_476: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2505, copy_130, 3, 0, 257);  slice_2505 = copy_130 = None
    slice_scatter_477: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2504, slice_scatter_476, 2, 0, 9223372036854775807);  slice_2504 = slice_scatter_476 = None
    slice_scatter_478: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2503, slice_scatter_477, 1, 0, 256);  slice_2503 = slice_scatter_477 = None
    slice_scatter_479: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_667, slice_scatter_478, 0, 0, 9223372036854775807);  permute_667 = slice_scatter_478 = None
    permute_668: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_479, [0, 2, 1, 3]);  slice_scatter_479 = None
    view_796: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_668, [1, 4, 256, 513]);  permute_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_43: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_43, [1, 256, 1, 257]);  rev_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_798: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_796, [1, 1, 1024, 513])
    permute_670: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_798, [0, 2, 1, 3]);  view_798 = None
    slice_2514: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_670, 0, 0, 9223372036854775807);  permute_670 = None
    slice_2515: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2514, 1, -256, 9223372036854775807);  slice_2514 = None
    slice_2516: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2515, 2, 0, 9223372036854775807);  slice_2515 = None
    slice_2517: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2516, 3, -257, 9223372036854775807);  slice_2516 = None
    full_98: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_54: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_43, torch.bool);  expand_43 = None
    where_86: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_54, full_98, slice_2517);  convert_element_type_54 = full_98 = slice_2517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_799: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_796, [1, 1, 1024, 513])
    permute_671: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_799, [0, 2, 1, 3]);  view_799 = None
    slice_2522: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_671, 0, 0, 9223372036854775807);  permute_671 = None
    slice_2523: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2522, 1, -256, 9223372036854775807);  slice_2522 = None
    slice_2524: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2523, 2, 0, 9223372036854775807);  slice_2523 = None
    slice_2525: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2524, 3, -257, 9223372036854775807);  slice_2524 = None
    copy_131: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2525, where_86);  slice_2525 = where_86 = None
    view_800: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_796, [1, 1, 1024, 513]);  view_796 = None
    permute_672: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_800, [0, 2, 1, 3]);  view_800 = None
    slice_2526: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_672, 0, 0, 9223372036854775807)
    slice_2527: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2526, 1, -256, 9223372036854775807)
    slice_2528: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2527, 2, 0, 9223372036854775807)
    slice_scatter_480: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2528, copy_131, 3, -257, 9223372036854775807);  slice_2528 = copy_131 = None
    slice_scatter_481: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2527, slice_scatter_480, 2, 0, 9223372036854775807);  slice_2527 = slice_scatter_480 = None
    slice_scatter_482: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2526, slice_scatter_481, 1, -256, 9223372036854775807);  slice_2526 = slice_scatter_481 = None
    slice_scatter_483: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_672, slice_scatter_482, 0, 0, 9223372036854775807);  permute_672 = slice_scatter_482 = None
    permute_673: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_483, [0, 2, 1, 3]);  slice_scatter_483 = None
    view_801: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_673, [1, 4, 256, 513]);  permute_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_803: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_783, [1, 12, 1024, 513]);  view_783 = None
    permute_675: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_803, [0, 2, 1, 3]);  view_803 = None
    view_804: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_801, [1, 1, 1024, 513]);  view_801 = None
    permute_676: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_804, [0, 2, 1, 3]);  view_804 = None
    add_112: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_675, permute_676);  permute_675 = permute_676 = None
    permute_677: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_112, [0, 2, 1, 3]);  add_112 = None
    view_806: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_677, [12, 4, 256, 513]);  permute_677 = None
    view_807: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_806, [1, 12, 1024, 513]);  view_806 = None
    permute_678: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_807, [0, 2, 1, 3]);  view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_82: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_678, memory_format = torch.contiguous_format);  permute_678 = None
    amax_10: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_82, [-1], True)
    sub_84: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_82, amax_10);  clone_82 = amax_10 = None
    exp_10: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
    sum_11: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_107: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_2533: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807)
    slice_2534: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_2533, 1, 0, 9223372036854775807);  slice_2533 = None
    unsqueeze_205: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_2534, 2);  slice_2534 = None
    unsqueeze_206: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_87: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_206, scalar_tensor_43, div_107);  unsqueeze_206 = scalar_tensor_43 = div_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_83: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_87);  where_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_808: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_755, [1024, 1, 12, 64]);  view_755 = None
    permute_679: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_808, [1, 0, 2, 3]);  view_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_680: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_83, [0, 2, 1, 3]);  clone_83 = None
    view_809: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_680, [12, 4, 256, 513]);  permute_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_681: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_679, [0, 2, 1, 3]);  permute_679 = None
    view_810: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_681, [12, 1024, 64]);  permute_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_42: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_810, [0, 0, 256, 256], -1.0);  view_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_65: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_42, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_43: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_809, [0, 257], 0.0);  view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_811: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_43, [12, 4, -1]);  constant_pad_nd_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_2535: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_811, 0, 0, 9223372036854775807);  view_811 = None
    slice_2536: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_2535, 1, 0, 9223372036854775807);  slice_2535 = None
    slice_2537: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_2536, 2, 0, -256);  slice_2536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_812: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_2537, [12, 4, 256, 769]);  slice_2537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_2538: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_812, 0, 0, 9223372036854775807);  view_812 = None
    slice_2539: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2538, 1, 0, 9223372036854775807);  slice_2538 = None
    slice_2540: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2539, 2, 0, 9223372036854775807);  slice_2539 = None
    slice_2541: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_2540, 3, 0, -1);  slice_2540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_207: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2541, 4);  slice_2541 = None
    permute_682: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_207, [0, 1, 2, 4, 3]);  unsqueeze_207 = None
    unsqueeze_208: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_65, 4);  as_strided_65 = None
    permute_683: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_208, [0, 1, 4, 3, 2]);  unsqueeze_208 = None
    permute_684: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_682, [0, 1, 2, 4, 3]);  permute_682 = None
    view_813: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_684, [48, 256, 768]);  permute_684 = None
    permute_685: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_683, [0, 1, 4, 3, 2]);  permute_683 = None
    clone_84: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
    view_814: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_84, [48, 768, 64]);  clone_84 = None
    bmm_21: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_813, view_814);  view_813 = view_814 = None
    view_815: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_21, [12, 4, 256, 1, 64]);  bmm_21 = None
    permute_686: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_815, [0, 1, 2, 4, 3]);  view_815 = None
    view_816: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_686, [12, 4, 256, 64]);  permute_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_817: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_816, [1, 12, 1024, 64]);  view_816 = None
    permute_687: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_817, [0, 2, 1, 3]);  view_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_688: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_687, [1, 0, 2, 3]);  permute_687 = None
    clone_85: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_688, memory_format = torch.contiguous_format);  permute_688 = None
    view_818: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_85, [1024, 1, 768]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_689: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_818, [1, 0, 2]);  view_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_819: "f32[1024, 768]" = torch.ops.aten.view.default(permute_689, [1024, 768]);  permute_689 = None
    permute_690: "f32[768, 768]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_63: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg167_1, view_819, permute_690);  arg167_1 = view_819 = permute_690 = None
    view_820: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_63, [1, 1024, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_86: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_820);  view_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_114: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_86, add_109);  clone_86 = add_109 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 1024, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    add_115: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_86: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_41);  add_114 = getitem_41 = None
    mul_81: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_20);  sub_86 = rsqrt_20 = None
    mul_82: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg168_1);  mul_81 = arg168_1 = None
    add_116: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_82, arg169_1);  mul_82 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_821: "f32[1024, 768]" = torch.ops.aten.view.default(add_116, [1024, 768])
    permute_691: "f32[768, 3072]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_64: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg171_1, view_821, permute_691);  arg171_1 = view_821 = permute_691 = None
    view_822: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_64, [1, 1024, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_822, 0.5)
    mul_84: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_822, 0.7071067811865476);  view_822 = None
    erf_10: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
    add_117: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_85: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_83, add_117);  mul_83 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_823: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_85, [1024, 3072]);  mul_85 = None
    permute_692: "f32[3072, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    addmm_65: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg173_1, view_823, permute_692);  arg173_1 = view_823 = permute_692 = None
    view_824: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_65, [1, 1024, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_87: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_824);  view_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_118: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_87, add_116);  clone_87 = add_116 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 1024, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    add_119: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_87: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_118, getitem_43);  add_118 = getitem_43 = None
    mul_86: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_21);  sub_87 = rsqrt_21 = None
    mul_87: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_86, arg174_1);  mul_86 = arg174_1 = None
    add_120: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_87, arg175_1);  mul_87 = arg175_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_693: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(add_120, [1, 0, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    view_825: "f32[1024, 768]" = torch.ops.aten.view.default(permute_693, [1024, 768])
    permute_694: "f32[768, 768]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_66: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg177_1, view_825, permute_694);  arg177_1 = view_825 = permute_694 = None
    view_826: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_66, [1024, 1, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    view_827: "f32[1024, 768]" = torch.ops.aten.view.default(permute_693, [1024, 768])
    permute_695: "f32[768, 768]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_67: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg179_1, view_827, permute_695);  arg179_1 = view_827 = permute_695 = None
    view_828: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_67, [1024, 1, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_829: "f32[1024, 768]" = torch.ops.aten.view.default(permute_693, [1024, 768]);  permute_693 = None
    permute_696: "f32[768, 768]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_68: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg181_1, view_829, permute_696);  arg181_1 = view_829 = permute_696 = None
    view_830: "f32[1024, 1, 768]" = torch.ops.aten.view.default(addmm_68, [1024, 1, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    div_110: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_826, 8.0);  view_826 = None
    view_831: "f32[1024, 768]" = torch.ops.aten.view.default(div_110, [1024, 768]);  div_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:569, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_834: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_828, [1024, 1, 12, 64]);  view_828 = None
    permute_698: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_834, [1, 0, 2, 3]);  view_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_700: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_698, [0, 2, 1, 3]);  permute_698 = None
    view_836: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_700, [12, 1024, 64]);  permute_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_838: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_836, [12, 2, 512, 64]);  view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_67: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_838, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_210: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_67, 4);  as_strided_67 = None
    permute_702: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_210, [0, 1, 4, 2, 3]);  unsqueeze_210 = None
    view_839: "f32[1024, 1, 768]" = torch.ops.aten.view.default(view_831, [1024, 1, 768]);  view_831 = None
    view_840: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_839, [1024, 1, 12, 64]);  view_839 = None
    permute_704: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_840, [1, 0, 2, 3]);  view_840 = None
    permute_705: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_704, [0, 2, 1, 3]);  permute_704 = None
    view_841: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_705, [12, 1024, 64]);  permute_705 = None
    view_842: "f32[12, 2, 512, 64]" = torch.ops.aten.view.default(view_841, [12, 2, 512, 64]);  view_841 = None
    as_strided_68: "f32[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_842, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_842 = None
    unsqueeze_211: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_68, 4);  as_strided_68 = None
    permute_706: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_211, [0, 1, 2, 4, 3]);  unsqueeze_211 = None
    permute_707: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_706, [0, 1, 2, 4, 3]);  permute_706 = None
    clone_88: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
    view_843: "f32[36, 512, 64]" = torch.ops.aten.view.default(clone_88, [36, 512, 64]);  clone_88 = None
    permute_708: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_702, [0, 1, 4, 3, 2]);  permute_702 = None
    clone_89: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_708, memory_format = torch.contiguous_format);  permute_708 = None
    view_844: "f32[36, 64, 512]" = torch.ops.aten.view.default(clone_89, [36, 64, 512]);  clone_89 = None
    bmm_22: "f32[36, 512, 512]" = torch.ops.aten.bmm.default(view_843, view_844);  view_843 = view_844 = None
    view_845: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_22, [12, 3, 512, 1, 512]);  bmm_22 = None
    permute_709: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_845, [0, 1, 2, 4, 3]);  view_845 = None
    view_846: "f32[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_709, [12, 3, 512, 512]);  permute_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_44: "f32[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_846, [0, 0, 0, 1], 0.0);  view_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_847: "f32[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_44, [12, 3, 512, 513]);  constant_pad_nd_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_99: "f32[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2542: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_847, 0, 0, 9223372036854775807)
    slice_2543: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2542, 1, 0, 9223372036854775807);  slice_2542 = None
    slice_2544: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2543, 2, 0, 256);  slice_2543 = None
    slice_2545: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2544, 3, 0, 257);  slice_2544 = None
    slice_2546: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_99, 0, 0, 9223372036854775807)
    slice_2547: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2546, 1, 0, -1);  slice_2546 = None
    slice_2548: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2547, 2, 0, 9223372036854775807);  slice_2547 = None
    slice_2549: "f32[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2548, 3, 256, 9223372036854775807);  slice_2548 = None
    copy_132: "f32[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2549, slice_2545);  slice_2549 = slice_2545 = None
    slice_2550: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_99, 0, 0, 9223372036854775807)
    slice_2551: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2550, 1, 0, -1)
    slice_2552: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2551, 2, 0, 9223372036854775807)
    slice_scatter_484: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2552, copy_132, 3, 256, 9223372036854775807);  slice_2552 = copy_132 = None
    slice_scatter_485: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2551, slice_scatter_484, 2, 0, 9223372036854775807);  slice_2551 = slice_scatter_484 = None
    slice_scatter_486: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2550, slice_scatter_485, 1, 0, -1);  slice_2550 = slice_scatter_485 = None
    slice_scatter_487: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_99, slice_scatter_486, 0, 0, 9223372036854775807);  full_99 = slice_scatter_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2557: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_847, 0, 0, 9223372036854775807)
    select_220: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2557, 1, -1);  slice_2557 = None
    slice_2558: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_220, 1, 256, 9223372036854775807);  select_220 = None
    slice_2559: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2558, 2, 0, 257);  slice_2558 = None
    slice_2563: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_487, 0, 0, 9223372036854775807)
    select_222: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2563, 1, -1);  slice_2563 = None
    slice_2564: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_222, 1, 0, 9223372036854775807);  select_222 = None
    slice_2565: "f32[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2564, 2, 256, 9223372036854775807);  slice_2564 = None
    copy_133: "f32[12, 256, 257]" = torch.ops.aten.copy.default(slice_2565, slice_2559);  slice_2565 = slice_2559 = None
    slice_2566: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_487, 0, 0, 9223372036854775807)
    select_223: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2566, 1, -1)
    slice_2567: "f32[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_223, 1, 0, 9223372036854775807)
    slice_scatter_488: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2567, copy_133, 2, 256, 9223372036854775807);  slice_2567 = copy_133 = None
    slice_scatter_489: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_223, slice_scatter_488, 1, 0, 9223372036854775807);  select_223 = slice_scatter_488 = None
    select_scatter_44: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2566, slice_scatter_489, 1, -1);  slice_2566 = slice_scatter_489 = None
    slice_scatter_490: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_487, select_scatter_44, 0, 0, 9223372036854775807);  slice_scatter_487 = select_scatter_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_2571: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_847, 0, 0, 9223372036854775807)
    slice_2572: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2571, 1, 0, 9223372036854775807);  slice_2571 = None
    slice_2573: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2572, 2, -257, -1);  slice_2572 = None
    slice_2574: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2573, 3, 257, 9223372036854775807);  slice_2573 = None
    slice_2579: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_490, 0, 0, 9223372036854775807)
    slice_2580: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2579, 1, 1, 9223372036854775807);  slice_2579 = None
    slice_2581: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2580, 2, 0, 9223372036854775807);  slice_2580 = None
    slice_2582: "f32[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2581, 3, 0, 256);  slice_2581 = None
    copy_134: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2582, slice_2574);  slice_2582 = slice_2574 = None
    slice_2583: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_490, 0, 0, 9223372036854775807)
    slice_2584: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2583, 1, 1, 9223372036854775807)
    slice_2585: "f32[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2584, 2, 0, 9223372036854775807)
    slice_scatter_491: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2585, copy_134, 3, 0, 256);  slice_2585 = copy_134 = None
    slice_scatter_492: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2584, slice_scatter_491, 2, 0, 9223372036854775807);  slice_2584 = slice_scatter_491 = None
    slice_scatter_493: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2583, slice_scatter_492, 1, 1, 9223372036854775807);  slice_2583 = slice_scatter_492 = None
    slice_scatter_494: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_490, slice_scatter_493, 0, 0, 9223372036854775807);  slice_scatter_490 = slice_scatter_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_2590: "f32[12, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_847, 0, 0, 9223372036854775807);  view_847 = None
    select_225: "f32[12, 512, 513]" = torch.ops.aten.select.int(slice_2590, 1, 0);  slice_2590 = None
    slice_2591: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_225, 1, 0, 255);  select_225 = None
    slice_2592: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2591, 2, -255, 9223372036854775807);  slice_2591 = None
    slice_2596: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_494, 0, 0, 9223372036854775807)
    select_227: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2596, 1, 0);  slice_2596 = None
    slice_2597: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_227, 1, 1, 256);  select_227 = None
    slice_2598: "f32[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2597, 2, 1, 256);  slice_2597 = None
    copy_135: "f32[12, 255, 255]" = torch.ops.aten.copy.default(slice_2598, slice_2592);  slice_2598 = slice_2592 = None
    slice_2599: "f32[12, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_494, 0, 0, 9223372036854775807)
    select_228: "f32[12, 256, 513]" = torch.ops.aten.select.int(slice_2599, 1, 0)
    slice_2600: "f32[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_228, 1, 1, 256)
    slice_scatter_495: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2600, copy_135, 2, 1, 256);  slice_2600 = copy_135 = None
    slice_scatter_496: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_228, slice_scatter_495, 1, 1, 256);  select_228 = slice_scatter_495 = None
    select_scatter_45: "f32[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2599, slice_scatter_496, 1, 0);  slice_2599 = slice_scatter_496 = None
    slice_scatter_497: "f32[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_494, select_scatter_45, 0, 0, 9223372036854775807);  slice_scatter_494 = select_scatter_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_100: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_44: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_212: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_44, -2);  iota_44 = None
    iota_45: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_213: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
    sub_89: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_212, unsqueeze_213);  unsqueeze_212 = unsqueeze_213 = None
    le_22: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_89, 0);  sub_89 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_88: "f32[256, 257]" = torch.ops.aten.where.self(le_22, full_100, scalar_tensor_44);  le_22 = full_100 = scalar_tensor_44 = None
    rev_44: "f32[256, 257]" = torch.ops.prims.rev.default(where_88, [0]);  where_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_214: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_44, 0);  rev_44 = None
    slice_2604: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_214, 1, 0, 9223372036854775807);  unsqueeze_214 = None
    unsqueeze_215: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_2604, 2);  slice_2604 = None
    slice_2605: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_215, 3, 0, 9223372036854775807);  unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_45: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_2605, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_44: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_2605, [1, 256, 12, 257]);  slice_2605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_850: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_497, [1, 12, 1024, 513])
    permute_712: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_850, [0, 2, 1, 3]);  view_850 = None
    slice_2610: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_712, 0, 0, 9223372036854775807);  permute_712 = None
    slice_2611: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2610, 1, 0, 256);  slice_2610 = None
    slice_2612: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2611, 2, 0, 9223372036854775807);  slice_2611 = None
    slice_2613: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2612, 3, 0, 257);  slice_2612 = None
    full_101: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_55: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_44, torch.bool);  expand_44 = None
    where_89: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_55, full_101, slice_2613);  convert_element_type_55 = full_101 = slice_2613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_851: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_497, [1, 12, 1024, 513])
    permute_713: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_851, [0, 2, 1, 3]);  view_851 = None
    slice_2618: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_713, 0, 0, 9223372036854775807);  permute_713 = None
    slice_2619: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2618, 1, 0, 256);  slice_2618 = None
    slice_2620: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2619, 2, 0, 9223372036854775807);  slice_2619 = None
    slice_2621: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2620, 3, 0, 257);  slice_2620 = None
    copy_136: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2621, where_89);  slice_2621 = where_89 = None
    view_852: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_497, [1, 12, 1024, 513]);  slice_scatter_497 = None
    permute_714: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_852, [0, 2, 1, 3]);  view_852 = None
    slice_2622: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_714, 0, 0, 9223372036854775807)
    slice_2623: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2622, 1, 0, 256)
    slice_2624: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2623, 2, 0, 9223372036854775807)
    slice_scatter_498: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2624, copy_136, 3, 0, 257);  slice_2624 = copy_136 = None
    slice_scatter_499: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2623, slice_scatter_498, 2, 0, 9223372036854775807);  slice_2623 = slice_scatter_498 = None
    slice_scatter_500: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2622, slice_scatter_499, 1, 0, 256);  slice_2622 = slice_scatter_499 = None
    slice_scatter_501: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_714, slice_scatter_500, 0, 0, 9223372036854775807);  permute_714 = slice_scatter_500 = None
    permute_715: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_501, [0, 2, 1, 3]);  slice_scatter_501 = None
    view_853: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_715, [12, 4, 256, 513]);  permute_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_45: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_45, [1, 256, 12, 257]);  rev_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_855: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_853, [1, 12, 1024, 513])
    permute_717: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_855, [0, 2, 1, 3]);  view_855 = None
    slice_2633: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_717, 0, 0, 9223372036854775807);  permute_717 = None
    slice_2634: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2633, 1, -256, 9223372036854775807);  slice_2633 = None
    slice_2635: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2634, 2, 0, 9223372036854775807);  slice_2634 = None
    slice_2636: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2635, 3, -257, 9223372036854775807);  slice_2635 = None
    full_102: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_56: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_45, torch.bool);  expand_45 = None
    where_90: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_56, full_102, slice_2636);  convert_element_type_56 = full_102 = slice_2636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_856: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_853, [1, 12, 1024, 513])
    permute_718: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_856, [0, 2, 1, 3]);  view_856 = None
    slice_2641: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_718, 0, 0, 9223372036854775807);  permute_718 = None
    slice_2642: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2641, 1, -256, 9223372036854775807);  slice_2641 = None
    slice_2643: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2642, 2, 0, 9223372036854775807);  slice_2642 = None
    slice_2644: "f32[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2643, 3, -257, 9223372036854775807);  slice_2643 = None
    copy_137: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2644, where_90);  slice_2644 = where_90 = None
    view_857: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_853, [1, 12, 1024, 513]);  view_853 = None
    permute_719: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_857, [0, 2, 1, 3]);  view_857 = None
    slice_2645: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice.Tensor(permute_719, 0, 0, 9223372036854775807)
    slice_2646: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2645, 1, -256, 9223372036854775807)
    slice_2647: "f32[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(slice_2646, 2, 0, 9223372036854775807)
    slice_scatter_502: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2647, copy_137, 3, -257, 9223372036854775807);  slice_2647 = copy_137 = None
    slice_scatter_503: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2646, slice_scatter_502, 2, 0, 9223372036854775807);  slice_2646 = slice_scatter_502 = None
    slice_scatter_504: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2645, slice_scatter_503, 1, -256, 9223372036854775807);  slice_2645 = slice_scatter_503 = None
    slice_scatter_505: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_719, slice_scatter_504, 0, 0, 9223372036854775807);  permute_719 = slice_scatter_504 = None
    permute_720: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_505, [0, 2, 1, 3]);  slice_scatter_505 = None
    view_858: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_720, [12, 4, 256, 513]);  permute_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:576, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    ne_11: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg193_1, 0);  arg193_1 = None
    slice_2652: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(ne_11, 0, 0, 9223372036854775807);  ne_11 = None
    slice_2653: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_2652, 1, 0, 9223372036854775807);  slice_2652 = None
    unsqueeze_216: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_2653, 2);  slice_2653 = None
    unsqueeze_217: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
    convert_element_type_57: "f32[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_217, torch.float32)
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_91: "f32[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_217, scalar_tensor_45, convert_element_type_57);  unsqueeze_217 = scalar_tensor_45 = convert_element_type_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:584, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
    full_103: "f32[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:833, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_722: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_103, [0, 2, 1, 3]);  full_103 = None
    view_860: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_722, [1, 1024, 1]);  permute_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_723: "f32[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_91, [0, 2, 1, 3]);  where_91 = None
    view_861: "f32[1, 1024, 1]" = torch.ops.aten.view.default(permute_723, [1, 1024, 1]);  permute_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_862: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_860, [1, 2, 512, 1]);  view_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_69: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_862, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:766, code: hidden_states = hidden_states.view(
    view_863: "f32[1, 2, 512, 1]" = torch.ops.aten.view.default(view_861, [1, 2, 512, 1]);  view_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    as_strided_70: "f32[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_863, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    unsqueeze_218: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_69, 4);  as_strided_69 = None
    permute_724: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_218, [0, 1, 2, 4, 3]);  unsqueeze_218 = None
    unsqueeze_219: "f32[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_70, 4);  as_strided_70 = None
    permute_725: "f32[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_219, [0, 1, 4, 2, 3]);  unsqueeze_219 = None
    mul_88: "f32[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_724, permute_725);  permute_724 = permute_725 = None
    view_864: "f32[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_88, [1, 3, 512, 512]);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_45: "f32[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_864, [0, 0, 0, 1], 0.0);  view_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_865: "f32[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_45, [1, 3, 512, 513]);  constant_pad_nd_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    full_104: "f32[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2654: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_865, 0, 0, 9223372036854775807)
    slice_2655: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2654, 1, 0, 9223372036854775807);  slice_2654 = None
    slice_2656: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2655, 2, 0, 256);  slice_2655 = None
    slice_2657: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2656, 3, 0, 257);  slice_2656 = None
    slice_2658: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_104, 0, 0, 9223372036854775807)
    slice_2659: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2658, 1, 0, -1);  slice_2658 = None
    slice_2660: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2659, 2, 0, 9223372036854775807);  slice_2659 = None
    slice_2661: "f32[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2660, 3, 256, 9223372036854775807);  slice_2660 = None
    copy_138: "f32[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2661, slice_2657);  slice_2661 = slice_2657 = None
    slice_2662: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(full_104, 0, 0, 9223372036854775807)
    slice_2663: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2662, 1, 0, -1)
    slice_2664: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2663, 2, 0, 9223372036854775807)
    slice_scatter_506: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2664, copy_138, 3, 256, 9223372036854775807);  slice_2664 = copy_138 = None
    slice_scatter_507: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2663, slice_scatter_506, 2, 0, 9223372036854775807);  slice_2663 = slice_scatter_506 = None
    slice_scatter_508: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2662, slice_scatter_507, 1, 0, -1);  slice_2662 = slice_scatter_507 = None
    slice_scatter_509: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_104, slice_scatter_508, 0, 0, 9223372036854775807);  full_104 = slice_scatter_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    slice_2669: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_865, 0, 0, 9223372036854775807)
    select_230: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_2669, 1, -1);  slice_2669 = None
    slice_2670: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_230, 1, 256, 9223372036854775807);  select_230 = None
    slice_2671: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2670, 2, 0, 257);  slice_2670 = None
    slice_2675: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_509, 0, 0, 9223372036854775807)
    select_232: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2675, 1, -1);  slice_2675 = None
    slice_2676: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_232, 1, 0, 9223372036854775807);  select_232 = None
    slice_2677: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2676, 2, 256, 9223372036854775807);  slice_2676 = None
    copy_139: "f32[1, 256, 257]" = torch.ops.aten.copy.default(slice_2677, slice_2671);  slice_2677 = slice_2671 = None
    slice_2678: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_509, 0, 0, 9223372036854775807)
    select_233: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2678, 1, -1)
    slice_2679: "f32[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_233, 1, 0, 9223372036854775807)
    slice_scatter_510: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2679, copy_139, 2, 256, 9223372036854775807);  slice_2679 = copy_139 = None
    slice_scatter_511: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_233, slice_scatter_510, 1, 0, 9223372036854775807);  select_233 = slice_scatter_510 = None
    select_scatter_46: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2678, slice_scatter_511, 1, -1);  slice_2678 = slice_scatter_511 = None
    slice_scatter_512: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_509, select_scatter_46, 0, 0, 9223372036854775807);  slice_scatter_509 = select_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    slice_2683: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_865, 0, 0, 9223372036854775807)
    slice_2684: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(slice_2683, 1, 0, 9223372036854775807);  slice_2683 = None
    slice_2685: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2684, 2, -257, -1);  slice_2684 = None
    slice_2686: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2685, 3, 257, 9223372036854775807);  slice_2685 = None
    slice_2691: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_512, 0, 0, 9223372036854775807)
    slice_2692: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2691, 1, 1, 9223372036854775807);  slice_2691 = None
    slice_2693: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2692, 2, 0, 9223372036854775807);  slice_2692 = None
    slice_2694: "f32[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2693, 3, 0, 256);  slice_2693 = None
    copy_140: "f32[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2694, slice_2686);  slice_2694 = slice_2686 = None
    slice_2695: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_512, 0, 0, 9223372036854775807)
    slice_2696: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2695, 1, 1, 9223372036854775807)
    slice_2697: "f32[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(slice_2696, 2, 0, 9223372036854775807)
    slice_scatter_513: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2697, copy_140, 3, 0, 256);  slice_2697 = copy_140 = None
    slice_scatter_514: "f32[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2696, slice_scatter_513, 2, 0, 9223372036854775807);  slice_2696 = slice_scatter_513 = None
    slice_scatter_515: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2695, slice_scatter_514, 1, 1, 9223372036854775807);  slice_2695 = slice_scatter_514 = None
    slice_scatter_516: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_512, slice_scatter_515, 0, 0, 9223372036854775807);  slice_scatter_512 = slice_scatter_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    slice_2702: "f32[1, 3, 512, 513]" = torch.ops.aten.slice.Tensor(view_865, 0, 0, 9223372036854775807);  view_865 = None
    select_235: "f32[1, 512, 513]" = torch.ops.aten.select.int(slice_2702, 1, 0);  slice_2702 = None
    slice_2703: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_235, 1, 0, 255);  select_235 = None
    slice_2704: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2703, 2, -255, 9223372036854775807);  slice_2703 = None
    slice_2708: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_516, 0, 0, 9223372036854775807)
    select_237: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2708, 1, 0);  slice_2708 = None
    slice_2709: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_237, 1, 1, 256);  select_237 = None
    slice_2710: "f32[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2709, 2, 1, 256);  slice_2709 = None
    copy_141: "f32[1, 255, 255]" = torch.ops.aten.copy.default(slice_2710, slice_2704);  slice_2710 = slice_2704 = None
    slice_2711: "f32[1, 4, 256, 513]" = torch.ops.aten.slice.Tensor(slice_scatter_516, 0, 0, 9223372036854775807)
    select_238: "f32[1, 256, 513]" = torch.ops.aten.select.int(slice_2711, 1, 0)
    slice_2712: "f32[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_238, 1, 1, 256)
    slice_scatter_517: "f32[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2712, copy_141, 2, 1, 256);  slice_2712 = copy_141 = None
    slice_scatter_518: "f32[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_238, slice_scatter_517, 1, 1, 256);  select_238 = slice_scatter_517 = None
    select_scatter_47: "f32[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_2711, slice_scatter_518, 1, 0);  slice_2711 = slice_scatter_518 = None
    slice_scatter_519: "f32[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_scatter_516, select_scatter_47, 0, 0, 9223372036854775807);  slice_scatter_516 = select_scatter_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_105: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_46: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_220: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_46, -2);  iota_46 = None
    iota_47: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_221: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
    sub_91: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_220, unsqueeze_221);  unsqueeze_220 = unsqueeze_221 = None
    le_23: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_91, 0);  sub_91 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_92: "f32[256, 257]" = torch.ops.aten.where.self(le_23, full_105, scalar_tensor_46);  le_23 = full_105 = scalar_tensor_46 = None
    rev_46: "f32[256, 257]" = torch.ops.prims.rev.default(where_92, [0]);  where_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805, code: beginning_mask = beginning_mask_2d[None, :, None, :]
    unsqueeze_222: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_46, 0);  rev_46 = None
    slice_2716: "f32[1, 256, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_222, 1, 0, 9223372036854775807);  unsqueeze_222 = None
    unsqueeze_223: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(slice_2716, 2);  slice_2716 = None
    slice_2717: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(unsqueeze_223, 3, 0, 9223372036854775807);  unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:806, code: ending_mask = beginning_mask.flip(dims=(1, 3))
    rev_47: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(slice_2717, [1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand_46: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(slice_2717, [1, 256, 1, 257]);  slice_2717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_868: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_519, [1, 1, 1024, 513])
    permute_728: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_868, [0, 2, 1, 3]);  view_868 = None
    slice_2722: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_728, 0, 0, 9223372036854775807);  permute_728 = None
    slice_2723: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2722, 1, 0, 256);  slice_2722 = None
    slice_2724: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2723, 2, 0, 9223372036854775807);  slice_2723 = None
    slice_2725: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2724, 3, 0, 257);  slice_2724 = None
    full_106: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type_58: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_46, torch.bool);  expand_46 = None
    where_93: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_58, full_106, slice_2725);  convert_element_type_58 = full_106 = slice_2725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    view_869: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_519, [1, 1, 1024, 513])
    permute_729: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_869, [0, 2, 1, 3]);  view_869 = None
    slice_2730: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_729, 0, 0, 9223372036854775807);  permute_729 = None
    slice_2731: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2730, 1, 0, 256);  slice_2730 = None
    slice_2732: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2731, 2, 0, 9223372036854775807);  slice_2731 = None
    slice_2733: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2732, 3, 0, 257);  slice_2732 = None
    copy_142: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2733, where_93);  slice_2733 = where_93 = None
    view_870: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(slice_scatter_519, [1, 1, 1024, 513]);  slice_scatter_519 = None
    permute_730: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_870, [0, 2, 1, 3]);  view_870 = None
    slice_2734: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_730, 0, 0, 9223372036854775807)
    slice_2735: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2734, 1, 0, 256)
    slice_2736: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2735, 2, 0, 9223372036854775807)
    slice_scatter_520: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2736, copy_142, 3, 0, 257);  slice_2736 = copy_142 = None
    slice_scatter_521: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2735, slice_scatter_520, 2, 0, 9223372036854775807);  slice_2735 = slice_scatter_520 = None
    slice_scatter_522: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2734, slice_scatter_521, 1, 0, 256);  slice_2734 = slice_scatter_521 = None
    slice_scatter_523: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_730, slice_scatter_522, 0, 0, 9223372036854775807);  permute_730 = slice_scatter_522 = None
    permute_731: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_523, [0, 2, 1, 3]);  slice_scatter_523 = None
    view_871: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_731, [1, 4, 256, 513]);  permute_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_47: "f32[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_47, [1, 256, 1, 257]);  rev_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_873: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_871, [1, 1, 1024, 513])
    permute_733: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_873, [0, 2, 1, 3]);  view_873 = None
    slice_2745: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_733, 0, 0, 9223372036854775807);  permute_733 = None
    slice_2746: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2745, 1, -256, 9223372036854775807);  slice_2745 = None
    slice_2747: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2746, 2, 0, 9223372036854775807);  slice_2746 = None
    slice_2748: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2747, 3, -257, 9223372036854775807);  slice_2747 = None
    full_107: "f32[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_59: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_47, torch.bool);  expand_47 = None
    where_94: "f32[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_59, full_107, slice_2748);  convert_element_type_59 = full_107 = slice_2748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    view_874: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_871, [1, 1, 1024, 513])
    permute_734: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_874, [0, 2, 1, 3]);  view_874 = None
    slice_2753: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_734, 0, 0, 9223372036854775807);  permute_734 = None
    slice_2754: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2753, 1, -256, 9223372036854775807);  slice_2753 = None
    slice_2755: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2754, 2, 0, 9223372036854775807);  slice_2754 = None
    slice_2756: "f32[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2755, 3, -257, 9223372036854775807);  slice_2755 = None
    copy_143: "f32[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2756, where_94);  slice_2756 = where_94 = None
    view_875: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_871, [1, 1, 1024, 513]);  view_871 = None
    permute_735: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_875, [0, 2, 1, 3]);  view_875 = None
    slice_2757: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice.Tensor(permute_735, 0, 0, 9223372036854775807)
    slice_2758: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2757, 1, -256, 9223372036854775807)
    slice_2759: "f32[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(slice_2758, 2, 0, 9223372036854775807)
    slice_scatter_524: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2759, copy_143, 3, -257, 9223372036854775807);  slice_2759 = copy_143 = None
    slice_scatter_525: "f32[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2758, slice_scatter_524, 2, 0, 9223372036854775807);  slice_2758 = slice_scatter_524 = None
    slice_scatter_526: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_2757, slice_scatter_525, 1, -256, 9223372036854775807);  slice_2757 = slice_scatter_525 = None
    slice_scatter_527: "f32[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_735, slice_scatter_526, 0, 0, 9223372036854775807);  permute_735 = slice_scatter_526 = None
    permute_736: "f32[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_527, [0, 2, 1, 3]);  slice_scatter_527 = None
    view_876: "f32[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_736, [1, 4, 256, 513]);  permute_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    view_878: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_858, [1, 12, 1024, 513]);  view_858 = None
    permute_738: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_878, [0, 2, 1, 3]);  view_878 = None
    view_879: "f32[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_876, [1, 1, 1024, 513]);  view_876 = None
    permute_739: "f32[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_879, [0, 2, 1, 3]);  view_879 = None
    add_123: "f32[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_738, permute_739);  permute_738 = permute_739 = None
    permute_740: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_123, [0, 2, 1, 3]);  add_123 = None
    view_881: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_740, [12, 4, 256, 513]);  permute_740 = None
    view_882: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_881, [1, 12, 1024, 513]);  view_881 = None
    permute_741: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_882, [0, 2, 1, 3]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    clone_90: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_741, memory_format = torch.contiguous_format);  permute_741 = None
    amax_11: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_90, [-1], True)
    sub_92: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_90, amax_11);  clone_90 = amax_11 = None
    exp_11: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
    sum_12: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_117: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    slice_2764: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807);  arg194_1 = None
    slice_2765: "b8[1, 1024]" = torch.ops.aten.slice.Tensor(slice_2764, 1, 0, 9223372036854775807);  slice_2764 = None
    unsqueeze_224: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_2765, 2);  slice_2765 = None
    unsqueeze_225: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_95: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_225, scalar_tensor_47, div_117);  unsqueeze_225 = scalar_tensor_47 = div_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    clone_91: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(where_95);  where_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    view_883: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_830, [1024, 1, 12, 64]);  view_830 = None
    permute_742: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_883, [1, 0, 2, 3]);  view_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    permute_743: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(clone_91, [0, 2, 1, 3]);  clone_91 = None
    view_884: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_743, [12, 4, 256, 513]);  permute_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    permute_744: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_742, [0, 2, 1, 3]);  permute_742 = None
    view_885: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_744, [12, 1024, 64]);  permute_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    constant_pad_nd_46: "f32[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_885, [0, 0, 256, 256], -1.0);  view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    as_strided_71: "f32[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_46, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_47: "f32[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_884, [0, 257], 0.0);  view_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_886: "f32[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_47, [12, 4, -1]);  constant_pad_nd_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_2766: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(view_886, 0, 0, 9223372036854775807);  view_886 = None
    slice_2767: "f32[12, 4, 197120]" = torch.ops.aten.slice.Tensor(slice_2766, 1, 0, 9223372036854775807);  slice_2766 = None
    slice_2768: "f32[12, 4, 196864]" = torch.ops.aten.slice.Tensor(slice_2767, 2, 0, -256);  slice_2767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_887: "f32[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_2768, [12, 4, 256, 769]);  slice_2768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_2769: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(view_887, 0, 0, 9223372036854775807);  view_887 = None
    slice_2770: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2769, 1, 0, 9223372036854775807);  slice_2769 = None
    slice_2771: "f32[12, 4, 256, 769]" = torch.ops.aten.slice.Tensor(slice_2770, 2, 0, 9223372036854775807);  slice_2770 = None
    slice_2772: "f32[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(slice_2771, 3, 0, -1);  slice_2771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    unsqueeze_226: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2772, 4);  slice_2772 = None
    permute_745: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_226, [0, 1, 2, 4, 3]);  unsqueeze_226 = None
    unsqueeze_227: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_71, 4);  as_strided_71 = None
    permute_746: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_227, [0, 1, 4, 3, 2]);  unsqueeze_227 = None
    permute_747: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_745, [0, 1, 2, 4, 3]);  permute_745 = None
    view_888: "f32[48, 256, 768]" = torch.ops.aten.view.default(permute_747, [48, 256, 768]);  permute_747 = None
    permute_748: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_746, [0, 1, 4, 3, 2]);  permute_746 = None
    clone_92: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None
    view_889: "f32[48, 768, 64]" = torch.ops.aten.view.default(clone_92, [48, 768, 64]);  clone_92 = None
    bmm_23: "f32[48, 256, 64]" = torch.ops.aten.bmm.default(view_888, view_889);  view_888 = view_889 = None
    view_890: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_23, [12, 4, 256, 1, 64]);  bmm_23 = None
    permute_749: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_890, [0, 1, 2, 4, 3]);  view_890 = None
    view_891: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_749, [12, 4, 256, 64]);  permute_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    view_892: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_891, [1, 12, 1024, 64]);  view_891 = None
    permute_750: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_892, [0, 2, 1, 3]);  view_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    permute_751: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_750, [1, 0, 2, 3]);  permute_750 = None
    clone_93: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
    view_893: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_93, [1024, 1, 768]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_752: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(view_893, [1, 0, 2]);  view_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_894: "f32[1024, 768]" = torch.ops.aten.view.default(permute_752, [1024, 768]);  permute_752 = None
    permute_753: "f32[768, 768]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_69: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg183_1, view_894, permute_753);  arg183_1 = view_894 = permute_753 = None
    view_895: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_69, [1, 1024, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    clone_94: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_895);  view_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_125: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_94, add_120);  clone_94 = add_120 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1024, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    add_126: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_94: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_125, getitem_45);  add_125 = getitem_45 = None
    mul_89: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_22);  sub_94 = rsqrt_22 = None
    mul_90: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_89, arg184_1);  mul_89 = arg184_1 = None
    add_127: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_90, arg185_1);  mul_90 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_896: "f32[1024, 768]" = torch.ops.aten.view.default(add_127, [1024, 768])
    permute_754: "f32[768, 3072]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_70: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg187_1, view_896, permute_754);  arg187_1 = view_896 = permute_754 = None
    view_897: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_70, [1, 1024, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_91: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_897, 0.5)
    mul_92: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_897, 0.7071067811865476);  view_897 = None
    erf_11: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
    add_128: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_93: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_91, add_128);  mul_91 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_898: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_93, [1024, 3072]);  mul_93 = None
    permute_755: "f32[3072, 768]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    addmm_71: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg189_1, view_898, permute_755);  arg189_1 = view_898 = permute_755 = None
    view_899: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_71, [1, 1024, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    clone_95: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_899);  view_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_129: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_95, add_127);  clone_95 = add_127 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 1024, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    add_130: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_95: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_129, getitem_47);  add_129 = getitem_47 = None
    mul_94: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_23);  sub_95 = rsqrt_23 = None
    mul_95: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_94, arg190_1);  mul_94 = arg190_1 = None
    add_131: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_95, arg191_1);  mul_95 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1348, code: hidden_states = hidden_states[:, : hidden_states.shape[1] - padding_len]
    slice_2773: "f32[1, 1024, 768]" = torch.ops.aten.slice.Tensor(add_131, 0, 0, 9223372036854775807);  add_131 = None
    return (slice_2773,)
    