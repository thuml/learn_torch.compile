from __future__ import annotations



def forward(self, arg0_1: "f32[1, 1, 768]", arg1_1: "f32[768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[2304, 768]", arg7_1: "f32[732, 12]", arg8_1: "f32[768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[2304, 768]", arg17_1: "f32[732, 12]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768]", arg26_1: "f32[2304, 768]", arg27_1: "f32[732, 12]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[2304, 768]", arg37_1: "f32[732, 12]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[2304, 768]", arg47_1: "f32[732, 12]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[2304, 768]", arg57_1: "f32[732, 12]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[2304, 768]", arg67_1: "f32[732, 12]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[2304, 768]", arg77_1: "f32[732, 12]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[768]", arg86_1: "f32[2304, 768]", arg87_1: "f32[732, 12]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[768]", arg91_1: "f32[768]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[2304, 768]", arg97_1: "f32[732, 12]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[2304, 768]", arg107_1: "f32[732, 12]", arg108_1: "f32[768]", arg109_1: "f32[768]", arg110_1: "f32[768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[768]", arg116_1: "f32[2304, 768]", arg117_1: "f32[732, 12]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768, 3, 16, 16]", arg124_1: "f32[768]", arg125_1: "f32[768, 768]", arg126_1: "f32[768]", arg127_1: "f32[3072, 768]", arg128_1: "f32[3072]", arg129_1: "f32[768, 3072]", arg130_1: "f32[768]", arg131_1: "f32[768, 768]", arg132_1: "f32[768]", arg133_1: "f32[3072, 768]", arg134_1: "f32[3072]", arg135_1: "f32[768, 3072]", arg136_1: "f32[768]", arg137_1: "f32[768, 768]", arg138_1: "f32[768]", arg139_1: "f32[3072, 768]", arg140_1: "f32[3072]", arg141_1: "f32[768, 3072]", arg142_1: "f32[768]", arg143_1: "f32[768, 768]", arg144_1: "f32[768]", arg145_1: "f32[3072, 768]", arg146_1: "f32[3072]", arg147_1: "f32[768, 3072]", arg148_1: "f32[768]", arg149_1: "f32[768, 768]", arg150_1: "f32[768]", arg151_1: "f32[3072, 768]", arg152_1: "f32[3072]", arg153_1: "f32[768, 3072]", arg154_1: "f32[768]", arg155_1: "f32[768, 768]", arg156_1: "f32[768]", arg157_1: "f32[3072, 768]", arg158_1: "f32[3072]", arg159_1: "f32[768, 3072]", arg160_1: "f32[768]", arg161_1: "f32[768, 768]", arg162_1: "f32[768]", arg163_1: "f32[3072, 768]", arg164_1: "f32[3072]", arg165_1: "f32[768, 3072]", arg166_1: "f32[768]", arg167_1: "f32[768, 768]", arg168_1: "f32[768]", arg169_1: "f32[3072, 768]", arg170_1: "f32[3072]", arg171_1: "f32[768, 3072]", arg172_1: "f32[768]", arg173_1: "f32[768, 768]", arg174_1: "f32[768]", arg175_1: "f32[3072, 768]", arg176_1: "f32[3072]", arg177_1: "f32[768, 3072]", arg178_1: "f32[768]", arg179_1: "f32[768, 768]", arg180_1: "f32[768]", arg181_1: "f32[3072, 768]", arg182_1: "f32[3072]", arg183_1: "f32[768, 3072]", arg184_1: "f32[768]", arg185_1: "f32[768, 768]", arg186_1: "f32[768]", arg187_1: "f32[3072, 768]", arg188_1: "f32[3072]", arg189_1: "f32[768, 3072]", arg190_1: "f32[768]", arg191_1: "f32[768, 768]", arg192_1: "f32[768]", arg193_1: "f32[3072, 768]", arg194_1: "f32[3072]", arg195_1: "f32[768, 3072]", arg196_1: "f32[768]", arg197_1: "f32[1000, 768]", arg198_1: "f32[1000]", arg199_1: "f32[768]", arg200_1: "i64[197, 197]", arg201_1: "f32[768]", arg202_1: "i64[197, 197]", arg203_1: "f32[768]", arg204_1: "i64[197, 197]", arg205_1: "f32[768]", arg206_1: "i64[197, 197]", arg207_1: "f32[768]", arg208_1: "i64[197, 197]", arg209_1: "f32[768]", arg210_1: "i64[197, 197]", arg211_1: "f32[768]", arg212_1: "i64[197, 197]", arg213_1: "f32[768]", arg214_1: "i64[197, 197]", arg215_1: "f32[768]", arg216_1: "i64[197, 197]", arg217_1: "f32[768]", arg218_1: "i64[197, 197]", arg219_1: "f32[768]", arg220_1: "i64[197, 197]", arg221_1: "f32[768]", arg222_1: "i64[197, 197]", arg223_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(arg223_1, arg123_1, arg124_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg223_1 = arg123_1 = arg124_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.view.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:405, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(arg0_1, [8, -1, -1]);  arg0_1 = None
    cat: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:408, code: x = self.pos_drop(x)
    clone: "f32[8, 197, 768]" = torch.ops.aten.clone.default(cat);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 197, 1]" = var_mean[0]
    getitem_1: "f32[8, 197, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  getitem_1 = None
    mul: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_1: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_1: "f32[2304]" = torch.ops.aten.cat.default([arg4_1, arg199_1, arg5_1]);  arg4_1 = arg199_1 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_1: "f32[1576, 768]" = torch.ops.aten.view.default(add_1, [1576, 768]);  add_1 = None
    permute_1: "f32[768, 2304]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_1, view_1, permute_1);  cat_1 = view_1 = permute_1 = None
    view_2: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm, [8, 197, 2304]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_3: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_2, [8, 197, 3, 12, -1]);  view_2 = None
    permute_2: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_2: "f32[8, 12, 197, 64]" = unbind[0]
    getitem_3: "f32[8, 12, 197, 64]" = unbind[1]
    getitem_4: "f32[8, 12, 197, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_4: "i64[38809]" = torch.ops.aten.view.default(arg200_1, [-1]);  arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg7_1, [view_4]);  arg7_1 = view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_5: "f32[197, 197, 12]" = torch.ops.aten.view.default(index, [197, 197, -1]);  index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_3: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_5, [2, 0, 1]);  view_5 = None
    clone_1: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_1, 0);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze, [0, 3], 0.0);  unsqueeze = None
    slice_1: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 197);  constant_pad_nd = None
    expand_1: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_1, [8, 12, 197, 197]);  slice_1 = None
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_2, getitem_3, getitem_4, expand_1, False);  getitem_2 = getitem_3 = getitem_4 = expand_1 = None
    getitem_5: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_4: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_6: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_4, [8, 197, 768]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_7: "f32[1576, 768]" = torch.ops.aten.view.default(view_6, [1576, 768]);  view_6 = None
    permute_5: "f32[768, 768]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    addmm_1: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg126_1, view_7, permute_5);  arg126_1 = view_7 = permute_5 = None
    view_8: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_1, [8, 197, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_2: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_2: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg1_1, clone_2);  arg1_1 = clone_2 = None
    add_2: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(clone, mul_2);  clone = mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem_9: "f32[8, 197, 1]" = var_mean_1[0]
    getitem_10: "f32[8, 197, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
    rsqrt_1: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_1: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_10);  getitem_10 = None
    mul_3: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_4: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg9_1);  mul_3 = arg9_1 = None
    add_4: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_4, arg10_1);  mul_4 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_9: "f32[1576, 768]" = torch.ops.aten.view.default(add_4, [1576, 768]);  add_4 = None
    permute_6: "f32[768, 3072]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    addmm_2: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg128_1, view_9, permute_6);  arg128_1 = view_9 = permute_6 = None
    view_10: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 197, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
    mul_6: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.7071067811865476);  view_10 = None
    erf: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_5: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_3: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_3, [1576, 3072]);  clone_3 = None
    permute_7: "f32[3072, 768]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    addmm_3: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg130_1, view_11, permute_7);  arg130_1 = view_11 = permute_7 = None
    view_12: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_3, [8, 197, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_4: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_8: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg8_1, clone_4);  arg8_1 = clone_4 = None
    add_6: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_2, mul_8);  add_2 = mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_11: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_12: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_12);  getitem_12 = None
    mul_9: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_10: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg12_1);  mul_9 = arg12_1 = None
    add_8: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_10, arg13_1);  mul_10 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_2: "f32[2304]" = torch.ops.aten.cat.default([arg14_1, arg201_1, arg15_1]);  arg14_1 = arg201_1 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_13: "f32[1576, 768]" = torch.ops.aten.view.default(add_8, [1576, 768]);  add_8 = None
    permute_8: "f32[768, 2304]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_4: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_2, view_13, permute_8);  cat_2 = view_13 = permute_8 = None
    view_14: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_4, [8, 197, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_15: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_14, [8, 197, 3, 12, -1]);  view_14 = None
    permute_9: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_13: "f32[8, 12, 197, 64]" = unbind_1[0]
    getitem_14: "f32[8, 12, 197, 64]" = unbind_1[1]
    getitem_15: "f32[8, 12, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_16: "i64[38809]" = torch.ops.aten.view.default(arg202_1, [-1]);  arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_1: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg17_1, [view_16]);  arg17_1 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_17: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_1, [197, 197, -1]);  index_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_10: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_17, [2, 0, 1]);  view_17 = None
    clone_5: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_1: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_5, 0);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_1: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_1, [0, 3], 0.0);  unsqueeze_1 = None
    slice_2: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_1, -1, 0, 197);  constant_pad_nd_1 = None
    expand_2: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_2, [8, 12, 197, 197]);  slice_2 = None
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_13, getitem_14, getitem_15, expand_2, False);  getitem_13 = getitem_14 = getitem_15 = expand_2 = None
    getitem_16: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_11: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    view_18: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_11, [8, 197, 768]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_19: "f32[1576, 768]" = torch.ops.aten.view.default(view_18, [1576, 768]);  view_18 = None
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_5: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg132_1, view_19, permute_12);  arg132_1 = view_19 = permute_12 = None
    view_20: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_5, [8, 197, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_6: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_11: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg11_1, clone_6);  arg11_1 = clone_6 = None
    add_9: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_6, mul_11);  add_6 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_21: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_21);  getitem_21 = None
    mul_12: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    mul_13: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_12, arg19_1);  mul_12 = arg19_1 = None
    add_11: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_13, arg20_1);  mul_13 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_21: "f32[1576, 768]" = torch.ops.aten.view.default(add_11, [1576, 768]);  add_11 = None
    permute_13: "f32[768, 3072]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    addmm_6: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg134_1, view_21, permute_13);  arg134_1 = view_21 = permute_13 = None
    view_22: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 197, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_14: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
    mul_15: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476);  view_22 = None
    erf_1: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_12: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_12);  mul_14 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_7: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_7, [1576, 3072]);  clone_7 = None
    permute_14: "f32[3072, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_7: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg136_1, view_23, permute_14);  arg136_1 = view_23 = permute_14 = None
    view_24: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_7, [8, 197, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_8: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_17: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg18_1, clone_8);  arg18_1 = clone_8 = None
    add_13: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_9, mul_17);  add_9 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_23: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_23);  getitem_23 = None
    mul_18: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    mul_19: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg22_1);  mul_18 = arg22_1 = None
    add_15: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_19, arg23_1);  mul_19 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_3: "f32[2304]" = torch.ops.aten.cat.default([arg24_1, arg203_1, arg25_1]);  arg24_1 = arg203_1 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_25: "f32[1576, 768]" = torch.ops.aten.view.default(add_15, [1576, 768]);  add_15 = None
    permute_15: "f32[768, 2304]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_8: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_3, view_25, permute_15);  cat_3 = view_25 = permute_15 = None
    view_26: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_8, [8, 197, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_27: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_26, [8, 197, 3, 12, -1]);  view_26 = None
    permute_16: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_27, [2, 0, 3, 1, 4]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_2 = torch.ops.aten.unbind.int(permute_16);  permute_16 = None
    getitem_24: "f32[8, 12, 197, 64]" = unbind_2[0]
    getitem_25: "f32[8, 12, 197, 64]" = unbind_2[1]
    getitem_26: "f32[8, 12, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_28: "i64[38809]" = torch.ops.aten.view.default(arg204_1, [-1]);  arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_2: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg27_1, [view_28]);  arg27_1 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_29: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_2, [197, 197, -1]);  index_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_17: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_29, [2, 0, 1]);  view_29 = None
    clone_9: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_2: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_9, 0);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_2: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_2, [0, 3], 0.0);  unsqueeze_2 = None
    slice_3: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_2, -1, 0, 197);  constant_pad_nd_2 = None
    expand_3: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_3, [8, 12, 197, 197]);  slice_3 = None
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_24, getitem_25, getitem_26, expand_3, False);  getitem_24 = getitem_25 = getitem_26 = expand_3 = None
    getitem_27: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_18: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
    view_30: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_18, [8, 197, 768]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_31: "f32[1576, 768]" = torch.ops.aten.view.default(view_30, [1576, 768]);  view_30 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_9: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg138_1, view_31, permute_19);  arg138_1 = view_31 = permute_19 = None
    view_32: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_9, [8, 197, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_10: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_32);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_20: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg21_1, clone_10);  arg21_1 = clone_10 = None
    add_16: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_13, mul_20);  add_13 = mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_31: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_32: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_17: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-06);  getitem_31 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_5: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_32);  getitem_32 = None
    mul_21: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    mul_22: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg29_1);  mul_21 = arg29_1 = None
    add_18: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_22, arg30_1);  mul_22 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_33: "f32[1576, 768]" = torch.ops.aten.view.default(add_18, [1576, 768]);  add_18 = None
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    addmm_10: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg140_1, view_33, permute_20);  arg140_1 = view_33 = permute_20 = None
    view_34: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 197, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_23: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
    mul_24: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.7071067811865476);  view_34 = None
    erf_2: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_19: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_23, add_19);  mul_23 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_11: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_25);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_11, [1576, 3072]);  clone_11 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    addmm_11: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg142_1, view_35, permute_21);  arg142_1 = view_35 = permute_21 = None
    view_36: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_11, [8, 197, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_12: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_26: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg28_1, clone_12);  arg28_1 = clone_12 = None
    add_20: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_16, mul_26);  add_16 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_34: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_6: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_34);  getitem_34 = None
    mul_27: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    mul_28: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_27, arg32_1);  mul_27 = arg32_1 = None
    add_22: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_28, arg33_1);  mul_28 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_4: "f32[2304]" = torch.ops.aten.cat.default([arg34_1, arg205_1, arg35_1]);  arg34_1 = arg205_1 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_37: "f32[1576, 768]" = torch.ops.aten.view.default(add_22, [1576, 768]);  add_22 = None
    permute_22: "f32[768, 2304]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_12: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_4, view_37, permute_22);  cat_4 = view_37 = permute_22 = None
    view_38: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 197, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_39: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_38, [8, 197, 3, 12, -1]);  view_38 = None
    permute_23: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_39, [2, 0, 3, 1, 4]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_3 = torch.ops.aten.unbind.int(permute_23);  permute_23 = None
    getitem_35: "f32[8, 12, 197, 64]" = unbind_3[0]
    getitem_36: "f32[8, 12, 197, 64]" = unbind_3[1]
    getitem_37: "f32[8, 12, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_40: "i64[38809]" = torch.ops.aten.view.default(arg206_1, [-1]);  arg206_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_3: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg37_1, [view_40]);  arg37_1 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_41: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_3, [197, 197, -1]);  index_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_24: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_41, [2, 0, 1]);  view_41 = None
    clone_13: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_3: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_13, 0);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_3: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_3, [0, 3], 0.0);  unsqueeze_3 = None
    slice_4: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_3, -1, 0, 197);  constant_pad_nd_3 = None
    expand_4: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_4, [8, 12, 197, 197]);  slice_4 = None
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_35, getitem_36, getitem_37, expand_4, False);  getitem_35 = getitem_36 = getitem_37 = expand_4 = None
    getitem_38: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_25: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
    view_42: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_25, [8, 197, 768]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_43: "f32[1576, 768]" = torch.ops.aten.view.default(view_42, [1576, 768]);  view_42 = None
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_13: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg144_1, view_43, permute_26);  arg144_1 = view_43 = permute_26 = None
    view_44: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_13, [8, 197, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_14: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_29: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg31_1, clone_14);  arg31_1 = clone_14 = None
    add_23: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_20, mul_29);  add_20 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_43: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_7: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_43);  getitem_43 = None
    mul_30: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    mul_31: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg39_1);  mul_30 = arg39_1 = None
    add_25: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_31, arg40_1);  mul_31 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[1576, 768]" = torch.ops.aten.view.default(add_25, [1576, 768]);  add_25 = None
    permute_27: "f32[768, 3072]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    addmm_14: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg146_1, view_45, permute_27);  arg146_1 = view_45 = permute_27 = None
    view_46: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 197, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
    mul_33: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476);  view_46 = None
    erf_3: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_26: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_34: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_26);  mul_32 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_15, [1576, 3072]);  clone_15 = None
    permute_28: "f32[3072, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_15: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg148_1, view_47, permute_28);  arg148_1 = view_47 = permute_28 = None
    view_48: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_15, [8, 197, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_35: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg38_1, clone_16);  arg38_1 = clone_16 = None
    add_27: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_23, mul_35);  add_23 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 197, 1]" = var_mean_8[0]
    getitem_45: "f32[8, 197, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_8: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_45);  getitem_45 = None
    mul_36: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    mul_37: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg42_1);  mul_36 = arg42_1 = None
    add_29: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_37, arg43_1);  mul_37 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_5: "f32[2304]" = torch.ops.aten.cat.default([arg44_1, arg207_1, arg45_1]);  arg44_1 = arg207_1 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_49: "f32[1576, 768]" = torch.ops.aten.view.default(add_29, [1576, 768]);  add_29 = None
    permute_29: "f32[768, 2304]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_16: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_5, view_49, permute_29);  cat_5 = view_49 = permute_29 = None
    view_50: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_16, [8, 197, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_51: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_50, [8, 197, 3, 12, -1]);  view_50 = None
    permute_30: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_4 = torch.ops.aten.unbind.int(permute_30);  permute_30 = None
    getitem_46: "f32[8, 12, 197, 64]" = unbind_4[0]
    getitem_47: "f32[8, 12, 197, 64]" = unbind_4[1]
    getitem_48: "f32[8, 12, 197, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_52: "i64[38809]" = torch.ops.aten.view.default(arg208_1, [-1]);  arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_4: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg47_1, [view_52]);  arg47_1 = view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_53: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_4, [197, 197, -1]);  index_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_31: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_53, [2, 0, 1]);  view_53 = None
    clone_17: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_4: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_17, 0);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_4: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_4, [0, 3], 0.0);  unsqueeze_4 = None
    slice_5: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_4, -1, 0, 197);  constant_pad_nd_4 = None
    expand_5: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_5, [8, 12, 197, 197]);  slice_5 = None
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_46, getitem_47, getitem_48, expand_5, False);  getitem_46 = getitem_47 = getitem_48 = expand_5 = None
    getitem_49: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_32: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
    view_54: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_32, [8, 197, 768]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_55: "f32[1576, 768]" = torch.ops.aten.view.default(view_54, [1576, 768]);  view_54 = None
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_17: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg150_1, view_55, permute_33);  arg150_1 = view_55 = permute_33 = None
    view_56: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_17, [8, 197, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_18: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_38: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg41_1, clone_18);  arg41_1 = clone_18 = None
    add_30: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_27, mul_38);  add_27 = mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
    getitem_53: "f32[8, 197, 1]" = var_mean_9[0]
    getitem_54: "f32[8, 197, 1]" = var_mean_9[1];  var_mean_9 = None
    add_31: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-06);  getitem_53 = None
    rsqrt_9: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_9: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_54);  getitem_54 = None
    mul_39: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    mul_40: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_39, arg49_1);  mul_39 = arg49_1 = None
    add_32: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_40, arg50_1);  mul_40 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_57: "f32[1576, 768]" = torch.ops.aten.view.default(add_32, [1576, 768]);  add_32 = None
    permute_34: "f32[768, 3072]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_18: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg152_1, view_57, permute_34);  arg152_1 = view_57 = permute_34 = None
    view_58: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 197, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_41: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_42: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_4: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_33: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_43: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_41, add_33);  mul_41 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_19: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_43);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_19, [1576, 3072]);  clone_19 = None
    permute_35: "f32[3072, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_19: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg154_1, view_59, permute_35);  arg154_1 = view_59 = permute_35 = None
    view_60: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_19, [8, 197, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_20: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_44: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg48_1, clone_20);  arg48_1 = clone_20 = None
    add_34: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_30, mul_44);  add_30 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
    getitem_55: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_56: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_10: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_34, getitem_56);  getitem_56 = None
    mul_45: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    mul_46: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg52_1);  mul_45 = arg52_1 = None
    add_36: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_46, arg53_1);  mul_46 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_6: "f32[2304]" = torch.ops.aten.cat.default([arg54_1, arg209_1, arg55_1]);  arg54_1 = arg209_1 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_61: "f32[1576, 768]" = torch.ops.aten.view.default(add_36, [1576, 768]);  add_36 = None
    permute_36: "f32[768, 2304]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_20: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_6, view_61, permute_36);  cat_6 = view_61 = permute_36 = None
    view_62: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_20, [8, 197, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_63: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_62, [8, 197, 3, 12, -1]);  view_62 = None
    permute_37: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
    getitem_57: "f32[8, 12, 197, 64]" = unbind_5[0]
    getitem_58: "f32[8, 12, 197, 64]" = unbind_5[1]
    getitem_59: "f32[8, 12, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_64: "i64[38809]" = torch.ops.aten.view.default(arg210_1, [-1]);  arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_5: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg57_1, [view_64]);  arg57_1 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_65: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_5, [197, 197, -1]);  index_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_38: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_65, [2, 0, 1]);  view_65 = None
    clone_21: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_5: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_21, 0);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_5: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_5, [0, 3], 0.0);  unsqueeze_5 = None
    slice_6: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_5, -1, 0, 197);  constant_pad_nd_5 = None
    expand_6: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_6, [8, 12, 197, 197]);  slice_6 = None
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_57, getitem_58, getitem_59, expand_6, False);  getitem_57 = getitem_58 = getitem_59 = expand_6 = None
    getitem_60: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_39: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
    view_66: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_39, [8, 197, 768]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_67: "f32[1576, 768]" = torch.ops.aten.view.default(view_66, [1576, 768]);  view_66 = None
    permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    addmm_21: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg156_1, view_67, permute_40);  arg156_1 = view_67 = permute_40 = None
    view_68: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_21, [8, 197, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_22: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_47: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg51_1, clone_22);  arg51_1 = clone_22 = None
    add_37: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_34, mul_47);  add_34 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_65: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_11: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_65);  getitem_65 = None
    mul_48: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    mul_49: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg59_1);  mul_48 = arg59_1 = None
    add_39: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_49, arg60_1);  mul_49 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_69: "f32[1576, 768]" = torch.ops.aten.view.default(add_39, [1576, 768]);  add_39 = None
    permute_41: "f32[768, 3072]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_22: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg158_1, view_69, permute_41);  arg158_1 = view_69 = permute_41 = None
    view_70: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 197, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_50: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
    mul_51: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476);  view_70 = None
    erf_5: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_40: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_52: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_50, add_40);  mul_50 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_23, [1576, 3072]);  clone_23 = None
    permute_42: "f32[3072, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_23: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg160_1, view_71, permute_42);  arg160_1 = view_71 = permute_42 = None
    view_72: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_23, [8, 197, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_53: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg58_1, clone_24);  arg58_1 = clone_24 = None
    add_41: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_37, mul_53);  add_37 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 197, 1]" = var_mean_12[0]
    getitem_67: "f32[8, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_12: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_67);  getitem_67 = None
    mul_54: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    mul_55: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg62_1);  mul_54 = arg62_1 = None
    add_43: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_55, arg63_1);  mul_55 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_7: "f32[2304]" = torch.ops.aten.cat.default([arg64_1, arg211_1, arg65_1]);  arg64_1 = arg211_1 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_73: "f32[1576, 768]" = torch.ops.aten.view.default(add_43, [1576, 768]);  add_43 = None
    permute_43: "f32[768, 2304]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    addmm_24: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_7, view_73, permute_43);  cat_7 = view_73 = permute_43 = None
    view_74: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 197, 2304]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_75: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_74, [8, 197, 3, 12, -1]);  view_74 = None
    permute_44: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_75, [2, 0, 3, 1, 4]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_6 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
    getitem_68: "f32[8, 12, 197, 64]" = unbind_6[0]
    getitem_69: "f32[8, 12, 197, 64]" = unbind_6[1]
    getitem_70: "f32[8, 12, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_76: "i64[38809]" = torch.ops.aten.view.default(arg212_1, [-1]);  arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_6: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg67_1, [view_76]);  arg67_1 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_77: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_6, [197, 197, -1]);  index_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_45: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_77, [2, 0, 1]);  view_77 = None
    clone_25: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_6: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_25, 0);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_6: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_6, [0, 3], 0.0);  unsqueeze_6 = None
    slice_7: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_6, -1, 0, 197);  constant_pad_nd_6 = None
    expand_7: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_7, [8, 12, 197, 197]);  slice_7 = None
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_68, getitem_69, getitem_70, expand_7, False);  getitem_68 = getitem_69 = getitem_70 = expand_7 = None
    getitem_71: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_46: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
    view_78: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_46, [8, 197, 768]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_79: "f32[1576, 768]" = torch.ops.aten.view.default(view_78, [1576, 768]);  view_78 = None
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    addmm_25: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg162_1, view_79, permute_47);  arg162_1 = view_79 = permute_47 = None
    view_80: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_25, [8, 197, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_26: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_56: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg61_1, clone_26);  arg61_1 = clone_26 = None
    add_44: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_41, mul_56);  add_41 = mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
    getitem_75: "f32[8, 197, 1]" = var_mean_13[0]
    getitem_76: "f32[8, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
    rsqrt_13: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_13: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_76);  getitem_76 = None
    mul_57: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    mul_58: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg69_1);  mul_57 = arg69_1 = None
    add_46: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_58, arg70_1);  mul_58 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_81: "f32[1576, 768]" = torch.ops.aten.view.default(add_46, [1576, 768]);  add_46 = None
    permute_48: "f32[768, 3072]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_26: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg164_1, view_81, permute_48);  arg164_1 = view_81 = permute_48 = None
    view_82: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 197, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_59: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
    mul_60: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.7071067811865476);  view_82 = None
    erf_6: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_47: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_61: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_59, add_47);  mul_59 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_27: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_61);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_27, [1576, 3072]);  clone_27 = None
    permute_49: "f32[3072, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_27: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg166_1, view_83, permute_49);  arg166_1 = view_83 = permute_49 = None
    view_84: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_27, [8, 197, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_28: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_62: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg68_1, clone_28);  arg68_1 = clone_28 = None
    add_48: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_44, mul_62);  add_44 = mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 197, 1]" = var_mean_14[0]
    getitem_78: "f32[8, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_14: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_14: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_78);  getitem_78 = None
    mul_63: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    mul_64: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg72_1);  mul_63 = arg72_1 = None
    add_50: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_64, arg73_1);  mul_64 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_8: "f32[2304]" = torch.ops.aten.cat.default([arg74_1, arg213_1, arg75_1]);  arg74_1 = arg213_1 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_85: "f32[1576, 768]" = torch.ops.aten.view.default(add_50, [1576, 768]);  add_50 = None
    permute_50: "f32[768, 2304]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    addmm_28: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_8, view_85, permute_50);  cat_8 = view_85 = permute_50 = None
    view_86: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_28, [8, 197, 2304]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_87: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_86, [8, 197, 3, 12, -1]);  view_86 = None
    permute_51: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_7 = torch.ops.aten.unbind.int(permute_51);  permute_51 = None
    getitem_79: "f32[8, 12, 197, 64]" = unbind_7[0]
    getitem_80: "f32[8, 12, 197, 64]" = unbind_7[1]
    getitem_81: "f32[8, 12, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_88: "i64[38809]" = torch.ops.aten.view.default(arg214_1, [-1]);  arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_7: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg77_1, [view_88]);  arg77_1 = view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_89: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_7, [197, 197, -1]);  index_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_52: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_89, [2, 0, 1]);  view_89 = None
    clone_29: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_7: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_29, 0);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_7: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_7, [0, 3], 0.0);  unsqueeze_7 = None
    slice_8: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_7, -1, 0, 197);  constant_pad_nd_7 = None
    expand_8: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_8, [8, 12, 197, 197]);  slice_8 = None
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_79, getitem_80, getitem_81, expand_8, False);  getitem_79 = getitem_80 = getitem_81 = expand_8 = None
    getitem_82: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_53: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
    view_90: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_53, [8, 197, 768]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_91: "f32[1576, 768]" = torch.ops.aten.view.default(view_90, [1576, 768]);  view_90 = None
    permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_29: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg168_1, view_91, permute_54);  arg168_1 = view_91 = permute_54 = None
    view_92: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_29, [8, 197, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_30: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_92);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_65: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg71_1, clone_30);  arg71_1 = clone_30 = None
    add_51: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_48, mul_65);  add_48 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 197, 1]" = var_mean_15[0]
    getitem_87: "f32[8, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_15: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_15: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_87);  getitem_87 = None
    mul_66: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    mul_67: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg79_1);  mul_66 = arg79_1 = None
    add_53: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_67, arg80_1);  mul_67 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_93: "f32[1576, 768]" = torch.ops.aten.view.default(add_53, [1576, 768]);  add_53 = None
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_30: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg170_1, view_93, permute_55);  arg170_1 = view_93 = permute_55 = None
    view_94: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 197, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
    mul_69: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.7071067811865476);  view_94 = None
    erf_7: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_54: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_70: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_54);  mul_68 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_95: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_31, [1576, 3072]);  clone_31 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    addmm_31: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg172_1, view_95, permute_56);  arg172_1 = view_95 = permute_56 = None
    view_96: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_31, [8, 197, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_71: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg78_1, clone_32);  arg78_1 = clone_32 = None
    add_55: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_51, mul_71);  add_51 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_89: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_16: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_89);  getitem_89 = None
    mul_72: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    mul_73: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg82_1);  mul_72 = arg82_1 = None
    add_57: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_73, arg83_1);  mul_73 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_9: "f32[2304]" = torch.ops.aten.cat.default([arg84_1, arg215_1, arg85_1]);  arg84_1 = arg215_1 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_97: "f32[1576, 768]" = torch.ops.aten.view.default(add_57, [1576, 768]);  add_57 = None
    permute_57: "f32[768, 2304]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_32: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_9, view_97, permute_57);  cat_9 = view_97 = permute_57 = None
    view_98: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_32, [8, 197, 2304]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_99: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_98, [8, 197, 3, 12, -1]);  view_98 = None
    permute_58: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_99, [2, 0, 3, 1, 4]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_8 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
    getitem_90: "f32[8, 12, 197, 64]" = unbind_8[0]
    getitem_91: "f32[8, 12, 197, 64]" = unbind_8[1]
    getitem_92: "f32[8, 12, 197, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_100: "i64[38809]" = torch.ops.aten.view.default(arg216_1, [-1]);  arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_8: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg87_1, [view_100]);  arg87_1 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_101: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_8, [197, 197, -1]);  index_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_59: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_101, [2, 0, 1]);  view_101 = None
    clone_33: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_8: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_33, 0);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_8: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_8, [0, 3], 0.0);  unsqueeze_8 = None
    slice_9: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_8, -1, 0, 197);  constant_pad_nd_8 = None
    expand_9: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_9, [8, 12, 197, 197]);  slice_9 = None
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_90, getitem_91, getitem_92, expand_9, False);  getitem_90 = getitem_91 = getitem_92 = expand_9 = None
    getitem_93: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_60: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
    view_102: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_60, [8, 197, 768]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_103: "f32[1576, 768]" = torch.ops.aten.view.default(view_102, [1576, 768]);  view_102 = None
    permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_33: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg174_1, view_103, permute_61);  arg174_1 = view_103 = permute_61 = None
    view_104: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_33, [8, 197, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_34: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_74: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg81_1, clone_34);  arg81_1 = clone_34 = None
    add_58: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_55, mul_74);  add_55 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
    getitem_97: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_98: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_17: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_58, getitem_98);  getitem_98 = None
    mul_75: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    mul_76: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_75, arg89_1);  mul_75 = arg89_1 = None
    add_60: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_76, arg90_1);  mul_76 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[1576, 768]" = torch.ops.aten.view.default(add_60, [1576, 768]);  add_60 = None
    permute_62: "f32[768, 3072]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg176_1, view_105, permute_62);  arg176_1 = view_105 = permute_62 = None
    view_106: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_78: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476);  view_106 = None
    erf_8: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_61: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_79: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_61);  mul_77 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_35: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_35, [1576, 3072]);  clone_35 = None
    permute_63: "f32[3072, 768]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    addmm_35: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg178_1, view_107, permute_63);  arg178_1 = view_107 = permute_63 = None
    view_108: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_35, [8, 197, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_36: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_80: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg88_1, clone_36);  arg88_1 = clone_36 = None
    add_62: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_58, mul_80);  add_58 = mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
    getitem_99: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_100: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_18: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_62, getitem_100);  getitem_100 = None
    mul_81: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    mul_82: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg92_1);  mul_81 = arg92_1 = None
    add_64: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_82, arg93_1);  mul_82 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_10: "f32[2304]" = torch.ops.aten.cat.default([arg94_1, arg217_1, arg95_1]);  arg94_1 = arg217_1 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_109: "f32[1576, 768]" = torch.ops.aten.view.default(add_64, [1576, 768]);  add_64 = None
    permute_64: "f32[768, 2304]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_36: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_10, view_109, permute_64);  cat_10 = view_109 = permute_64 = None
    view_110: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 197, 2304]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_111: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_110, [8, 197, 3, 12, -1]);  view_110 = None
    permute_65: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_9 = torch.ops.aten.unbind.int(permute_65);  permute_65 = None
    getitem_101: "f32[8, 12, 197, 64]" = unbind_9[0]
    getitem_102: "f32[8, 12, 197, 64]" = unbind_9[1]
    getitem_103: "f32[8, 12, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_112: "i64[38809]" = torch.ops.aten.view.default(arg218_1, [-1]);  arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_9: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg97_1, [view_112]);  arg97_1 = view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_113: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_9, [197, 197, -1]);  index_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_66: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_113, [2, 0, 1]);  view_113 = None
    clone_37: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_9: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_37, 0);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_9: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_9, [0, 3], 0.0);  unsqueeze_9 = None
    slice_10: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_9, -1, 0, 197);  constant_pad_nd_9 = None
    expand_10: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_10, [8, 12, 197, 197]);  slice_10 = None
    _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_101, getitem_102, getitem_103, expand_10, False);  getitem_101 = getitem_102 = getitem_103 = expand_10 = None
    getitem_104: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_67: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
    view_114: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_67, [8, 197, 768]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_115: "f32[1576, 768]" = torch.ops.aten.view.default(view_114, [1576, 768]);  view_114 = None
    permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_37: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg180_1, view_115, permute_68);  arg180_1 = view_115 = permute_68 = None
    view_116: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_37, [8, 197, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_38: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_83: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg91_1, clone_38);  arg91_1 = clone_38 = None
    add_65: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_62, mul_83);  add_62 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_109: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_19: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_109);  getitem_109 = None
    mul_84: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    mul_85: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg99_1);  mul_84 = arg99_1 = None
    add_67: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_85, arg100_1);  mul_85 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[1576, 768]" = torch.ops.aten.view.default(add_67, [1576, 768]);  add_67 = None
    permute_69: "f32[768, 3072]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_38: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg182_1, view_117, permute_69);  arg182_1 = view_117 = permute_69 = None
    view_118: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 197, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_87: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_9: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_68: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_88: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_68);  mul_86 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_39, [1576, 3072]);  clone_39 = None
    permute_70: "f32[3072, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_39: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg184_1, view_119, permute_70);  arg184_1 = view_119 = permute_70 = None
    view_120: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_39, [8, 197, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_89: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg98_1, clone_40);  arg98_1 = clone_40 = None
    add_69: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_65, mul_89);  add_65 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_111: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_20: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_111);  getitem_111 = None
    mul_90: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    mul_91: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg102_1);  mul_90 = arg102_1 = None
    add_71: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_91, arg103_1);  mul_91 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_11: "f32[2304]" = torch.ops.aten.cat.default([arg104_1, arg219_1, arg105_1]);  arg104_1 = arg219_1 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_121: "f32[1576, 768]" = torch.ops.aten.view.default(add_71, [1576, 768]);  add_71 = None
    permute_71: "f32[768, 2304]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_40: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_11, view_121, permute_71);  cat_11 = view_121 = permute_71 = None
    view_122: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_40, [8, 197, 2304]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_123: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_122, [8, 197, 3, 12, -1]);  view_122 = None
    permute_72: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_123, [2, 0, 3, 1, 4]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_10 = torch.ops.aten.unbind.int(permute_72);  permute_72 = None
    getitem_112: "f32[8, 12, 197, 64]" = unbind_10[0]
    getitem_113: "f32[8, 12, 197, 64]" = unbind_10[1]
    getitem_114: "f32[8, 12, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_124: "i64[38809]" = torch.ops.aten.view.default(arg220_1, [-1]);  arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_10: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg107_1, [view_124]);  arg107_1 = view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_125: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_10, [197, 197, -1]);  index_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_73: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_125, [2, 0, 1]);  view_125 = None
    clone_41: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_10: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_41, 0);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_10: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_10, [0, 3], 0.0);  unsqueeze_10 = None
    slice_11: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_10, -1, 0, 197);  constant_pad_nd_10 = None
    expand_11: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_11, [8, 12, 197, 197]);  slice_11 = None
    _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_112, getitem_113, getitem_114, expand_11, False);  getitem_112 = getitem_113 = getitem_114 = expand_11 = None
    getitem_115: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_74: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
    view_126: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_74, [8, 197, 768]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_127: "f32[1576, 768]" = torch.ops.aten.view.default(view_126, [1576, 768]);  view_126 = None
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_41: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg186_1, view_127, permute_75);  arg186_1 = view_127 = permute_75 = None
    view_128: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_41, [8, 197, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_42: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_128);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_92: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg101_1, clone_42);  arg101_1 = clone_42 = None
    add_72: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_69, mul_92);  add_69 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_119: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_120: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-06);  getitem_119 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_21: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_120);  getitem_120 = None
    mul_93: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, arg109_1);  mul_93 = arg109_1 = None
    add_74: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_94, arg110_1);  mul_94 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_129: "f32[1576, 768]" = torch.ops.aten.view.default(add_74, [1576, 768]);  add_74 = None
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    addmm_42: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg188_1, view_129, permute_76);  arg188_1 = view_129 = permute_76 = None
    view_130: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 197, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
    mul_96: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.7071067811865476);  view_130 = None
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_75: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_97: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_95, add_75);  mul_95 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_43: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_131: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_43, [1576, 3072]);  clone_43 = None
    permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_43: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg190_1, view_131, permute_77);  arg190_1 = view_131 = permute_77 = None
    view_132: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_43, [8, 197, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_44: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_132);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_98: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg108_1, clone_44);  arg108_1 = clone_44 = None
    add_76: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_72, mul_98);  add_72 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
    getitem_121: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_122: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_22: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_122);  getitem_122 = None
    mul_99: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_99, arg112_1);  mul_99 = arg112_1 = None
    add_78: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_100, arg113_1);  mul_100 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_12: "f32[2304]" = torch.ops.aten.cat.default([arg114_1, arg221_1, arg115_1]);  arg114_1 = arg221_1 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_133: "f32[1576, 768]" = torch.ops.aten.view.default(add_78, [1576, 768]);  add_78 = None
    permute_78: "f32[768, 2304]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_44: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_12, view_133, permute_78);  cat_12 = view_133 = permute_78 = None
    view_134: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_44, [8, 197, 2304]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_135: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_134, [8, 197, 3, 12, -1]);  view_134 = None
    permute_79: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_135, [2, 0, 3, 1, 4]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_11 = torch.ops.aten.unbind.int(permute_79);  permute_79 = None
    getitem_123: "f32[8, 12, 197, 64]" = unbind_11[0]
    getitem_124: "f32[8, 12, 197, 64]" = unbind_11[1]
    getitem_125: "f32[8, 12, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_136: "i64[38809]" = torch.ops.aten.view.default(arg222_1, [-1]);  arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_11: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg117_1, [view_136]);  arg117_1 = view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_137: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_11, [197, 197, -1]);  index_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_80: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_137, [2, 0, 1]);  view_137 = None
    clone_45: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_11: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_45, 0);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_11: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_11, [0, 3], 0.0);  unsqueeze_11 = None
    slice_12: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_11, -1, 0, 197);  constant_pad_nd_11 = None
    expand_12: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_12, [8, 12, 197, 197]);  slice_12 = None
    _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_123, getitem_124, getitem_125, expand_12, False);  getitem_123 = getitem_124 = getitem_125 = expand_12 = None
    getitem_126: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_81: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
    view_138: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_81, [8, 197, 768]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_139: "f32[1576, 768]" = torch.ops.aten.view.default(view_138, [1576, 768]);  view_138 = None
    permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    addmm_45: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg192_1, view_139, permute_82);  arg192_1 = view_139 = permute_82 = None
    view_140: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_45, [8, 197, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_46: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg111_1, clone_46);  arg111_1 = clone_46 = None
    add_79: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_76, mul_101);  add_76 = mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_131: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_23: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_131);  getitem_131 = None
    mul_102: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_102, arg119_1);  mul_102 = arg119_1 = None
    add_81: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_103, arg120_1);  mul_103 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_141: "f32[1576, 768]" = torch.ops.aten.view.default(add_81, [1576, 768]);  add_81 = None
    permute_83: "f32[768, 3072]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    addmm_46: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg194_1, view_141, permute_83);  arg194_1 = view_141 = permute_83 = None
    view_142: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 197, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_104: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
    mul_105: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.7071067811865476);  view_142 = None
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_82: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_104, add_82);  mul_104 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_143: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_47, [1576, 3072]);  clone_47 = None
    permute_84: "f32[3072, 768]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_47: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg196_1, view_143, permute_84);  arg196_1 = view_143 = permute_84 = None
    view_144: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_47, [8, 197, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_107: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg118_1, clone_48);  arg118_1 = clone_48 = None
    add_83: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_79, mul_107);  add_79 = mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:421, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_13: "f32[8, 197, 768]" = torch.ops.aten.slice.Tensor(add_83, 0, 0, 9223372036854775807);  add_83 = None
    slice_14: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(slice_13, 1, 1, 9223372036854775807);  slice_13 = None
    mean: "f32[8, 768]" = torch.ops.aten.mean.dim(slice_14, [1]);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(mean, [1], correction = 0, keepdim = True)
    getitem_132: "f32[8, 1]" = var_mean_24[0]
    getitem_133: "f32[8, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_24: "f32[8, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_24: "f32[8, 768]" = torch.ops.aten.sub.Tensor(mean, getitem_133);  mean = getitem_133 = None
    mul_108: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    mul_109: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_108, arg121_1);  mul_108 = arg121_1 = None
    add_85: "f32[8, 768]" = torch.ops.aten.add.Tensor(mul_109, arg122_1);  mul_109 = arg122_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:423, code: x = self.head_drop(x)
    clone_49: "f32[8, 768]" = torch.ops.aten.clone.default(add_85);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:424, code: return x if pre_logits else self.head(x)
    permute_85: "f32[768, 1000]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    addmm_48: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg198_1, clone_49, permute_85);  arg198_1 = clone_49 = permute_85 = None
    return (addmm_48,)
    