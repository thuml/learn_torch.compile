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
    mul_2: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_2, 0.3535533905932738);  getitem_2 = None
    permute_4: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_3, [0, 1, 3, 2]);  getitem_3 = None
    mul_3: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_4, 0.3535533905932738);  permute_4 = None
    expand_1: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_2, [8, 12, 197, 64]);  mul_2 = None
    clone_2: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_6: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_2, [96, 197, 64]);  clone_2 = None
    expand_2: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_3, [8, 12, 64, 197]);  mul_3 = None
    clone_3: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_7: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_3, [96, 64, 197]);  clone_3 = None
    bmm: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
    view_8: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm, [8, 12, 197, 197]);  bmm = None
    add_2: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_8, unsqueeze);  view_8 = unsqueeze = None
    amax: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_2, [-1], True)
    sub_1: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_2, amax);  add_2 = amax = None
    exp: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    expand_3: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div, [8, 12, 197, 197]);  div = None
    view_9: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_3, [96, 197, 197]);  expand_3 = None
    expand_4: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_4, [8, 12, 197, 64]);  getitem_4 = None
    clone_4: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_10: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_4, [96, 197, 64]);  clone_4 = None
    bmm_1: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
    view_11: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_1, [8, 12, 197, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_5: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone_5: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_12: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_5, [8, 197, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_13: "f32[1576, 768]" = torch.ops.aten.view.default(view_12, [1576, 768]);  view_12 = None
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    addmm_1: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg126_1, view_13, permute_6);  arg126_1 = view_13 = permute_6 = None
    view_14: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_1, [8, 197, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_6: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_14);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_4: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg1_1, clone_6);  arg1_1 = clone_6 = None
    add_3: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(clone, mul_4);  clone = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_5: "f32[8, 197, 1]" = var_mean_1[0]
    getitem_6: "f32[8, 197, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-06);  getitem_5 = None
    rsqrt_1: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_6);  getitem_6 = None
    mul_5: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_6: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg9_1);  mul_5 = arg9_1 = None
    add_5: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_6, arg10_1);  mul_6 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_15: "f32[1576, 768]" = torch.ops.aten.view.default(add_5, [1576, 768]);  add_5 = None
    permute_7: "f32[768, 3072]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    addmm_2: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg128_1, view_15, permute_7);  arg128_1 = view_15 = permute_7 = None
    view_16: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 197, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
    mul_8: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476);  view_16 = None
    erf: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_6: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_6);  mul_7 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_7: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_7, [1576, 3072]);  clone_7 = None
    permute_8: "f32[3072, 768]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    addmm_3: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg130_1, view_17, permute_8);  arg130_1 = view_17 = permute_8 = None
    view_18: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_3, [8, 197, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_8: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_10: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg8_1, clone_8);  arg8_1 = clone_8 = None
    add_7: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_3, mul_10);  add_3 = mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_7: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_8: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-06);  getitem_7 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_8);  getitem_8 = None
    mul_11: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_12: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_11, arg12_1);  mul_11 = arg12_1 = None
    add_9: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_12, arg13_1);  mul_12 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_2: "f32[2304]" = torch.ops.aten.cat.default([arg14_1, arg201_1, arg15_1]);  arg14_1 = arg201_1 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_19: "f32[1576, 768]" = torch.ops.aten.view.default(add_9, [1576, 768]);  add_9 = None
    permute_9: "f32[768, 2304]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_4: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_2, view_19, permute_9);  cat_2 = view_19 = permute_9 = None
    view_20: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_4, [8, 197, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_21: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_20, [8, 197, 3, 12, -1]);  view_20 = None
    permute_10: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_21, [2, 0, 3, 1, 4]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_1 = torch.ops.aten.unbind.int(permute_10);  permute_10 = None
    getitem_9: "f32[8, 12, 197, 64]" = unbind_1[0]
    getitem_10: "f32[8, 12, 197, 64]" = unbind_1[1]
    getitem_11: "f32[8, 12, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_22: "i64[38809]" = torch.ops.aten.view.default(arg202_1, [-1]);  arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_1: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg17_1, [view_22]);  arg17_1 = view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_23: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_1, [197, 197, -1]);  index_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_11: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_23, [2, 0, 1]);  view_23 = None
    clone_9: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_1: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_9, 0);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_13: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_9, 0.3535533905932738);  getitem_9 = None
    permute_12: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
    mul_14: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
    expand_5: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_13, [8, 12, 197, 64]);  mul_13 = None
    clone_10: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_24: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_10, [96, 197, 64]);  clone_10 = None
    expand_6: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_14, [8, 12, 64, 197]);  mul_14 = None
    clone_11: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_25: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_11, [96, 64, 197]);  clone_11 = None
    bmm_2: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = view_25 = None
    view_26: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_2, [8, 12, 197, 197]);  bmm_2 = None
    add_10: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_26, unsqueeze_1);  view_26 = unsqueeze_1 = None
    amax_1: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_10, [-1], True)
    sub_4: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_10, amax_1);  add_10 = amax_1 = None
    exp_1: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    expand_7: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_1, [8, 12, 197, 197]);  div_1 = None
    view_27: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_7, [96, 197, 197]);  expand_7 = None
    expand_8: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_11, [8, 12, 197, 64]);  getitem_11 = None
    clone_12: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_28: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_12, [96, 197, 64]);  clone_12 = None
    bmm_3: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_27, view_28);  view_27 = view_28 = None
    view_29: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_3, [8, 12, 197, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_13: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_13: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_30: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_13, [8, 197, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_31: "f32[1576, 768]" = torch.ops.aten.view.default(view_30, [1576, 768]);  view_30 = None
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_5: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg132_1, view_31, permute_14);  arg132_1 = view_31 = permute_14 = None
    view_32: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_5, [8, 197, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_14: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_32);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_15: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg11_1, clone_14);  arg11_1 = clone_14 = None
    add_11: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_7, mul_15);  add_7 = mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13);  getitem_13 = None
    mul_16: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_17: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg19_1);  mul_16 = arg19_1 = None
    add_13: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_17, arg20_1);  mul_17 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_33: "f32[1576, 768]" = torch.ops.aten.view.default(add_13, [1576, 768]);  add_13 = None
    permute_15: "f32[768, 3072]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    addmm_6: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg134_1, view_33, permute_15);  arg134_1 = view_33 = permute_15 = None
    view_34: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 197, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
    mul_19: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.7071067811865476);  view_34 = None
    erf_1: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_14: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_20: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_14);  mul_18 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_15, [1576, 3072]);  clone_15 = None
    permute_16: "f32[3072, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_7: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg136_1, view_35, permute_16);  arg136_1 = view_35 = permute_16 = None
    view_36: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_7, [8, 197, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_21: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg18_1, clone_16);  arg18_1 = clone_16 = None
    add_15: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_11, mul_21);  add_11 = mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_15: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_16: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_15);  getitem_15 = None
    mul_22: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_23: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg22_1);  mul_22 = arg22_1 = None
    add_17: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_23, arg23_1);  mul_23 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_3: "f32[2304]" = torch.ops.aten.cat.default([arg24_1, arg203_1, arg25_1]);  arg24_1 = arg203_1 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_37: "f32[1576, 768]" = torch.ops.aten.view.default(add_17, [1576, 768]);  add_17 = None
    permute_17: "f32[768, 2304]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_8: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_3, view_37, permute_17);  cat_3 = view_37 = permute_17 = None
    view_38: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_8, [8, 197, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_39: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_38, [8, 197, 3, 12, -1]);  view_38 = None
    permute_18: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_39, [2, 0, 3, 1, 4]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_2 = torch.ops.aten.unbind.int(permute_18);  permute_18 = None
    getitem_16: "f32[8, 12, 197, 64]" = unbind_2[0]
    getitem_17: "f32[8, 12, 197, 64]" = unbind_2[1]
    getitem_18: "f32[8, 12, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_40: "i64[38809]" = torch.ops.aten.view.default(arg204_1, [-1]);  arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_2: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg27_1, [view_40]);  arg27_1 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_41: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_2, [197, 197, -1]);  index_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_19: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_41, [2, 0, 1]);  view_41 = None
    clone_17: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_2: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_17, 0);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_24: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_16, 0.3535533905932738);  getitem_16 = None
    permute_20: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_17, [0, 1, 3, 2]);  getitem_17 = None
    mul_25: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_20, 0.3535533905932738);  permute_20 = None
    expand_9: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_24, [8, 12, 197, 64]);  mul_24 = None
    clone_18: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_42: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_18, [96, 197, 64]);  clone_18 = None
    expand_10: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_25, [8, 12, 64, 197]);  mul_25 = None
    clone_19: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_43: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_19, [96, 64, 197]);  clone_19 = None
    bmm_4: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_42, view_43);  view_42 = view_43 = None
    view_44: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_4, [8, 12, 197, 197]);  bmm_4 = None
    add_18: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_44, unsqueeze_2);  view_44 = unsqueeze_2 = None
    amax_2: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_18, [-1], True)
    sub_7: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_18, amax_2);  add_18 = amax_2 = None
    exp_2: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    expand_11: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_2, [8, 12, 197, 197]);  div_2 = None
    view_45: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_11, [96, 197, 197]);  expand_11 = None
    expand_12: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_18, [8, 12, 197, 64]);  getitem_18 = None
    clone_20: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_46: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_20, [96, 197, 64]);  clone_20 = None
    bmm_5: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_45, view_46);  view_45 = view_46 = None
    view_47: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_5, [8, 12, 197, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_21: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
    clone_21: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_48: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_21, [8, 197, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_49: "f32[1576, 768]" = torch.ops.aten.view.default(view_48, [1576, 768]);  view_48 = None
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_9: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg138_1, view_49, permute_22);  arg138_1 = view_49 = permute_22 = None
    view_50: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_9, [8, 197, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_22: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_50);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_26: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg21_1, clone_22);  arg21_1 = clone_22 = None
    add_19: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_15, mul_26);  add_15 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_19: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_20: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_19, 1e-06);  getitem_19 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_20);  getitem_20 = None
    mul_27: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_28: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_27, arg29_1);  mul_27 = arg29_1 = None
    add_21: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_28, arg30_1);  mul_28 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_51: "f32[1576, 768]" = torch.ops.aten.view.default(add_21, [1576, 768]);  add_21 = None
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    addmm_10: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg140_1, view_51, permute_23);  arg140_1 = view_51 = permute_23 = None
    view_52: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 197, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_29: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.5)
    mul_30: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.7071067811865476);  view_52 = None
    erf_2: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
    add_22: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_31: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_29, add_22);  mul_29 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_31);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_23, [1576, 3072]);  clone_23 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    addmm_11: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg142_1, view_53, permute_24);  arg142_1 = view_53 = permute_24 = None
    view_54: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_11, [8, 197, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_32: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg28_1, clone_24);  arg28_1 = clone_24 = None
    add_23: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_19, mul_32);  add_19 = mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_21: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_22: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_24: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_21, 1e-06);  getitem_21 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_9: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_22);  getitem_22 = None
    mul_33: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_34: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg32_1);  mul_33 = arg32_1 = None
    add_25: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_34, arg33_1);  mul_34 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_4: "f32[2304]" = torch.ops.aten.cat.default([arg34_1, arg205_1, arg35_1]);  arg34_1 = arg205_1 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_55: "f32[1576, 768]" = torch.ops.aten.view.default(add_25, [1576, 768]);  add_25 = None
    permute_25: "f32[768, 2304]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_12: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_4, view_55, permute_25);  cat_4 = view_55 = permute_25 = None
    view_56: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 197, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_57: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_56, [8, 197, 3, 12, -1]);  view_56 = None
    permute_26: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_57, [2, 0, 3, 1, 4]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_3 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
    getitem_23: "f32[8, 12, 197, 64]" = unbind_3[0]
    getitem_24: "f32[8, 12, 197, 64]" = unbind_3[1]
    getitem_25: "f32[8, 12, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_58: "i64[38809]" = torch.ops.aten.view.default(arg206_1, [-1]);  arg206_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_3: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg37_1, [view_58]);  arg37_1 = view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_59: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_3, [197, 197, -1]);  index_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_27: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_59, [2, 0, 1]);  view_59 = None
    clone_25: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_3: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_25, 0);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_35: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_23, 0.3535533905932738);  getitem_23 = None
    permute_28: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_24, [0, 1, 3, 2]);  getitem_24 = None
    mul_36: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
    expand_13: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_35, [8, 12, 197, 64]);  mul_35 = None
    clone_26: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_60: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_26, [96, 197, 64]);  clone_26 = None
    expand_14: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_36, [8, 12, 64, 197]);  mul_36 = None
    clone_27: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_61: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_27, [96, 64, 197]);  clone_27 = None
    bmm_6: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_60, view_61);  view_60 = view_61 = None
    view_62: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_6, [8, 12, 197, 197]);  bmm_6 = None
    add_26: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_62, unsqueeze_3);  view_62 = unsqueeze_3 = None
    amax_3: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_26, [-1], True)
    sub_10: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_26, amax_3);  add_26 = amax_3 = None
    exp_3: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    expand_15: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_3, [8, 12, 197, 197]);  div_3 = None
    view_63: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_15, [96, 197, 197]);  expand_15 = None
    expand_16: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_25, [8, 12, 197, 64]);  getitem_25 = None
    clone_28: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_64: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_28, [96, 197, 64]);  clone_28 = None
    bmm_7: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_63, view_64);  view_63 = view_64 = None
    view_65: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_7, [8, 12, 197, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_29: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_29: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_66: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_29, [8, 197, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_67: "f32[1576, 768]" = torch.ops.aten.view.default(view_66, [1576, 768]);  view_66 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_13: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg144_1, view_67, permute_30);  arg144_1 = view_67 = permute_30 = None
    view_68: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_13, [8, 197, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_30: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_37: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg31_1, clone_30);  arg31_1 = clone_30 = None
    add_27: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_23, mul_37);  add_23 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_27: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_27);  getitem_27 = None
    mul_38: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_39: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_38, arg39_1);  mul_38 = arg39_1 = None
    add_29: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_39, arg40_1);  mul_39 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_69: "f32[1576, 768]" = torch.ops.aten.view.default(add_29, [1576, 768]);  add_29 = None
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    addmm_14: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg146_1, view_69, permute_31);  arg146_1 = view_69 = permute_31 = None
    view_70: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 197, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_40: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
    mul_41: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476);  view_70 = None
    erf_3: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_30: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_42: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_30);  mul_40 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_42);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_31, [1576, 3072]);  clone_31 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_15: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg148_1, view_71, permute_32);  arg148_1 = view_71 = permute_32 = None
    view_72: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_15, [8, 197, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_43: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg38_1, clone_32);  arg38_1 = clone_32 = None
    add_31: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_27, mul_43);  add_27 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 197, 1]" = var_mean_8[0]
    getitem_29: "f32[8, 197, 1]" = var_mean_8[1];  var_mean_8 = None
    add_32: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_8: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_12: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_29);  getitem_29 = None
    mul_44: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_45: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg42_1);  mul_44 = arg42_1 = None
    add_33: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_45, arg43_1);  mul_45 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_5: "f32[2304]" = torch.ops.aten.cat.default([arg44_1, arg207_1, arg45_1]);  arg44_1 = arg207_1 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_73: "f32[1576, 768]" = torch.ops.aten.view.default(add_33, [1576, 768]);  add_33 = None
    permute_33: "f32[768, 2304]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_16: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_5, view_73, permute_33);  cat_5 = view_73 = permute_33 = None
    view_74: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_16, [8, 197, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_75: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_74, [8, 197, 3, 12, -1]);  view_74 = None
    permute_34: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_75, [2, 0, 3, 1, 4]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_4 = torch.ops.aten.unbind.int(permute_34);  permute_34 = None
    getitem_30: "f32[8, 12, 197, 64]" = unbind_4[0]
    getitem_31: "f32[8, 12, 197, 64]" = unbind_4[1]
    getitem_32: "f32[8, 12, 197, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_76: "i64[38809]" = torch.ops.aten.view.default(arg208_1, [-1]);  arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_4: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg47_1, [view_76]);  arg47_1 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_77: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_4, [197, 197, -1]);  index_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_35: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_77, [2, 0, 1]);  view_77 = None
    clone_33: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_4: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_33, 0);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_46: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_30, 0.3535533905932738);  getitem_30 = None
    permute_36: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_31, [0, 1, 3, 2]);  getitem_31 = None
    mul_47: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_36, 0.3535533905932738);  permute_36 = None
    expand_17: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_46, [8, 12, 197, 64]);  mul_46 = None
    clone_34: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_78: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_34, [96, 197, 64]);  clone_34 = None
    expand_18: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_47, [8, 12, 64, 197]);  mul_47 = None
    clone_35: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_79: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_35, [96, 64, 197]);  clone_35 = None
    bmm_8: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
    view_80: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_8, [8, 12, 197, 197]);  bmm_8 = None
    add_34: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_80, unsqueeze_4);  view_80 = unsqueeze_4 = None
    amax_4: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_34, [-1], True)
    sub_13: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_34, amax_4);  add_34 = amax_4 = None
    exp_4: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    expand_19: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_4, [8, 12, 197, 197]);  div_4 = None
    view_81: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_19, [96, 197, 197]);  expand_19 = None
    expand_20: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_32, [8, 12, 197, 64]);  getitem_32 = None
    clone_36: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_82: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_36, [96, 197, 64]);  clone_36 = None
    bmm_9: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_81, view_82);  view_81 = view_82 = None
    view_83: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_9, [8, 12, 197, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_37: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
    clone_37: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_84: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_37, [8, 197, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_85: "f32[1576, 768]" = torch.ops.aten.view.default(view_84, [1576, 768]);  view_84 = None
    permute_38: "f32[768, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_17: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg150_1, view_85, permute_38);  arg150_1 = view_85 = permute_38 = None
    view_86: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_17, [8, 197, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_38: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_86);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_48: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg41_1, clone_38);  arg41_1 = clone_38 = None
    add_35: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_31, mul_48);  add_31 = mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 197, 1]" = var_mean_9[0]
    getitem_34: "f32[8, 197, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_9: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_34);  getitem_34 = None
    mul_49: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_50: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg49_1);  mul_49 = arg49_1 = None
    add_37: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_50, arg50_1);  mul_50 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_87: "f32[1576, 768]" = torch.ops.aten.view.default(add_37, [1576, 768]);  add_37 = None
    permute_39: "f32[768, 3072]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_18: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg152_1, view_87, permute_39);  arg152_1 = view_87 = permute_39 = None
    view_88: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 197, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_51: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
    mul_52: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476);  view_88 = None
    erf_4: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
    add_38: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_53: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_38);  mul_51 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_53);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_89: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_39, [1576, 3072]);  clone_39 = None
    permute_40: "f32[3072, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_19: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg154_1, view_89, permute_40);  arg154_1 = view_89 = permute_40 = None
    view_90: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_19, [8, 197, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_54: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg48_1, clone_40);  arg48_1 = clone_40 = None
    add_39: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_35, mul_54);  add_35 = mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_35: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_36: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_40: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_35, 1e-06);  getitem_35 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_15: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_36);  getitem_36 = None
    mul_55: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_56: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_55, arg52_1);  mul_55 = arg52_1 = None
    add_41: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_56, arg53_1);  mul_56 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_6: "f32[2304]" = torch.ops.aten.cat.default([arg54_1, arg209_1, arg55_1]);  arg54_1 = arg209_1 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_91: "f32[1576, 768]" = torch.ops.aten.view.default(add_41, [1576, 768]);  add_41 = None
    permute_41: "f32[768, 2304]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_20: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_6, view_91, permute_41);  cat_6 = view_91 = permute_41 = None
    view_92: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_20, [8, 197, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_93: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_92, [8, 197, 3, 12, -1]);  view_92 = None
    permute_42: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_5 = torch.ops.aten.unbind.int(permute_42);  permute_42 = None
    getitem_37: "f32[8, 12, 197, 64]" = unbind_5[0]
    getitem_38: "f32[8, 12, 197, 64]" = unbind_5[1]
    getitem_39: "f32[8, 12, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_94: "i64[38809]" = torch.ops.aten.view.default(arg210_1, [-1]);  arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_5: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg57_1, [view_94]);  arg57_1 = view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_95: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_5, [197, 197, -1]);  index_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_43: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_95, [2, 0, 1]);  view_95 = None
    clone_41: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_5: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_41, 0);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_57: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_37, 0.3535533905932738);  getitem_37 = None
    permute_44: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_38, [0, 1, 3, 2]);  getitem_38 = None
    mul_58: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_44, 0.3535533905932738);  permute_44 = None
    expand_21: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_57, [8, 12, 197, 64]);  mul_57 = None
    clone_42: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_96: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_42, [96, 197, 64]);  clone_42 = None
    expand_22: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_58, [8, 12, 64, 197]);  mul_58 = None
    clone_43: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_97: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_43, [96, 64, 197]);  clone_43 = None
    bmm_10: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_96, view_97);  view_96 = view_97 = None
    view_98: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_10, [8, 12, 197, 197]);  bmm_10 = None
    add_42: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_98, unsqueeze_5);  view_98 = unsqueeze_5 = None
    amax_5: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_42, [-1], True)
    sub_16: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_42, amax_5);  add_42 = amax_5 = None
    exp_5: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    expand_23: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_5, [8, 12, 197, 197]);  div_5 = None
    view_99: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_23, [96, 197, 197]);  expand_23 = None
    expand_24: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_39, [8, 12, 197, 64]);  getitem_39 = None
    clone_44: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_100: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_44, [96, 197, 64]);  clone_44 = None
    bmm_11: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
    view_101: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_11, [8, 12, 197, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_45: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
    clone_45: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_102: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_45, [8, 197, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_103: "f32[1576, 768]" = torch.ops.aten.view.default(view_102, [1576, 768]);  view_102 = None
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    addmm_21: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg156_1, view_103, permute_46);  arg156_1 = view_103 = permute_46 = None
    view_104: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_21, [8, 197, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_46: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_59: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg51_1, clone_46);  arg51_1 = clone_46 = None
    add_43: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_39, mul_59);  add_39 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_41: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_41);  getitem_41 = None
    mul_60: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_61: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_60, arg59_1);  mul_60 = arg59_1 = None
    add_45: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_61, arg60_1);  mul_61 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[1576, 768]" = torch.ops.aten.view.default(add_45, [1576, 768]);  add_45 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_22: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg158_1, view_105, permute_47);  arg158_1 = view_105 = permute_47 = None
    view_106: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 197, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_62: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_63: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476);  view_106 = None
    erf_5: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_46: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_64: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_46);  mul_62 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_47, [1576, 3072]);  clone_47 = None
    permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_23: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg160_1, view_107, permute_48);  arg160_1 = view_107 = permute_48 = None
    view_108: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_23, [8, 197, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_65: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg58_1, clone_48);  arg58_1 = clone_48 = None
    add_47: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_43, mul_65);  add_43 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 197, 1]" = var_mean_12[0]
    getitem_43: "f32[8, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    add_48: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_12: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_18: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_43);  getitem_43 = None
    mul_66: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_67: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg62_1);  mul_66 = arg62_1 = None
    add_49: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_67, arg63_1);  mul_67 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_7: "f32[2304]" = torch.ops.aten.cat.default([arg64_1, arg211_1, arg65_1]);  arg64_1 = arg211_1 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_109: "f32[1576, 768]" = torch.ops.aten.view.default(add_49, [1576, 768]);  add_49 = None
    permute_49: "f32[768, 2304]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    addmm_24: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_7, view_109, permute_49);  cat_7 = view_109 = permute_49 = None
    view_110: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 197, 2304]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_111: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_110, [8, 197, 3, 12, -1]);  view_110 = None
    permute_50: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_6 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_44: "f32[8, 12, 197, 64]" = unbind_6[0]
    getitem_45: "f32[8, 12, 197, 64]" = unbind_6[1]
    getitem_46: "f32[8, 12, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_112: "i64[38809]" = torch.ops.aten.view.default(arg212_1, [-1]);  arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_6: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg67_1, [view_112]);  arg67_1 = view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_113: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_6, [197, 197, -1]);  index_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_51: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_113, [2, 0, 1]);  view_113 = None
    clone_49: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_6: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_49, 0);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_68: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_44, 0.3535533905932738);  getitem_44 = None
    permute_52: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_45, [0, 1, 3, 2]);  getitem_45 = None
    mul_69: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_52, 0.3535533905932738);  permute_52 = None
    expand_25: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_68, [8, 12, 197, 64]);  mul_68 = None
    clone_50: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_114: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_50, [96, 197, 64]);  clone_50 = None
    expand_26: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_69, [8, 12, 64, 197]);  mul_69 = None
    clone_51: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_115: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_51, [96, 64, 197]);  clone_51 = None
    bmm_12: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
    view_116: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_12, [8, 12, 197, 197]);  bmm_12 = None
    add_50: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_116, unsqueeze_6);  view_116 = unsqueeze_6 = None
    amax_6: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_50, [-1], True)
    sub_19: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_50, amax_6);  add_50 = amax_6 = None
    exp_6: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    expand_27: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_6, [8, 12, 197, 197]);  div_6 = None
    view_117: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_27, [96, 197, 197]);  expand_27 = None
    expand_28: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_46, [8, 12, 197, 64]);  getitem_46 = None
    clone_52: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_118: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_52, [96, 197, 64]);  clone_52 = None
    bmm_13: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_117, view_118);  view_117 = view_118 = None
    view_119: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_13, [8, 12, 197, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_53: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    clone_53: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    view_120: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_53, [8, 197, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_121: "f32[1576, 768]" = torch.ops.aten.view.default(view_120, [1576, 768]);  view_120 = None
    permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    addmm_25: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg162_1, view_121, permute_54);  arg162_1 = view_121 = permute_54 = None
    view_122: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_25, [8, 197, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_54: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_122);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_70: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg61_1, clone_54);  arg61_1 = clone_54 = None
    add_51: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_47, mul_70);  add_47 = mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_47: "f32[8, 197, 1]" = var_mean_13[0]
    getitem_48: "f32[8, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    add_52: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-06);  getitem_47 = None
    rsqrt_13: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_48);  getitem_48 = None
    mul_71: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_72: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_71, arg69_1);  mul_71 = arg69_1 = None
    add_53: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_72, arg70_1);  mul_72 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_123: "f32[1576, 768]" = torch.ops.aten.view.default(add_53, [1576, 768]);  add_53 = None
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_26: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg164_1, view_123, permute_55);  arg164_1 = view_123 = permute_55 = None
    view_124: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 197, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_73: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.5)
    mul_74: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.7071067811865476);  view_124 = None
    erf_6: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_54: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_75: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_73, add_54);  mul_73 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_55: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_75);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_125: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_55, [1576, 3072]);  clone_55 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_27: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg166_1, view_125, permute_56);  arg166_1 = view_125 = permute_56 = None
    view_126: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_27, [8, 197, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_76: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg68_1, clone_56);  arg68_1 = clone_56 = None
    add_55: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_51, mul_76);  add_51 = mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_49: "f32[8, 197, 1]" = var_mean_14[0]
    getitem_50: "f32[8, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    add_56: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-06);  getitem_49 = None
    rsqrt_14: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_21: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_50);  getitem_50 = None
    mul_77: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_78: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_77, arg72_1);  mul_77 = arg72_1 = None
    add_57: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_78, arg73_1);  mul_78 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_8: "f32[2304]" = torch.ops.aten.cat.default([arg74_1, arg213_1, arg75_1]);  arg74_1 = arg213_1 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_127: "f32[1576, 768]" = torch.ops.aten.view.default(add_57, [1576, 768]);  add_57 = None
    permute_57: "f32[768, 2304]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    addmm_28: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_8, view_127, permute_57);  cat_8 = view_127 = permute_57 = None
    view_128: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_28, [8, 197, 2304]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_129: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_128, [8, 197, 3, 12, -1]);  view_128 = None
    permute_58: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_129, [2, 0, 3, 1, 4]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_7 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
    getitem_51: "f32[8, 12, 197, 64]" = unbind_7[0]
    getitem_52: "f32[8, 12, 197, 64]" = unbind_7[1]
    getitem_53: "f32[8, 12, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_130: "i64[38809]" = torch.ops.aten.view.default(arg214_1, [-1]);  arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_7: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg77_1, [view_130]);  arg77_1 = view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_131: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_7, [197, 197, -1]);  index_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_59: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_131, [2, 0, 1]);  view_131 = None
    clone_57: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_7: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_57, 0);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_79: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_51, 0.3535533905932738);  getitem_51 = None
    permute_60: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_52, [0, 1, 3, 2]);  getitem_52 = None
    mul_80: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_60, 0.3535533905932738);  permute_60 = None
    expand_29: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_79, [8, 12, 197, 64]);  mul_79 = None
    clone_58: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_132: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_58, [96, 197, 64]);  clone_58 = None
    expand_30: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_80, [8, 12, 64, 197]);  mul_80 = None
    clone_59: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_133: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_59, [96, 64, 197]);  clone_59 = None
    bmm_14: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_132, view_133);  view_132 = view_133 = None
    view_134: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_14, [8, 12, 197, 197]);  bmm_14 = None
    add_58: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_134, unsqueeze_7);  view_134 = unsqueeze_7 = None
    amax_7: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_58, [-1], True)
    sub_22: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_58, amax_7);  add_58 = amax_7 = None
    exp_7: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    expand_31: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_7, [8, 12, 197, 197]);  div_7 = None
    view_135: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_31, [96, 197, 197]);  expand_31 = None
    expand_32: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_53, [8, 12, 197, 64]);  getitem_53 = None
    clone_60: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_136: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_60, [96, 197, 64]);  clone_60 = None
    bmm_15: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
    view_137: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_15, [8, 12, 197, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_61: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    clone_61: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    view_138: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_61, [8, 197, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_139: "f32[1576, 768]" = torch.ops.aten.view.default(view_138, [1576, 768]);  view_138 = None
    permute_62: "f32[768, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_29: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg168_1, view_139, permute_62);  arg168_1 = view_139 = permute_62 = None
    view_140: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_29, [8, 197, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_62: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_81: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg71_1, clone_62);  arg71_1 = clone_62 = None
    add_59: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_55, mul_81);  add_55 = mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_15[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_15: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_55);  getitem_55 = None
    mul_82: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_83: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg79_1);  mul_82 = arg79_1 = None
    add_61: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_83, arg80_1);  mul_83 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_141: "f32[1576, 768]" = torch.ops.aten.view.default(add_61, [1576, 768]);  add_61 = None
    permute_63: "f32[768, 3072]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_30: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg170_1, view_141, permute_63);  arg170_1 = view_141 = permute_63 = None
    view_142: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 197, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_84: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
    mul_85: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.7071067811865476);  view_142 = None
    erf_7: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_85);  mul_85 = None
    add_62: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_86: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_62);  mul_84 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_63: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_86);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_143: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_63, [1576, 3072]);  clone_63 = None
    permute_64: "f32[3072, 768]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    addmm_31: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg172_1, view_143, permute_64);  arg172_1 = view_143 = permute_64 = None
    view_144: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_31, [8, 197, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_64: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_87: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg78_1, clone_64);  arg78_1 = clone_64 = None
    add_63: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_59, mul_87);  add_59 = mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_57: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_64: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_24: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_57);  getitem_57 = None
    mul_88: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_89: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg82_1);  mul_88 = arg82_1 = None
    add_65: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_89, arg83_1);  mul_89 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_9: "f32[2304]" = torch.ops.aten.cat.default([arg84_1, arg215_1, arg85_1]);  arg84_1 = arg215_1 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_145: "f32[1576, 768]" = torch.ops.aten.view.default(add_65, [1576, 768]);  add_65 = None
    permute_65: "f32[768, 2304]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_32: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_9, view_145, permute_65);  cat_9 = view_145 = permute_65 = None
    view_146: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_32, [8, 197, 2304]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_147: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_146, [8, 197, 3, 12, -1]);  view_146 = None
    permute_66: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_147, [2, 0, 3, 1, 4]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_8 = torch.ops.aten.unbind.int(permute_66);  permute_66 = None
    getitem_58: "f32[8, 12, 197, 64]" = unbind_8[0]
    getitem_59: "f32[8, 12, 197, 64]" = unbind_8[1]
    getitem_60: "f32[8, 12, 197, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_148: "i64[38809]" = torch.ops.aten.view.default(arg216_1, [-1]);  arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_8: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg87_1, [view_148]);  arg87_1 = view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_149: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_8, [197, 197, -1]);  index_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_67: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_149, [2, 0, 1]);  view_149 = None
    clone_65: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_8: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_65, 0);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_90: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_58, 0.3535533905932738);  getitem_58 = None
    permute_68: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_59, [0, 1, 3, 2]);  getitem_59 = None
    mul_91: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_68, 0.3535533905932738);  permute_68 = None
    expand_33: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_90, [8, 12, 197, 64]);  mul_90 = None
    clone_66: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_150: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_66, [96, 197, 64]);  clone_66 = None
    expand_34: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_91, [8, 12, 64, 197]);  mul_91 = None
    clone_67: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_151: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_67, [96, 64, 197]);  clone_67 = None
    bmm_16: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_150, view_151);  view_150 = view_151 = None
    view_152: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_16, [8, 12, 197, 197]);  bmm_16 = None
    add_66: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_152, unsqueeze_8);  view_152 = unsqueeze_8 = None
    amax_8: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_66, [-1], True)
    sub_25: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_66, amax_8);  add_66 = amax_8 = None
    exp_8: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    expand_35: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_8, [8, 12, 197, 197]);  div_8 = None
    view_153: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_35, [96, 197, 197]);  expand_35 = None
    expand_36: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_60, [8, 12, 197, 64]);  getitem_60 = None
    clone_68: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_154: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_68, [96, 197, 64]);  clone_68 = None
    bmm_17: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_153, view_154);  view_153 = view_154 = None
    view_155: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_17, [8, 12, 197, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_69: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
    clone_69: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_156: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_69, [8, 197, 768]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_157: "f32[1576, 768]" = torch.ops.aten.view.default(view_156, [1576, 768]);  view_156 = None
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_33: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg174_1, view_157, permute_70);  arg174_1 = view_157 = permute_70 = None
    view_158: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_33, [8, 197, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_70: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_158);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_92: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg81_1, clone_70);  arg81_1 = clone_70 = None
    add_67: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_63, mul_92);  add_63 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_61: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_62: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_68: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_61, 1e-06);  getitem_61 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_62);  getitem_62 = None
    mul_93: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, arg89_1);  mul_93 = arg89_1 = None
    add_69: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_94, arg90_1);  mul_94 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_159: "f32[1576, 768]" = torch.ops.aten.view.default(add_69, [1576, 768]);  add_69 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg176_1, view_159, permute_71);  arg176_1 = view_159 = permute_71 = None
    view_160: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.5)
    mul_96: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.7071067811865476);  view_160 = None
    erf_8: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_70: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_97: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_95, add_70);  mul_95 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_71: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_161: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_71, [1576, 3072]);  clone_71 = None
    permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    addmm_35: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg178_1, view_161, permute_72);  arg178_1 = view_161 = permute_72 = None
    view_162: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_35, [8, 197, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_72: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_98: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg88_1, clone_72);  arg88_1 = clone_72 = None
    add_71: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_67, mul_98);  add_67 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_63: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_64: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_72: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-06);  getitem_63 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_27: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_64);  getitem_64 = None
    mul_99: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_99, arg92_1);  mul_99 = arg92_1 = None
    add_73: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_100, arg93_1);  mul_100 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_10: "f32[2304]" = torch.ops.aten.cat.default([arg94_1, arg217_1, arg95_1]);  arg94_1 = arg217_1 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_163: "f32[1576, 768]" = torch.ops.aten.view.default(add_73, [1576, 768]);  add_73 = None
    permute_73: "f32[768, 2304]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_36: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_10, view_163, permute_73);  cat_10 = view_163 = permute_73 = None
    view_164: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 197, 2304]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_165: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_164, [8, 197, 3, 12, -1]);  view_164 = None
    permute_74: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_165, [2, 0, 3, 1, 4]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_9 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_65: "f32[8, 12, 197, 64]" = unbind_9[0]
    getitem_66: "f32[8, 12, 197, 64]" = unbind_9[1]
    getitem_67: "f32[8, 12, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_166: "i64[38809]" = torch.ops.aten.view.default(arg218_1, [-1]);  arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_9: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg97_1, [view_166]);  arg97_1 = view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_167: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_9, [197, 197, -1]);  index_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_75: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_167, [2, 0, 1]);  view_167 = None
    clone_73: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_9: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_73, 0);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_101: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_65, 0.3535533905932738);  getitem_65 = None
    permute_76: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_66, [0, 1, 3, 2]);  getitem_66 = None
    mul_102: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_76, 0.3535533905932738);  permute_76 = None
    expand_37: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_101, [8, 12, 197, 64]);  mul_101 = None
    clone_74: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_168: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_74, [96, 197, 64]);  clone_74 = None
    expand_38: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_102, [8, 12, 64, 197]);  mul_102 = None
    clone_75: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_169: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_75, [96, 64, 197]);  clone_75 = None
    bmm_18: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
    view_170: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_18, [8, 12, 197, 197]);  bmm_18 = None
    add_74: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_170, unsqueeze_9);  view_170 = unsqueeze_9 = None
    amax_9: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_74, [-1], True)
    sub_28: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_74, amax_9);  add_74 = amax_9 = None
    exp_9: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    expand_39: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_9, [8, 12, 197, 197]);  div_9 = None
    view_171: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_39, [96, 197, 197]);  expand_39 = None
    expand_40: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_67, [8, 12, 197, 64]);  getitem_67 = None
    clone_76: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_172: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_76, [96, 197, 64]);  clone_76 = None
    bmm_19: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_171, view_172);  view_171 = view_172 = None
    view_173: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_19, [8, 12, 197, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_77: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    clone_77: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_174: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_77, [8, 197, 768]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_175: "f32[1576, 768]" = torch.ops.aten.view.default(view_174, [1576, 768]);  view_174 = None
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_37: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg180_1, view_175, permute_78);  arg180_1 = view_175 = permute_78 = None
    view_176: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_37, [8, 197, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_78: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_176);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg91_1, clone_78);  arg91_1 = clone_78 = None
    add_75: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_71, mul_103);  add_71 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_69: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_76: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_69);  getitem_69 = None
    mul_104: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_105: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_104, arg99_1);  mul_104 = arg99_1 = None
    add_77: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_105, arg100_1);  mul_105 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_177: "f32[1576, 768]" = torch.ops.aten.view.default(add_77, [1576, 768]);  add_77 = None
    permute_79: "f32[768, 3072]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_38: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg182_1, view_177, permute_79);  arg182_1 = view_177 = permute_79 = None
    view_178: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 197, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    mul_107: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476);  view_178 = None
    erf_9: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_78: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_108: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_106, add_78);  mul_106 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_79: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_108);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_179: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_79, [1576, 3072]);  clone_79 = None
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_39: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg184_1, view_179, permute_80);  arg184_1 = view_179 = permute_80 = None
    view_180: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_39, [8, 197, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_80: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg98_1, clone_80);  arg98_1 = clone_80 = None
    add_79: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_75, mul_109);  add_75 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_71: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_80: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_30: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_71);  getitem_71 = None
    mul_110: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_111: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_110, arg102_1);  mul_110 = arg102_1 = None
    add_81: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_111, arg103_1);  mul_111 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_11: "f32[2304]" = torch.ops.aten.cat.default([arg104_1, arg219_1, arg105_1]);  arg104_1 = arg219_1 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_181: "f32[1576, 768]" = torch.ops.aten.view.default(add_81, [1576, 768]);  add_81 = None
    permute_81: "f32[768, 2304]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_40: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_11, view_181, permute_81);  cat_11 = view_181 = permute_81 = None
    view_182: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_40, [8, 197, 2304]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_183: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_182, [8, 197, 3, 12, -1]);  view_182 = None
    permute_82: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_183, [2, 0, 3, 1, 4]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_10 = torch.ops.aten.unbind.int(permute_82);  permute_82 = None
    getitem_72: "f32[8, 12, 197, 64]" = unbind_10[0]
    getitem_73: "f32[8, 12, 197, 64]" = unbind_10[1]
    getitem_74: "f32[8, 12, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_184: "i64[38809]" = torch.ops.aten.view.default(arg220_1, [-1]);  arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_10: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg107_1, [view_184]);  arg107_1 = view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_185: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_10, [197, 197, -1]);  index_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_83: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_185, [2, 0, 1]);  view_185 = None
    clone_81: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_10: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_81, 0);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_112: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_72, 0.3535533905932738);  getitem_72 = None
    permute_84: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_73, [0, 1, 3, 2]);  getitem_73 = None
    mul_113: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_84, 0.3535533905932738);  permute_84 = None
    expand_41: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_112, [8, 12, 197, 64]);  mul_112 = None
    clone_82: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_186: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_82, [96, 197, 64]);  clone_82 = None
    expand_42: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_113, [8, 12, 64, 197]);  mul_113 = None
    clone_83: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_187: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_83, [96, 64, 197]);  clone_83 = None
    bmm_20: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_186, view_187);  view_186 = view_187 = None
    view_188: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_20, [8, 12, 197, 197]);  bmm_20 = None
    add_82: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_188, unsqueeze_10);  view_188 = unsqueeze_10 = None
    amax_10: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_82, [-1], True)
    sub_31: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_82, amax_10);  add_82 = amax_10 = None
    exp_10: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    expand_43: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_10, [8, 12, 197, 197]);  div_10 = None
    view_189: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_43, [96, 197, 197]);  expand_43 = None
    expand_44: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_74, [8, 12, 197, 64]);  getitem_74 = None
    clone_84: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_190: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_84, [96, 197, 64]);  clone_84 = None
    bmm_21: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_189, view_190);  view_189 = view_190 = None
    view_191: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_21, [8, 12, 197, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_85: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    clone_85: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_192: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_85, [8, 197, 768]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_193: "f32[1576, 768]" = torch.ops.aten.view.default(view_192, [1576, 768]);  view_192 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_41: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg186_1, view_193, permute_86);  arg186_1 = view_193 = permute_86 = None
    view_194: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_41, [8, 197, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_86: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_194);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_114: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg101_1, clone_86);  arg101_1 = clone_86 = None
    add_83: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_79, mul_114);  add_79 = mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_75: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_76: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_84: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_76);  getitem_76 = None
    mul_115: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_116: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_115, arg109_1);  mul_115 = arg109_1 = None
    add_85: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_116, arg110_1);  mul_116 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_195: "f32[1576, 768]" = torch.ops.aten.view.default(add_85, [1576, 768]);  add_85 = None
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    addmm_42: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg188_1, view_195, permute_87);  arg188_1 = view_195 = permute_87 = None
    view_196: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 197, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_117: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.5)
    mul_118: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476);  view_196 = None
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_86: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_119: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_117, add_86);  mul_117 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_87: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_119);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_197: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_87, [1576, 3072]);  clone_87 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_43: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg190_1, view_197, permute_88);  arg190_1 = view_197 = permute_88 = None
    view_198: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_43, [8, 197, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_88: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_198);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_120: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg108_1, clone_88);  arg108_1 = clone_88 = None
    add_87: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_83, mul_120);  add_83 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_78: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_88: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_33: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_78);  getitem_78 = None
    mul_121: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_122: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_121, arg112_1);  mul_121 = arg112_1 = None
    add_89: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_122, arg113_1);  mul_122 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_12: "f32[2304]" = torch.ops.aten.cat.default([arg114_1, arg221_1, arg115_1]);  arg114_1 = arg221_1 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_199: "f32[1576, 768]" = torch.ops.aten.view.default(add_89, [1576, 768]);  add_89 = None
    permute_89: "f32[768, 2304]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_44: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_12, view_199, permute_89);  cat_12 = view_199 = permute_89 = None
    view_200: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_44, [8, 197, 2304]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_201: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_200, [8, 197, 3, 12, -1]);  view_200 = None
    permute_90: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_11 = torch.ops.aten.unbind.int(permute_90);  permute_90 = None
    getitem_79: "f32[8, 12, 197, 64]" = unbind_11[0]
    getitem_80: "f32[8, 12, 197, 64]" = unbind_11[1]
    getitem_81: "f32[8, 12, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_202: "i64[38809]" = torch.ops.aten.view.default(arg222_1, [-1]);  arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_11: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg117_1, [view_202]);  arg117_1 = view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_203: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_11, [197, 197, -1]);  index_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_91: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_203, [2, 0, 1]);  view_203 = None
    clone_89: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_11: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_89, 0);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_123: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_79, 0.3535533905932738);  getitem_79 = None
    permute_92: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_80, [0, 1, 3, 2]);  getitem_80 = None
    mul_124: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_92, 0.3535533905932738);  permute_92 = None
    expand_45: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_123, [8, 12, 197, 64]);  mul_123 = None
    clone_90: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_204: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_90, [96, 197, 64]);  clone_90 = None
    expand_46: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_124, [8, 12, 64, 197]);  mul_124 = None
    clone_91: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_205: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_91, [96, 64, 197]);  clone_91 = None
    bmm_22: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
    view_206: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_22, [8, 12, 197, 197]);  bmm_22 = None
    add_90: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_206, unsqueeze_11);  view_206 = unsqueeze_11 = None
    amax_11: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_90, [-1], True)
    sub_34: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_90, amax_11);  add_90 = amax_11 = None
    exp_11: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    expand_47: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_11, [8, 12, 197, 197]);  div_11 = None
    view_207: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_47, [96, 197, 197]);  expand_47 = None
    expand_48: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_81, [8, 12, 197, 64]);  getitem_81 = None
    clone_92: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_208: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_92, [96, 197, 64]);  clone_92 = None
    bmm_23: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_207, view_208);  view_207 = view_208 = None
    view_209: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_23, [8, 12, 197, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_93: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
    clone_93: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    view_210: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_93, [8, 197, 768]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_211: "f32[1576, 768]" = torch.ops.aten.view.default(view_210, [1576, 768]);  view_210 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    addmm_45: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg192_1, view_211, permute_94);  arg192_1 = view_211 = permute_94 = None
    view_212: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_45, [8, 197, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_94: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_212);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_125: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg111_1, clone_94);  arg111_1 = clone_94 = None
    add_91: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_87, mul_125);  add_87 = mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_83: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_92: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_83);  getitem_83 = None
    mul_126: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_127: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_126, arg119_1);  mul_126 = arg119_1 = None
    add_93: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_127, arg120_1);  mul_127 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_213: "f32[1576, 768]" = torch.ops.aten.view.default(add_93, [1576, 768]);  add_93 = None
    permute_95: "f32[768, 3072]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    addmm_46: "f32[1576, 3072]" = torch.ops.aten.addmm.default(arg194_1, view_213, permute_95);  arg194_1 = view_213 = permute_95 = None
    view_214: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 197, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_128: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.5)
    mul_129: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.7071067811865476);  view_214 = None
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_94: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_130: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_128, add_94);  mul_128 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_95: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_130);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_215: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_95, [1576, 3072]);  clone_95 = None
    permute_96: "f32[3072, 768]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_47: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg196_1, view_215, permute_96);  arg196_1 = view_215 = permute_96 = None
    view_216: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_47, [8, 197, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_96: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_216);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_131: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(arg118_1, clone_96);  arg118_1 = clone_96 = None
    add_95: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_91, mul_131);  add_91 = mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:421, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_1: "f32[8, 197, 768]" = torch.ops.aten.slice.Tensor(add_95, 0, 0, 9223372036854775807);  add_95 = None
    slice_2: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(slice_1, 1, 1, 9223372036854775807);  slice_1 = None
    mean: "f32[8, 768]" = torch.ops.aten.mean.dim(slice_2, [1]);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(mean, [1], correction = 0, keepdim = True)
    getitem_84: "f32[8, 1]" = var_mean_24[0]
    getitem_85: "f32[8, 1]" = var_mean_24[1];  var_mean_24 = None
    add_96: "f32[8, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_24: "f32[8, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_36: "f32[8, 768]" = torch.ops.aten.sub.Tensor(mean, getitem_85);  mean = getitem_85 = None
    mul_132: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_133: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_132, arg121_1);  mul_132 = arg121_1 = None
    add_97: "f32[8, 768]" = torch.ops.aten.add.Tensor(mul_133, arg122_1);  mul_133 = arg122_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:423, code: x = self.head_drop(x)
    clone_97: "f32[8, 768]" = torch.ops.aten.clone.default(add_97);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:424, code: return x if pre_logits else self.head(x)
    permute_97: "f32[768, 1000]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    addmm_48: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg198_1, clone_97, permute_97);  arg198_1 = clone_97 = permute_97 = None
    return (addmm_48,)
    