from __future__ import annotations



def forward(self, arg0_1: "f32[1, 197, 384]", arg1_1: "f32[1, 1, 384]", arg2_1: "f32[384, 3, 16, 16]", arg3_1: "f32[384]", arg4_1: "f32[384]", arg5_1: "f32[384]", arg6_1: "f32[1152, 384]", arg7_1: "f32[1152]", arg8_1: "f32[384, 384]", arg9_1: "f32[384]", arg10_1: "f32[384]", arg11_1: "f32[384]", arg12_1: "f32[1536, 384]", arg13_1: "f32[1536]", arg14_1: "f32[384, 1536]", arg15_1: "f32[384]", arg16_1: "f32[384]", arg17_1: "f32[384]", arg18_1: "f32[1152, 384]", arg19_1: "f32[1152]", arg20_1: "f32[384, 384]", arg21_1: "f32[384]", arg22_1: "f32[384]", arg23_1: "f32[384]", arg24_1: "f32[1536, 384]", arg25_1: "f32[1536]", arg26_1: "f32[384, 1536]", arg27_1: "f32[384]", arg28_1: "f32[384]", arg29_1: "f32[384]", arg30_1: "f32[1152, 384]", arg31_1: "f32[1152]", arg32_1: "f32[384, 384]", arg33_1: "f32[384]", arg34_1: "f32[384]", arg35_1: "f32[384]", arg36_1: "f32[1536, 384]", arg37_1: "f32[1536]", arg38_1: "f32[384, 1536]", arg39_1: "f32[384]", arg40_1: "f32[384]", arg41_1: "f32[384]", arg42_1: "f32[1152, 384]", arg43_1: "f32[1152]", arg44_1: "f32[384, 384]", arg45_1: "f32[384]", arg46_1: "f32[384]", arg47_1: "f32[384]", arg48_1: "f32[1536, 384]", arg49_1: "f32[1536]", arg50_1: "f32[384, 1536]", arg51_1: "f32[384]", arg52_1: "f32[384]", arg53_1: "f32[384]", arg54_1: "f32[1152, 384]", arg55_1: "f32[1152]", arg56_1: "f32[384, 384]", arg57_1: "f32[384]", arg58_1: "f32[384]", arg59_1: "f32[384]", arg60_1: "f32[1536, 384]", arg61_1: "f32[1536]", arg62_1: "f32[384, 1536]", arg63_1: "f32[384]", arg64_1: "f32[384]", arg65_1: "f32[384]", arg66_1: "f32[1152, 384]", arg67_1: "f32[1152]", arg68_1: "f32[384, 384]", arg69_1: "f32[384]", arg70_1: "f32[384]", arg71_1: "f32[384]", arg72_1: "f32[1536, 384]", arg73_1: "f32[1536]", arg74_1: "f32[384, 1536]", arg75_1: "f32[384]", arg76_1: "f32[384]", arg77_1: "f32[384]", arg78_1: "f32[1152, 384]", arg79_1: "f32[1152]", arg80_1: "f32[384, 384]", arg81_1: "f32[384]", arg82_1: "f32[384]", arg83_1: "f32[384]", arg84_1: "f32[1536, 384]", arg85_1: "f32[1536]", arg86_1: "f32[384, 1536]", arg87_1: "f32[384]", arg88_1: "f32[384]", arg89_1: "f32[384]", arg90_1: "f32[1152, 384]", arg91_1: "f32[1152]", arg92_1: "f32[384, 384]", arg93_1: "f32[384]", arg94_1: "f32[384]", arg95_1: "f32[384]", arg96_1: "f32[1536, 384]", arg97_1: "f32[1536]", arg98_1: "f32[384, 1536]", arg99_1: "f32[384]", arg100_1: "f32[384]", arg101_1: "f32[384]", arg102_1: "f32[1152, 384]", arg103_1: "f32[1152]", arg104_1: "f32[384, 384]", arg105_1: "f32[384]", arg106_1: "f32[384]", arg107_1: "f32[384]", arg108_1: "f32[1536, 384]", arg109_1: "f32[1536]", arg110_1: "f32[384, 1536]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[384]", arg114_1: "f32[1152, 384]", arg115_1: "f32[1152]", arg116_1: "f32[384, 384]", arg117_1: "f32[384]", arg118_1: "f32[384]", arg119_1: "f32[384]", arg120_1: "f32[1536, 384]", arg121_1: "f32[1536]", arg122_1: "f32[384, 1536]", arg123_1: "f32[384]", arg124_1: "f32[384]", arg125_1: "f32[384]", arg126_1: "f32[1152, 384]", arg127_1: "f32[1152]", arg128_1: "f32[384, 384]", arg129_1: "f32[384]", arg130_1: "f32[384]", arg131_1: "f32[384]", arg132_1: "f32[1536, 384]", arg133_1: "f32[1536]", arg134_1: "f32[384, 1536]", arg135_1: "f32[384]", arg136_1: "f32[384]", arg137_1: "f32[384]", arg138_1: "f32[1152, 384]", arg139_1: "f32[1152]", arg140_1: "f32[384, 384]", arg141_1: "f32[384]", arg142_1: "f32[384]", arg143_1: "f32[384]", arg144_1: "f32[1536, 384]", arg145_1: "f32[1536]", arg146_1: "f32[384, 1536]", arg147_1: "f32[384]", arg148_1: "f32[384]", arg149_1: "f32[384]", arg150_1: "f32[1000, 384]", arg151_1: "f32[1000]", arg152_1: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:579, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    expand: "f32[4, 1, 384]" = torch.ops.aten.expand.default(arg1_1, [4, -1, -1]);  arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[4, 384, 14, 14]" = torch.ops.aten.convolution.default(arg152_1, arg2_1, arg3_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg152_1 = arg2_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[4, 384, 196]" = torch.ops.aten.reshape.default(convolution, [4, 384, 196]);  convolution = None
    permute: "f32[4, 196, 384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:579, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    cat: "f32[4, 197, 384]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:580, code: x = x + pos_embed
    add: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(cat, arg0_1);  cat = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[4, 197, 1]" = var_mean[0]
    getitem_1: "f32[4, 197, 1]" = var_mean[1];  var_mean = None
    sub: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    add_1: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
    add_2: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_1: "f32[788, 384]" = torch.ops.aten.reshape.default(add_2, [788, 384]);  add_2 = None
    permute_1: "f32[384, 1152]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg7_1, view_1, permute_1);  arg7_1 = view_1 = permute_1 = None
    view_2: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm, [4, 197, 1152]);  addmm = None
    view_3: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_2, [4, 197, 3, 6, 64]);  view_2 = None
    permute_2: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_2: "f32[4, 6, 197, 64]" = unbind[0]
    getitem_3: "f32[4, 6, 197, 64]" = unbind[1]
    getitem_4: "f32[4, 6, 197, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_2, getitem_3, getitem_4, None, False);  getitem_2 = getitem_3 = getitem_4 = None
    getitem_5: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_3: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_4: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_3, [4, 197, 384]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_5: "f32[788, 384]" = torch.ops.aten.reshape.default(view_4, [788, 384]);  view_4 = None
    permute_4: "f32[384, 384]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[788, 384]" = torch.ops.aten.mm.default(view_5, permute_4);  view_5 = permute_4 = None
    add_tensor_35: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_35, arg9_1);  mm_default_35 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_6: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_35, [4, 197, 384]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_3: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add, view_6);  add = view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_9: "f32[4, 197, 1]" = var_mean_1[0]
    getitem_10: "f32[4, 197, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_1: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_3, getitem_10);  getitem_10 = None
    add_4: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
    rsqrt_1: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    mul_2: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_3: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = arg10_1 = None
    add_5: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_7: "f32[788, 384]" = torch.ops.aten.reshape.default(add_5, [788, 384]);  add_5 = None
    permute_5: "f32[384, 1536]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[788, 1536]" = torch.ops.aten.mm.default(view_7, permute_5);  view_7 = permute_5 = None
    add_tensor_34: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_34, arg13_1);  mm_default_34 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_8: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_34, [4, 197, 1536]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_4: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.5)
    mul_5: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476);  view_8 = None
    erf: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_9: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_6, [788, 1536]);  mul_6 = None
    permute_6: "f32[1536, 384]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[788, 384]" = torch.ops.aten.mm.default(view_9, permute_6);  view_9 = permute_6 = None
    add_tensor_33: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_33, arg15_1);  mm_default_33 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_10: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_33, [4, 197, 384]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_7: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_3, view_10);  add_3 = view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_11: "f32[4, 197, 1]" = var_mean_2[0]
    getitem_12: "f32[4, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_2: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_7, getitem_12);  getitem_12 = None
    add_8: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
    rsqrt_2: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    mul_7: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_8: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_7, arg16_1);  mul_7 = arg16_1 = None
    add_9: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_8, arg17_1);  mul_8 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_11: "f32[788, 384]" = torch.ops.aten.reshape.default(add_9, [788, 384]);  add_9 = None
    permute_7: "f32[384, 1152]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    addmm_4: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg19_1, view_11, permute_7);  arg19_1 = view_11 = permute_7 = None
    view_12: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_4, [4, 197, 1152]);  addmm_4 = None
    view_13: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_12, [4, 197, 3, 6, 64]);  view_12 = None
    permute_8: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_13, [2, 0, 3, 1, 4]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_8);  permute_8 = None
    getitem_13: "f32[4, 6, 197, 64]" = unbind_1[0]
    getitem_14: "f32[4, 6, 197, 64]" = unbind_1[1]
    getitem_15: "f32[4, 6, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_13, getitem_14, getitem_15, None, False);  getitem_13 = getitem_14 = getitem_15 = None
    getitem_16: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_9: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    view_14: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_9, [4, 197, 384]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_15: "f32[788, 384]" = torch.ops.aten.reshape.default(view_14, [788, 384]);  view_14 = None
    permute_10: "f32[384, 384]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[788, 384]" = torch.ops.aten.mm.default(view_15, permute_10);  view_15 = permute_10 = None
    add_tensor_32: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_32, arg21_1);  mm_default_32 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_16: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_32, [4, 197, 384]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_10: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_7, view_16);  add_7 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_20: "f32[4, 197, 1]" = var_mean_3[0]
    getitem_21: "f32[4, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_3: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_10, getitem_21);  getitem_21 = None
    add_11: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_3: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_9: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    mul_10: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
    add_12: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_10, arg23_1);  mul_10 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[788, 384]" = torch.ops.aten.reshape.default(add_12, [788, 384]);  add_12 = None
    permute_11: "f32[384, 1536]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[788, 1536]" = torch.ops.aten.mm.default(view_17, permute_11);  view_17 = permute_11 = None
    add_tensor_31: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_31, arg25_1);  mm_default_31 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_31, [4, 197, 1536]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_11: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_12: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf_1: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_13, [788, 1536]);  mul_13 = None
    permute_12: "f32[1536, 384]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[788, 384]" = torch.ops.aten.mm.default(view_19, permute_12);  view_19 = permute_12 = None
    add_tensor_30: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_30, arg27_1);  mm_default_30 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_20: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_30, [4, 197, 384]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_14: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_10, view_20);  add_10 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_22: "f32[4, 197, 1]" = var_mean_4[0]
    getitem_23: "f32[4, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_4: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_14, getitem_23);  getitem_23 = None
    add_15: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_4: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    mul_14: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    mul_15: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_14, arg28_1);  mul_14 = arg28_1 = None
    add_16: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_15, arg29_1);  mul_15 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_21: "f32[788, 384]" = torch.ops.aten.reshape.default(add_16, [788, 384]);  add_16 = None
    permute_13: "f32[384, 1152]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_8: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg31_1, view_21, permute_13);  arg31_1 = view_21 = permute_13 = None
    view_22: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_8, [4, 197, 1152]);  addmm_8 = None
    view_23: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_22, [4, 197, 3, 6, 64]);  view_22 = None
    permute_14: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
    getitem_24: "f32[4, 6, 197, 64]" = unbind_2[0]
    getitem_25: "f32[4, 6, 197, 64]" = unbind_2[1]
    getitem_26: "f32[4, 6, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_24, getitem_25, getitem_26, None, False);  getitem_24 = getitem_25 = getitem_26 = None
    getitem_27: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_15: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
    view_24: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_15, [4, 197, 384]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_25: "f32[788, 384]" = torch.ops.aten.reshape.default(view_24, [788, 384]);  view_24 = None
    permute_16: "f32[384, 384]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[788, 384]" = torch.ops.aten.mm.default(view_25, permute_16);  view_25 = permute_16 = None
    add_tensor_29: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_29, arg33_1);  mm_default_29 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_26: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_29, [4, 197, 384]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_17: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_14, view_26);  add_14 = view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_31: "f32[4, 197, 1]" = var_mean_5[0]
    getitem_32: "f32[4, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_5: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_17, getitem_32);  getitem_32 = None
    add_18: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-06);  getitem_31 = None
    rsqrt_5: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_16: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    mul_17: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_16, arg34_1);  mul_16 = arg34_1 = None
    add_19: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_27: "f32[788, 384]" = torch.ops.aten.reshape.default(add_19, [788, 384]);  add_19 = None
    permute_17: "f32[384, 1536]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[788, 1536]" = torch.ops.aten.mm.default(view_27, permute_17);  view_27 = permute_17 = None
    add_tensor_28: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_28, arg37_1);  mm_default_28 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_28: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_28, [4, 197, 1536]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_28, 0.5)
    mul_19: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476);  view_28 = None
    erf_2: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_29: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_20, [788, 1536]);  mul_20 = None
    permute_18: "f32[1536, 384]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[788, 384]" = torch.ops.aten.mm.default(view_29, permute_18);  view_29 = permute_18 = None
    add_tensor_27: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_27, arg39_1);  mm_default_27 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_30: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_27, [4, 197, 384]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_21: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_17, view_30);  add_17 = view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_33: "f32[4, 197, 1]" = var_mean_6[0]
    getitem_34: "f32[4, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_6: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_21, getitem_34);  getitem_34 = None
    add_22: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_6: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    mul_21: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    mul_22: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_21, arg40_1);  mul_21 = arg40_1 = None
    add_23: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_22, arg41_1);  mul_22 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_31: "f32[788, 384]" = torch.ops.aten.reshape.default(add_23, [788, 384]);  add_23 = None
    permute_19: "f32[384, 1152]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_12: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg43_1, view_31, permute_19);  arg43_1 = view_31 = permute_19 = None
    view_32: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_12, [4, 197, 1152]);  addmm_12 = None
    view_33: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_32, [4, 197, 3, 6, 64]);  view_32 = None
    permute_20: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_20);  permute_20 = None
    getitem_35: "f32[4, 6, 197, 64]" = unbind_3[0]
    getitem_36: "f32[4, 6, 197, 64]" = unbind_3[1]
    getitem_37: "f32[4, 6, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_35, getitem_36, getitem_37, None, False);  getitem_35 = getitem_36 = getitem_37 = None
    getitem_38: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_21: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
    view_34: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_21, [4, 197, 384]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_35: "f32[788, 384]" = torch.ops.aten.reshape.default(view_34, [788, 384]);  view_34 = None
    permute_22: "f32[384, 384]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[788, 384]" = torch.ops.aten.mm.default(view_35, permute_22);  view_35 = permute_22 = None
    add_tensor_26: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_26, arg45_1);  mm_default_26 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_36: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_26, [4, 197, 384]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_24: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_21, view_36);  add_21 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_42: "f32[4, 197, 1]" = var_mean_7[0]
    getitem_43: "f32[4, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_7: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_24, getitem_43);  getitem_43 = None
    add_25: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_7: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_23: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    mul_24: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_23, arg46_1);  mul_23 = arg46_1 = None
    add_26: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_24, arg47_1);  mul_24 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[788, 384]" = torch.ops.aten.reshape.default(add_26, [788, 384]);  add_26 = None
    permute_23: "f32[384, 1536]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[788, 1536]" = torch.ops.aten.mm.default(view_37, permute_23);  view_37 = permute_23 = None
    add_tensor_25: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_25, arg49_1);  mm_default_25 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_38: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_25, [4, 197, 1536]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_25: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_26: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_3: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_27: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_27: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_27, [788, 1536]);  mul_27 = None
    permute_24: "f32[1536, 384]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[788, 384]" = torch.ops.aten.mm.default(view_39, permute_24);  view_39 = permute_24 = None
    add_tensor_24: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_24, arg51_1);  mm_default_24 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_40: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_24, [4, 197, 384]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_28: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_24, view_40);  add_24 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_44: "f32[4, 197, 1]" = var_mean_8[0]
    getitem_45: "f32[4, 197, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_8: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_28, getitem_45);  getitem_45 = None
    add_29: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_8: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    mul_28: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    mul_29: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_28, arg52_1);  mul_28 = arg52_1 = None
    add_30: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_29, arg53_1);  mul_29 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_41: "f32[788, 384]" = torch.ops.aten.reshape.default(add_30, [788, 384]);  add_30 = None
    permute_25: "f32[384, 1152]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_16: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg55_1, view_41, permute_25);  arg55_1 = view_41 = permute_25 = None
    view_42: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_16, [4, 197, 1152]);  addmm_16 = None
    view_43: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_42, [4, 197, 3, 6, 64]);  view_42 = None
    permute_26: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_43, [2, 0, 3, 1, 4]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
    getitem_46: "f32[4, 6, 197, 64]" = unbind_4[0]
    getitem_47: "f32[4, 6, 197, 64]" = unbind_4[1]
    getitem_48: "f32[4, 6, 197, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_46, getitem_47, getitem_48, None, False);  getitem_46 = getitem_47 = getitem_48 = None
    getitem_49: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_27: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
    view_44: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_27, [4, 197, 384]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_45: "f32[788, 384]" = torch.ops.aten.reshape.default(view_44, [788, 384]);  view_44 = None
    permute_28: "f32[384, 384]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[788, 384]" = torch.ops.aten.mm.default(view_45, permute_28);  view_45 = permute_28 = None
    add_tensor_23: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_23, arg57_1);  mm_default_23 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_46: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_23, [4, 197, 384]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_31: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_28, view_46);  add_28 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_53: "f32[4, 197, 1]" = var_mean_9[0]
    getitem_54: "f32[4, 197, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_9: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_31, getitem_54);  getitem_54 = None
    add_32: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-06);  getitem_53 = None
    rsqrt_9: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_30: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    mul_31: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_30, arg58_1);  mul_30 = arg58_1 = None
    add_33: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_31, arg59_1);  mul_31 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_47: "f32[788, 384]" = torch.ops.aten.reshape.default(add_33, [788, 384]);  add_33 = None
    permute_29: "f32[384, 1536]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[788, 1536]" = torch.ops.aten.mm.default(view_47, permute_29);  view_47 = permute_29 = None
    add_tensor_22: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_22, arg61_1);  mm_default_22 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_48: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_22, [4, 197, 1536]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_48, 0.5)
    mul_33: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476);  view_48 = None
    erf_4: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_34: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_34: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_49: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_34, [788, 1536]);  mul_34 = None
    permute_30: "f32[1536, 384]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[788, 384]" = torch.ops.aten.mm.default(view_49, permute_30);  view_49 = permute_30 = None
    add_tensor_21: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_21, arg63_1);  mm_default_21 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_50: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_21, [4, 197, 384]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_35: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_31, view_50);  add_31 = view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_55: "f32[4, 197, 1]" = var_mean_10[0]
    getitem_56: "f32[4, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_10: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_35, getitem_56);  getitem_56 = None
    add_36: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_10: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    mul_35: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    mul_36: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_35, arg64_1);  mul_35 = arg64_1 = None
    add_37: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_36, arg65_1);  mul_36 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_51: "f32[788, 384]" = torch.ops.aten.reshape.default(add_37, [788, 384]);  add_37 = None
    permute_31: "f32[384, 1152]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    addmm_20: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg67_1, view_51, permute_31);  arg67_1 = view_51 = permute_31 = None
    view_52: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_20, [4, 197, 1152]);  addmm_20 = None
    view_53: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_52, [4, 197, 3, 6, 64]);  view_52 = None
    permute_32: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_53, [2, 0, 3, 1, 4]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_32);  permute_32 = None
    getitem_57: "f32[4, 6, 197, 64]" = unbind_5[0]
    getitem_58: "f32[4, 6, 197, 64]" = unbind_5[1]
    getitem_59: "f32[4, 6, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_57, getitem_58, getitem_59, None, False);  getitem_57 = getitem_58 = getitem_59 = None
    getitem_60: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_33: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
    view_54: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_33, [4, 197, 384]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_55: "f32[788, 384]" = torch.ops.aten.reshape.default(view_54, [788, 384]);  view_54 = None
    permute_34: "f32[384, 384]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[788, 384]" = torch.ops.aten.mm.default(view_55, permute_34);  view_55 = permute_34 = None
    add_tensor_20: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_20, arg69_1);  mm_default_20 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_56: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_20, [4, 197, 384]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_38: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_35, view_56);  add_35 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_64: "f32[4, 197, 1]" = var_mean_11[0]
    getitem_65: "f32[4, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_11: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_38, getitem_65);  getitem_65 = None
    add_39: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_11: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    mul_37: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    mul_38: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_37, arg70_1);  mul_37 = arg70_1 = None
    add_40: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_38, arg71_1);  mul_38 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_57: "f32[788, 384]" = torch.ops.aten.reshape.default(add_40, [788, 384]);  add_40 = None
    permute_35: "f32[384, 1536]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[788, 1536]" = torch.ops.aten.mm.default(view_57, permute_35);  view_57 = permute_35 = None
    add_tensor_19: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_19, arg73_1);  mm_default_19 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_58: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_19, [4, 197, 1536]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_39: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_40: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_5: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_41: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_41: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_41, [788, 1536]);  mul_41 = None
    permute_36: "f32[1536, 384]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[788, 384]" = torch.ops.aten.mm.default(view_59, permute_36);  view_59 = permute_36 = None
    add_tensor_18: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_18, arg75_1);  mm_default_18 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_60: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_18, [4, 197, 384]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_42: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_38, view_60);  add_38 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_66: "f32[4, 197, 1]" = var_mean_12[0]
    getitem_67: "f32[4, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_12: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_42, getitem_67);  getitem_67 = None
    add_43: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_12: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    mul_42: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    mul_43: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_42, arg76_1);  mul_42 = arg76_1 = None
    add_44: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_43, arg77_1);  mul_43 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_61: "f32[788, 384]" = torch.ops.aten.reshape.default(add_44, [788, 384]);  add_44 = None
    permute_37: "f32[384, 1152]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_24: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg79_1, view_61, permute_37);  arg79_1 = view_61 = permute_37 = None
    view_62: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_24, [4, 197, 1152]);  addmm_24 = None
    view_63: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_62, [4, 197, 3, 6, 64]);  view_62 = None
    permute_38: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_38);  permute_38 = None
    getitem_68: "f32[4, 6, 197, 64]" = unbind_6[0]
    getitem_69: "f32[4, 6, 197, 64]" = unbind_6[1]
    getitem_70: "f32[4, 6, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_68, getitem_69, getitem_70, None, False);  getitem_68 = getitem_69 = getitem_70 = None
    getitem_71: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_39: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
    view_64: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_39, [4, 197, 384]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_65: "f32[788, 384]" = torch.ops.aten.reshape.default(view_64, [788, 384]);  view_64 = None
    permute_40: "f32[384, 384]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[788, 384]" = torch.ops.aten.mm.default(view_65, permute_40);  view_65 = permute_40 = None
    add_tensor_17: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_17, arg81_1);  mm_default_17 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_66: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_17, [4, 197, 384]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_45: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_42, view_66);  add_42 = view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_75: "f32[4, 197, 1]" = var_mean_13[0]
    getitem_76: "f32[4, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_13: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_45, getitem_76);  getitem_76 = None
    add_46: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
    rsqrt_13: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    mul_44: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    mul_45: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_44, arg82_1);  mul_44 = arg82_1 = None
    add_47: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_45, arg83_1);  mul_45 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_67: "f32[788, 384]" = torch.ops.aten.reshape.default(add_47, [788, 384]);  add_47 = None
    permute_41: "f32[384, 1536]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[788, 1536]" = torch.ops.aten.mm.default(view_67, permute_41);  view_67 = permute_41 = None
    add_tensor_16: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_16, arg85_1);  mm_default_16 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_68: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_16, [4, 197, 1536]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.5)
    mul_47: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476);  view_68 = None
    erf_6: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_48: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_48: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_69: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_48, [788, 1536]);  mul_48 = None
    permute_42: "f32[1536, 384]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[788, 384]" = torch.ops.aten.mm.default(view_69, permute_42);  view_69 = permute_42 = None
    add_tensor_15: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_15, arg87_1);  mm_default_15 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_70: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_15, [4, 197, 384]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_49: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_45, view_70);  add_45 = view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_77: "f32[4, 197, 1]" = var_mean_14[0]
    getitem_78: "f32[4, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_14: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_49, getitem_78);  getitem_78 = None
    add_50: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_14: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    mul_49: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    mul_50: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_49, arg88_1);  mul_49 = arg88_1 = None
    add_51: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_50, arg89_1);  mul_50 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_71: "f32[788, 384]" = torch.ops.aten.reshape.default(add_51, [788, 384]);  add_51 = None
    permute_43: "f32[384, 1152]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_28: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg91_1, view_71, permute_43);  arg91_1 = view_71 = permute_43 = None
    view_72: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_28, [4, 197, 1152]);  addmm_28 = None
    view_73: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_72, [4, 197, 3, 6, 64]);  view_72 = None
    permute_44: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_73, [2, 0, 3, 1, 4]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
    getitem_79: "f32[4, 6, 197, 64]" = unbind_7[0]
    getitem_80: "f32[4, 6, 197, 64]" = unbind_7[1]
    getitem_81: "f32[4, 6, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_79, getitem_80, getitem_81, None, False);  getitem_79 = getitem_80 = getitem_81 = None
    getitem_82: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_45: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
    view_74: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_45, [4, 197, 384]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_75: "f32[788, 384]" = torch.ops.aten.reshape.default(view_74, [788, 384]);  view_74 = None
    permute_46: "f32[384, 384]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[788, 384]" = torch.ops.aten.mm.default(view_75, permute_46);  view_75 = permute_46 = None
    add_tensor_14: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_14, arg93_1);  mm_default_14 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_76: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_14, [4, 197, 384]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_52: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_49, view_76);  add_49 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_86: "f32[4, 197, 1]" = var_mean_15[0]
    getitem_87: "f32[4, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_15: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_52, getitem_87);  getitem_87 = None
    add_53: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_15: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_51: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    mul_52: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_51, arg94_1);  mul_51 = arg94_1 = None
    add_54: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_52, arg95_1);  mul_52 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_77: "f32[788, 384]" = torch.ops.aten.reshape.default(add_54, [788, 384]);  add_54 = None
    permute_47: "f32[384, 1536]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[788, 1536]" = torch.ops.aten.mm.default(view_77, permute_47);  view_77 = permute_47 = None
    add_tensor_13: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_13, arg97_1);  mm_default_13 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_78: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_13, [4, 197, 1536]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_54: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_7: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_55: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_55: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_53, add_55);  mul_53 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_79: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_55, [788, 1536]);  mul_55 = None
    permute_48: "f32[1536, 384]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[788, 384]" = torch.ops.aten.mm.default(view_79, permute_48);  view_79 = permute_48 = None
    add_tensor_12: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_12, arg99_1);  mm_default_12 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_80: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_12, [4, 197, 384]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_56: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_52, view_80);  add_52 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_88: "f32[4, 197, 1]" = var_mean_16[0]
    getitem_89: "f32[4, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_16: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_56, getitem_89);  getitem_89 = None
    add_57: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_16: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    mul_56: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    mul_57: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_56, arg100_1);  mul_56 = arg100_1 = None
    add_58: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_57, arg101_1);  mul_57 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_81: "f32[788, 384]" = torch.ops.aten.reshape.default(add_58, [788, 384]);  add_58 = None
    permute_49: "f32[384, 1152]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_32: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg103_1, view_81, permute_49);  arg103_1 = view_81 = permute_49 = None
    view_82: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_32, [4, 197, 1152]);  addmm_32 = None
    view_83: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_82, [4, 197, 3, 6, 64]);  view_82 = None
    permute_50: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_83, [2, 0, 3, 1, 4]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_90: "f32[4, 6, 197, 64]" = unbind_8[0]
    getitem_91: "f32[4, 6, 197, 64]" = unbind_8[1]
    getitem_92: "f32[4, 6, 197, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_90, getitem_91, getitem_92, None, False);  getitem_90 = getitem_91 = getitem_92 = None
    getitem_93: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_51: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
    view_84: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_51, [4, 197, 384]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_85: "f32[788, 384]" = torch.ops.aten.reshape.default(view_84, [788, 384]);  view_84 = None
    permute_52: "f32[384, 384]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[788, 384]" = torch.ops.aten.mm.default(view_85, permute_52);  view_85 = permute_52 = None
    add_tensor_11: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_11, arg105_1);  mm_default_11 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_86: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_11, [4, 197, 384]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_59: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_56, view_86);  add_56 = view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_97: "f32[4, 197, 1]" = var_mean_17[0]
    getitem_98: "f32[4, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_17: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_59, getitem_98);  getitem_98 = None
    add_60: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_17: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_58: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    mul_59: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_58, arg106_1);  mul_58 = arg106_1 = None
    add_61: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_59, arg107_1);  mul_59 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_87: "f32[788, 384]" = torch.ops.aten.reshape.default(add_61, [788, 384]);  add_61 = None
    permute_53: "f32[384, 1536]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[788, 1536]" = torch.ops.aten.mm.default(view_87, permute_53);  view_87 = permute_53 = None
    add_tensor_10: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_10, arg109_1);  mm_default_10 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_88: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_10, [4, 197, 1536]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_60: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
    mul_61: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476);  view_88 = None
    erf_8: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_62: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_62: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_60, add_62);  mul_60 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_89: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_62, [788, 1536]);  mul_62 = None
    permute_54: "f32[1536, 384]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[788, 384]" = torch.ops.aten.mm.default(view_89, permute_54);  view_89 = permute_54 = None
    add_tensor_9: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_9, arg111_1);  mm_default_9 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_90: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_9, [4, 197, 384]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_63: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_59, view_90);  add_59 = view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_99: "f32[4, 197, 1]" = var_mean_18[0]
    getitem_100: "f32[4, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_18: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_63, getitem_100);  getitem_100 = None
    add_64: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_18: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    mul_63: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    mul_64: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_63, arg112_1);  mul_63 = arg112_1 = None
    add_65: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_64, arg113_1);  mul_64 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_91: "f32[788, 384]" = torch.ops.aten.reshape.default(add_65, [788, 384]);  add_65 = None
    permute_55: "f32[384, 1152]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_36: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg115_1, view_91, permute_55);  arg115_1 = view_91 = permute_55 = None
    view_92: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_36, [4, 197, 1152]);  addmm_36 = None
    view_93: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_92, [4, 197, 3, 6, 64]);  view_92 = None
    permute_56: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_56);  permute_56 = None
    getitem_101: "f32[4, 6, 197, 64]" = unbind_9[0]
    getitem_102: "f32[4, 6, 197, 64]" = unbind_9[1]
    getitem_103: "f32[4, 6, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_101, getitem_102, getitem_103, None, False);  getitem_101 = getitem_102 = getitem_103 = None
    getitem_104: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_57: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
    view_94: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_57, [4, 197, 384]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_95: "f32[788, 384]" = torch.ops.aten.reshape.default(view_94, [788, 384]);  view_94 = None
    permute_58: "f32[384, 384]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[788, 384]" = torch.ops.aten.mm.default(view_95, permute_58);  view_95 = permute_58 = None
    add_tensor_8: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_8, arg117_1);  mm_default_8 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_96: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_8, [4, 197, 384]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_66: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_63, view_96);  add_63 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_108: "f32[4, 197, 1]" = var_mean_19[0]
    getitem_109: "f32[4, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_19: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_66, getitem_109);  getitem_109 = None
    add_67: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_19: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_65: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    mul_66: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_65, arg118_1);  mul_65 = arg118_1 = None
    add_68: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_66, arg119_1);  mul_66 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_97: "f32[788, 384]" = torch.ops.aten.reshape.default(add_68, [788, 384]);  add_68 = None
    permute_59: "f32[384, 1536]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[788, 1536]" = torch.ops.aten.mm.default(view_97, permute_59);  view_97 = permute_59 = None
    add_tensor_7: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_7, arg121_1);  mm_default_7 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_98: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_7, [4, 197, 1536]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_68: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_9: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_69: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_99: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_69, [788, 1536]);  mul_69 = None
    permute_60: "f32[1536, 384]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[788, 384]" = torch.ops.aten.mm.default(view_99, permute_60);  view_99 = permute_60 = None
    add_tensor_6: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_6, arg123_1);  mm_default_6 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_100: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_6, [4, 197, 384]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_70: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_66, view_100);  add_66 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_110: "f32[4, 197, 1]" = var_mean_20[0]
    getitem_111: "f32[4, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_20: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_70, getitem_111);  getitem_111 = None
    add_71: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_20: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    mul_70: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    mul_71: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_70, arg124_1);  mul_70 = arg124_1 = None
    add_72: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_71, arg125_1);  mul_71 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_101: "f32[788, 384]" = torch.ops.aten.reshape.default(add_72, [788, 384]);  add_72 = None
    permute_61: "f32[384, 1152]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_40: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg127_1, view_101, permute_61);  arg127_1 = view_101 = permute_61 = None
    view_102: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_40, [4, 197, 1152]);  addmm_40 = None
    view_103: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_102, [4, 197, 3, 6, 64]);  view_102 = None
    permute_62: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
    getitem_112: "f32[4, 6, 197, 64]" = unbind_10[0]
    getitem_113: "f32[4, 6, 197, 64]" = unbind_10[1]
    getitem_114: "f32[4, 6, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_112, getitem_113, getitem_114, None, False);  getitem_112 = getitem_113 = getitem_114 = None
    getitem_115: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_63: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
    view_104: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_63, [4, 197, 384]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_105: "f32[788, 384]" = torch.ops.aten.reshape.default(view_104, [788, 384]);  view_104 = None
    permute_64: "f32[384, 384]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[788, 384]" = torch.ops.aten.mm.default(view_105, permute_64);  view_105 = permute_64 = None
    add_tensor_5: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_5, arg129_1);  mm_default_5 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_106: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_5, [4, 197, 384]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_73: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_70, view_106);  add_70 = view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_119: "f32[4, 197, 1]" = var_mean_21[0]
    getitem_120: "f32[4, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_21: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_73, getitem_120);  getitem_120 = None
    add_74: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-06);  getitem_119 = None
    rsqrt_21: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    mul_72: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    mul_73: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_72, arg130_1);  mul_72 = arg130_1 = None
    add_75: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_73, arg131_1);  mul_73 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_107: "f32[788, 384]" = torch.ops.aten.reshape.default(add_75, [788, 384]);  add_75 = None
    permute_65: "f32[384, 1536]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[788, 1536]" = torch.ops.aten.mm.default(view_107, permute_65);  view_107 = permute_65 = None
    add_tensor_4: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_4, arg133_1);  mm_default_4 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_108: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_4, [4, 197, 1536]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_108, 0.5)
    mul_75: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476);  view_108 = None
    erf_10: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_76: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_76: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_74, add_76);  mul_74 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_109: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_76, [788, 1536]);  mul_76 = None
    permute_66: "f32[1536, 384]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[788, 384]" = torch.ops.aten.mm.default(view_109, permute_66);  view_109 = permute_66 = None
    add_tensor_3: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_3, arg135_1);  mm_default_3 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_110: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_3, [4, 197, 384]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_77: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_73, view_110);  add_73 = view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_121: "f32[4, 197, 1]" = var_mean_22[0]
    getitem_122: "f32[4, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_22: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_77, getitem_122);  getitem_122 = None
    add_78: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_22: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    mul_77: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    mul_78: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_77, arg136_1);  mul_77 = arg136_1 = None
    add_79: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_78, arg137_1);  mul_78 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_111: "f32[788, 384]" = torch.ops.aten.reshape.default(add_79, [788, 384]);  add_79 = None
    permute_67: "f32[384, 1152]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_44: "f32[788, 1152]" = torch.ops.aten.addmm.default(arg139_1, view_111, permute_67);  arg139_1 = view_111 = permute_67 = None
    view_112: "f32[4, 197, 1152]" = torch.ops.aten.reshape.default(addmm_44, [4, 197, 1152]);  addmm_44 = None
    view_113: "f32[4, 197, 3, 6, 64]" = torch.ops.aten.reshape.default(view_112, [4, 197, 3, 6, 64]);  view_112 = None
    permute_68: "f32[3, 4, 6, 197, 64]" = torch.ops.aten.permute.default(view_113, [2, 0, 3, 1, 4]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
    getitem_123: "f32[4, 6, 197, 64]" = unbind_11[0]
    getitem_124: "f32[4, 6, 197, 64]" = unbind_11[1]
    getitem_125: "f32[4, 6, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_123, getitem_124, getitem_125, None, False);  getitem_123 = getitem_124 = getitem_125 = None
    getitem_126: "f32[4, 6, 197, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_69: "f32[4, 197, 6, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
    view_114: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(permute_69, [4, 197, 384]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_115: "f32[788, 384]" = torch.ops.aten.reshape.default(view_114, [788, 384]);  view_114 = None
    permute_70: "f32[384, 384]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[788, 384]" = torch.ops.aten.mm.default(view_115, permute_70);  view_115 = permute_70 = None
    add_tensor_2: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default_2, arg141_1);  mm_default_2 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_116: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_2, [4, 197, 384]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_80: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_77, view_116);  add_77 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_130: "f32[4, 197, 1]" = var_mean_23[0]
    getitem_131: "f32[4, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_23: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_80, getitem_131);  getitem_131 = None
    add_81: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
    rsqrt_23: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_79: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    mul_80: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_79, arg142_1);  mul_79 = arg142_1 = None
    add_82: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_80, arg143_1);  mul_80 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[788, 384]" = torch.ops.aten.reshape.default(add_82, [788, 384]);  add_82 = None
    permute_71: "f32[384, 1536]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[788, 1536]" = torch.ops.aten.mm.default(view_117, permute_71);  view_117 = permute_71 = None
    add_tensor_1: "f32[788, 1536]" = torch.ops.aten.add.Tensor(mm_default_1, arg145_1);  mm_default_1 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_118: "f32[4, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_1, [4, 197, 1536]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_81: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_82: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_11: "f32[4, 197, 1536]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_83: "f32[4, 197, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_83: "f32[4, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_81, add_83);  mul_81 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[788, 1536]" = torch.ops.aten.reshape.default(mul_83, [788, 1536]);  mul_83 = None
    permute_72: "f32[1536, 384]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[788, 384]" = torch.ops.aten.mm.default(view_119, permute_72);  view_119 = permute_72 = None
    add_tensor: "f32[788, 384]" = torch.ops.aten.add.Tensor(mm_default, arg147_1);  mm_default = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_120: "f32[4, 197, 384]" = torch.ops.aten.reshape.default(add_tensor, [4, 197, 384]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_84: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(add_80, view_120);  add_80 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_132: "f32[4, 197, 1]" = var_mean_24[0]
    getitem_133: "f32[4, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_24: "f32[4, 197, 384]" = torch.ops.aten.sub.Tensor(add_84, getitem_133);  add_84 = getitem_133 = None
    add_85: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_24: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    mul_84: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    mul_85: "f32[4, 197, 384]" = torch.ops.aten.mul.Tensor(mul_84, arg148_1);  mul_84 = arg148_1 = None
    add_86: "f32[4, 197, 384]" = torch.ops.aten.add.Tensor(mul_85, arg149_1);  mul_85 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:646, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    select: "f32[4, 384]" = torch.ops.aten.select.int(add_86, 1, 0);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:648, code: x = self.head_drop(x)
    clone_37: "f32[4, 384]" = torch.ops.aten.clone.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:649, code: return x if pre_logits else self.head(x)
    permute_73: "f32[384, 1000]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_48: "f32[4, 1000]" = torch.ops.aten.addmm.default(arg151_1, clone_37, permute_73);  arg151_1 = clone_37 = permute_73 = None
    return (addmm_48,)
    