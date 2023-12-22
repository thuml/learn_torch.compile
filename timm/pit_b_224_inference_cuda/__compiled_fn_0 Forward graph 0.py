from __future__ import annotations



def forward(self, arg0_1: "f32[1, 256, 31, 31]", arg1_1: "f32[1, 1, 256]", arg2_1: "f32[256, 3, 14, 14]", arg3_1: "f32[256]", arg4_1: "f32[256]", arg5_1: "f32[256]", arg6_1: "f32[768, 256]", arg7_1: "f32[768]", arg8_1: "f32[256, 256]", arg9_1: "f32[256]", arg10_1: "f32[256]", arg11_1: "f32[256]", arg12_1: "f32[1024, 256]", arg13_1: "f32[1024]", arg14_1: "f32[256, 1024]", arg15_1: "f32[256]", arg16_1: "f32[256]", arg17_1: "f32[256]", arg18_1: "f32[768, 256]", arg19_1: "f32[768]", arg20_1: "f32[256, 256]", arg21_1: "f32[256]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[1024, 256]", arg25_1: "f32[1024]", arg26_1: "f32[256, 1024]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[768, 256]", arg31_1: "f32[768]", arg32_1: "f32[256, 256]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[1024, 256]", arg37_1: "f32[1024]", arg38_1: "f32[256, 1024]", arg39_1: "f32[256]", arg40_1: "f32[512, 1, 3, 3]", arg41_1: "f32[512]", arg42_1: "f32[512, 256]", arg43_1: "f32[512]", arg44_1: "f32[512]", arg45_1: "f32[512]", arg46_1: "f32[1536, 512]", arg47_1: "f32[1536]", arg48_1: "f32[512, 512]", arg49_1: "f32[512]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[2048, 512]", arg53_1: "f32[2048]", arg54_1: "f32[512, 2048]", arg55_1: "f32[512]", arg56_1: "f32[512]", arg57_1: "f32[512]", arg58_1: "f32[1536, 512]", arg59_1: "f32[1536]", arg60_1: "f32[512, 512]", arg61_1: "f32[512]", arg62_1: "f32[512]", arg63_1: "f32[512]", arg64_1: "f32[2048, 512]", arg65_1: "f32[2048]", arg66_1: "f32[512, 2048]", arg67_1: "f32[512]", arg68_1: "f32[512]", arg69_1: "f32[512]", arg70_1: "f32[1536, 512]", arg71_1: "f32[1536]", arg72_1: "f32[512, 512]", arg73_1: "f32[512]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[2048, 512]", arg77_1: "f32[2048]", arg78_1: "f32[512, 2048]", arg79_1: "f32[512]", arg80_1: "f32[512]", arg81_1: "f32[512]", arg82_1: "f32[1536, 512]", arg83_1: "f32[1536]", arg84_1: "f32[512, 512]", arg85_1: "f32[512]", arg86_1: "f32[512]", arg87_1: "f32[512]", arg88_1: "f32[2048, 512]", arg89_1: "f32[2048]", arg90_1: "f32[512, 2048]", arg91_1: "f32[512]", arg92_1: "f32[512]", arg93_1: "f32[512]", arg94_1: "f32[1536, 512]", arg95_1: "f32[1536]", arg96_1: "f32[512, 512]", arg97_1: "f32[512]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[2048, 512]", arg101_1: "f32[2048]", arg102_1: "f32[512, 2048]", arg103_1: "f32[512]", arg104_1: "f32[512]", arg105_1: "f32[512]", arg106_1: "f32[1536, 512]", arg107_1: "f32[1536]", arg108_1: "f32[512, 512]", arg109_1: "f32[512]", arg110_1: "f32[512]", arg111_1: "f32[512]", arg112_1: "f32[2048, 512]", arg113_1: "f32[2048]", arg114_1: "f32[512, 2048]", arg115_1: "f32[512]", arg116_1: "f32[1024, 1, 3, 3]", arg117_1: "f32[1024]", arg118_1: "f32[1024, 512]", arg119_1: "f32[1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024]", arg122_1: "f32[3072, 1024]", arg123_1: "f32[3072]", arg124_1: "f32[1024, 1024]", arg125_1: "f32[1024]", arg126_1: "f32[1024]", arg127_1: "f32[1024]", arg128_1: "f32[4096, 1024]", arg129_1: "f32[4096]", arg130_1: "f32[1024, 4096]", arg131_1: "f32[1024]", arg132_1: "f32[1024]", arg133_1: "f32[1024]", arg134_1: "f32[3072, 1024]", arg135_1: "f32[3072]", arg136_1: "f32[1024, 1024]", arg137_1: "f32[1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024]", arg140_1: "f32[4096, 1024]", arg141_1: "f32[4096]", arg142_1: "f32[1024, 4096]", arg143_1: "f32[1024]", arg144_1: "f32[1024]", arg145_1: "f32[1024]", arg146_1: "f32[3072, 1024]", arg147_1: "f32[3072]", arg148_1: "f32[1024, 1024]", arg149_1: "f32[1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024]", arg152_1: "f32[4096, 1024]", arg153_1: "f32[4096]", arg154_1: "f32[1024, 4096]", arg155_1: "f32[1024]", arg156_1: "f32[1024]", arg157_1: "f32[1024]", arg158_1: "f32[3072, 1024]", arg159_1: "f32[3072]", arg160_1: "f32[1024, 1024]", arg161_1: "f32[1024]", arg162_1: "f32[1024]", arg163_1: "f32[1024]", arg164_1: "f32[4096, 1024]", arg165_1: "f32[4096]", arg166_1: "f32[1024, 4096]", arg167_1: "f32[1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024]", arg170_1: "f32[1000, 1024]", arg171_1: "f32[1000]", arg172_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:138, code: x = self.conv(x)
    convolution: "f32[8, 256, 31, 31]" = torch.ops.aten.convolution.default(arg172_1, arg2_1, arg3_1, [7, 7], [0, 0], [1, 1], False, [0, 0], 1);  arg172_1 = arg2_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:257, code: x = self.pos_drop(x + self.pos_embed)
    add: "f32[8, 256, 31, 31]" = torch.ops.aten.add.Tensor(convolution, arg0_1);  convolution = arg0_1 = None
    clone: "f32[8, 256, 31, 31]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:258, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    expand: "f32[8, 1, 256]" = torch.ops.aten.expand.default(arg1_1, [8, -1, -1]);  arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    view: "f32[8, 256, 961]" = torch.ops.aten.view.default(clone, [8, 256, 961]);  clone = None
    permute: "f32[8, 961, 256]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    cat: "f32[8, 962, 256]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 962, 1]" = var_mean[0]
    getitem_1: "f32[8, 962, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(cat, getitem_1);  getitem_1 = None
    mul: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
    add_2: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_1: "f32[7696, 256]" = torch.ops.aten.view.default(add_2, [7696, 256]);  add_2 = None
    permute_1: "f32[256, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm: "f32[7696, 768]" = torch.ops.aten.addmm.default(arg7_1, view_1, permute_1);  arg7_1 = view_1 = permute_1 = None
    view_2: "f32[8, 962, 768]" = torch.ops.aten.view.default(addmm, [8, 962, 768]);  addmm = None
    view_3: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.view.default(view_2, [8, 962, 3, 4, 64]);  view_2 = None
    permute_2: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_2: "f32[8, 4, 962, 64]" = unbind[0]
    getitem_3: "f32[8, 4, 962, 64]" = unbind[1]
    getitem_4: "f32[8, 4, 962, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_2, getitem_3, getitem_4, None, False);  getitem_2 = getitem_3 = getitem_4 = None
    getitem_5: "f32[8, 4, 962, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_3: "f32[8, 962, 4, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_4: "f32[8, 962, 256]" = torch.ops.aten.view.default(permute_3, [8, 962, 256]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_5: "f32[7696, 256]" = torch.ops.aten.view.default(view_4, [7696, 256]);  view_4 = None
    permute_4: "f32[256, 256]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    addmm_1: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg9_1, view_5, permute_4);  arg9_1 = view_5 = permute_4 = None
    view_6: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_1, [8, 962, 256]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_1: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_6);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_3: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(cat, clone_1);  cat = clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_9: "f32[8, 962, 1]" = var_mean_1[0]
    getitem_10: "f32[8, 962, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
    rsqrt_1: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_3, getitem_10);  getitem_10 = None
    mul_2: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_3: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = arg10_1 = None
    add_5: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_7: "f32[7696, 256]" = torch.ops.aten.view.default(add_5, [7696, 256]);  add_5 = None
    permute_5: "f32[256, 1024]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    addmm_2: "f32[7696, 1024]" = torch.ops.aten.addmm.default(arg13_1, view_7, permute_5);  arg13_1 = view_7 = permute_5 = None
    view_8: "f32[8, 962, 1024]" = torch.ops.aten.view.default(addmm_2, [8, 962, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_4: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_8, 0.5)
    mul_5: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476);  view_8 = None
    erf: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_2: "f32[8, 962, 1024]" = torch.ops.aten.clone.default(mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_9: "f32[7696, 1024]" = torch.ops.aten.view.default(clone_2, [7696, 1024]);  clone_2 = None
    permute_6: "f32[1024, 256]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    addmm_3: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg15_1, view_9, permute_6);  arg15_1 = view_9 = permute_6 = None
    view_10: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_3, [8, 962, 256]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_3: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_10);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_7: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_3, clone_3);  add_3 = clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_11: "f32[8, 962, 1]" = var_mean_2[0]
    getitem_12: "f32[8, 962, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
    rsqrt_2: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_2: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_7, getitem_12);  getitem_12 = None
    mul_7: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_8: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_7, arg16_1);  mul_7 = arg16_1 = None
    add_9: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_8, arg17_1);  mul_8 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_11: "f32[7696, 256]" = torch.ops.aten.view.default(add_9, [7696, 256]);  add_9 = None
    permute_7: "f32[256, 768]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    addmm_4: "f32[7696, 768]" = torch.ops.aten.addmm.default(arg19_1, view_11, permute_7);  arg19_1 = view_11 = permute_7 = None
    view_12: "f32[8, 962, 768]" = torch.ops.aten.view.default(addmm_4, [8, 962, 768]);  addmm_4 = None
    view_13: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.view.default(view_12, [8, 962, 3, 4, 64]);  view_12 = None
    permute_8: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.permute.default(view_13, [2, 0, 3, 1, 4]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_8);  permute_8 = None
    getitem_13: "f32[8, 4, 962, 64]" = unbind_1[0]
    getitem_14: "f32[8, 4, 962, 64]" = unbind_1[1]
    getitem_15: "f32[8, 4, 962, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_13, getitem_14, getitem_15, None, False);  getitem_13 = getitem_14 = getitem_15 = None
    getitem_16: "f32[8, 4, 962, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_9: "f32[8, 962, 4, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    view_14: "f32[8, 962, 256]" = torch.ops.aten.view.default(permute_9, [8, 962, 256]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_15: "f32[7696, 256]" = torch.ops.aten.view.default(view_14, [7696, 256]);  view_14 = None
    permute_10: "f32[256, 256]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_5: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg21_1, view_15, permute_10);  arg21_1 = view_15 = permute_10 = None
    view_16: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_5, [8, 962, 256]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_4: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_10: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_7, clone_4);  add_7 = clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 962, 1]" = var_mean_3[0]
    getitem_21: "f32[8, 962, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_3: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_10, getitem_21);  getitem_21 = None
    mul_9: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    mul_10: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
    add_12: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_10, arg23_1);  mul_10 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[7696, 256]" = torch.ops.aten.view.default(add_12, [7696, 256]);  add_12 = None
    permute_11: "f32[256, 1024]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_6: "f32[7696, 1024]" = torch.ops.aten.addmm.default(arg25_1, view_17, permute_11);  arg25_1 = view_17 = permute_11 = None
    view_18: "f32[8, 962, 1024]" = torch.ops.aten.view.default(addmm_6, [8, 962, 1024]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_11: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_12: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf_1: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_5: "f32[8, 962, 1024]" = torch.ops.aten.clone.default(mul_13);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[7696, 1024]" = torch.ops.aten.view.default(clone_5, [7696, 1024]);  clone_5 = None
    permute_12: "f32[1024, 256]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_7: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg27_1, view_19, permute_12);  arg27_1 = view_19 = permute_12 = None
    view_20: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_7, [8, 962, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_6: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_14: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_10, clone_6);  add_10 = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 962, 1]" = var_mean_4[0]
    getitem_23: "f32[8, 962, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_4: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_4: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_14, getitem_23);  getitem_23 = None
    mul_14: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    mul_15: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_14, arg28_1);  mul_14 = arg28_1 = None
    add_16: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_15, arg29_1);  mul_15 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_21: "f32[7696, 256]" = torch.ops.aten.view.default(add_16, [7696, 256]);  add_16 = None
    permute_13: "f32[256, 768]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_8: "f32[7696, 768]" = torch.ops.aten.addmm.default(arg31_1, view_21, permute_13);  arg31_1 = view_21 = permute_13 = None
    view_22: "f32[8, 962, 768]" = torch.ops.aten.view.default(addmm_8, [8, 962, 768]);  addmm_8 = None
    view_23: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.view.default(view_22, [8, 962, 3, 4, 64]);  view_22 = None
    permute_14: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
    getitem_24: "f32[8, 4, 962, 64]" = unbind_2[0]
    getitem_25: "f32[8, 4, 962, 64]" = unbind_2[1]
    getitem_26: "f32[8, 4, 962, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_24, getitem_25, getitem_26, None, False);  getitem_24 = getitem_25 = getitem_26 = None
    getitem_27: "f32[8, 4, 962, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_15: "f32[8, 962, 4, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
    view_24: "f32[8, 962, 256]" = torch.ops.aten.view.default(permute_15, [8, 962, 256]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_25: "f32[7696, 256]" = torch.ops.aten.view.default(view_24, [7696, 256]);  view_24 = None
    permute_16: "f32[256, 256]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    addmm_9: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg33_1, view_25, permute_16);  arg33_1 = view_25 = permute_16 = None
    view_26: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_9, [8, 962, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_7: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_26);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_17: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_14, clone_7);  add_14 = clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_31: "f32[8, 962, 1]" = var_mean_5[0]
    getitem_32: "f32[8, 962, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-06);  getitem_31 = None
    rsqrt_5: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_5: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_32);  getitem_32 = None
    mul_16: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    mul_17: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_16, arg34_1);  mul_16 = arg34_1 = None
    add_19: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_27: "f32[7696, 256]" = torch.ops.aten.view.default(add_19, [7696, 256]);  add_19 = None
    permute_17: "f32[256, 1024]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_10: "f32[7696, 1024]" = torch.ops.aten.addmm.default(arg37_1, view_27, permute_17);  arg37_1 = view_27 = permute_17 = None
    view_28: "f32[8, 962, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 962, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_28, 0.5)
    mul_19: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476);  view_28 = None
    erf_2: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_8: "f32[8, 962, 1024]" = torch.ops.aten.clone.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_29: "f32[7696, 1024]" = torch.ops.aten.view.default(clone_8, [7696, 1024]);  clone_8 = None
    permute_18: "f32[1024, 256]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    addmm_11: "f32[7696, 256]" = torch.ops.aten.addmm.default(arg39_1, view_29, permute_18);  arg39_1 = view_29 = permute_18 = None
    view_30: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_11, [8, 962, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_9: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_30);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_21: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_17, clone_9);  add_17 = clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    slice_1: "f32[8, 962, 256]" = torch.ops.aten.slice.Tensor(add_21, 0, 0, 9223372036854775807)
    slice_2: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 1);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:88, code: x = x[:, token_length:]
    slice_3: "f32[8, 962, 256]" = torch.ops.aten.slice.Tensor(add_21, 0, 0, 9223372036854775807);  add_21 = None
    slice_4: "f32[8, 961, 256]" = torch.ops.aten.slice.Tensor(slice_3, 1, 1, 9223372036854775807);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:89, code: x = x.transpose(1, 2).reshape(B, C, H, W)
    permute_19: "f32[8, 256, 961]" = torch.ops.aten.permute.default(slice_4, [0, 2, 1]);  slice_4 = None
    view_31: "f32[8, 256, 31, 31]" = torch.ops.aten.view.default(permute_19, [8, 256, 31, 31]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:110, code: x = self.conv(x)
    convolution_1: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(view_31, arg40_1, arg41_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 256);  view_31 = arg40_1 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    permute_20: "f32[256, 512]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    view_32: "f32[8, 256]" = torch.ops.aten.view.default(slice_2, [8, 256]);  slice_2 = None
    mm: "f32[8, 512]" = torch.ops.aten.mm.default(view_32, permute_20);  view_32 = permute_20 = None
    view_33: "f32[8, 1, 512]" = torch.ops.aten.view.default(mm, [8, 1, 512]);  mm = None
    add_22: "f32[8, 1, 512]" = torch.ops.aten.add.Tensor(view_33, arg43_1);  view_33 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    view_34: "f32[8, 512, 256]" = torch.ops.aten.view.default(convolution_1, [8, 512, 256]);  convolution_1 = None
    permute_21: "f32[8, 256, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_1: "f32[8, 257, 512]" = torch.ops.aten.cat.default([add_22, permute_21], 1);  add_22 = permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 257, 1]" = var_mean_6[0]
    getitem_34: "f32[8, 257, 1]" = var_mean_6[1];  var_mean_6 = None
    add_23: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_6: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_6: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(cat_1, getitem_34);  getitem_34 = None
    mul_21: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    mul_22: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_21, arg44_1);  mul_21 = arg44_1 = None
    add_24: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_22, arg45_1);  mul_22 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_35: "f32[2056, 512]" = torch.ops.aten.view.default(add_24, [2056, 512]);  add_24 = None
    permute_22: "f32[512, 1536]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_12: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg47_1, view_35, permute_22);  arg47_1 = view_35 = permute_22 = None
    view_36: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_12, [8, 257, 1536]);  addmm_12 = None
    view_37: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_36, [8, 257, 3, 8, 64]);  view_36 = None
    permute_23: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_37, [2, 0, 3, 1, 4]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_23);  permute_23 = None
    getitem_35: "f32[8, 8, 257, 64]" = unbind_3[0]
    getitem_36: "f32[8, 8, 257, 64]" = unbind_3[1]
    getitem_37: "f32[8, 8, 257, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_35, getitem_36, getitem_37, None, False);  getitem_35 = getitem_36 = getitem_37 = None
    getitem_38: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_24: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
    view_38: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_24, [8, 257, 512]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_39: "f32[2056, 512]" = torch.ops.aten.view.default(view_38, [2056, 512]);  view_38 = None
    permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_13: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg49_1, view_39, permute_25);  arg49_1 = view_39 = permute_25 = None
    view_40: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_13, [8, 257, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_10: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_25: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(cat_1, clone_10);  cat_1 = clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 257, 1]" = var_mean_7[0]
    getitem_43: "f32[8, 257, 1]" = var_mean_7[1];  var_mean_7 = None
    add_26: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_7: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_7: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_25, getitem_43);  getitem_43 = None
    mul_23: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    mul_24: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_23, arg50_1);  mul_23 = arg50_1 = None
    add_27: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_24, arg51_1);  mul_24 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_41: "f32[2056, 512]" = torch.ops.aten.view.default(add_27, [2056, 512]);  add_27 = None
    permute_26: "f32[512, 2048]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_14: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg53_1, view_41, permute_26);  arg53_1 = view_41 = permute_26 = None
    view_42: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_14, [8, 257, 2048]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_25: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_42, 0.5)
    mul_26: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_42, 0.7071067811865476);  view_42 = None
    erf_3: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_28: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_27: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_11: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_27);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_43: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_11, [2056, 2048]);  clone_11 = None
    permute_27: "f32[2048, 512]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_15: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg55_1, view_43, permute_27);  arg55_1 = view_43 = permute_27 = None
    view_44: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_15, [8, 257, 512]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_12: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_29: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_25, clone_12);  add_25 = clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 257, 1]" = var_mean_8[0]
    getitem_45: "f32[8, 257, 1]" = var_mean_8[1];  var_mean_8 = None
    add_30: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_8: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_8: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_29, getitem_45);  getitem_45 = None
    mul_28: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    mul_29: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_28, arg56_1);  mul_28 = arg56_1 = None
    add_31: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_29, arg57_1);  mul_29 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_45: "f32[2056, 512]" = torch.ops.aten.view.default(add_31, [2056, 512]);  add_31 = None
    permute_28: "f32[512, 1536]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_16: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg59_1, view_45, permute_28);  arg59_1 = view_45 = permute_28 = None
    view_46: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 257, 1536]);  addmm_16 = None
    view_47: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_46, [8, 257, 3, 8, 64]);  view_46 = None
    permute_29: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_47, [2, 0, 3, 1, 4]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_29);  permute_29 = None
    getitem_46: "f32[8, 8, 257, 64]" = unbind_4[0]
    getitem_47: "f32[8, 8, 257, 64]" = unbind_4[1]
    getitem_48: "f32[8, 8, 257, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_46, getitem_47, getitem_48, None, False);  getitem_46 = getitem_47 = getitem_48 = None
    getitem_49: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_30: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
    view_48: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_30, [8, 257, 512]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_49: "f32[2056, 512]" = torch.ops.aten.view.default(view_48, [2056, 512]);  view_48 = None
    permute_31: "f32[512, 512]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    addmm_17: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg61_1, view_49, permute_31);  arg61_1 = view_49 = permute_31 = None
    view_50: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_17, [8, 257, 512]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_13: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_50);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_32: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_29, clone_13);  add_29 = clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_53: "f32[8, 257, 1]" = var_mean_9[0]
    getitem_54: "f32[8, 257, 1]" = var_mean_9[1];  var_mean_9 = None
    add_33: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-06);  getitem_53 = None
    rsqrt_9: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_9: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_32, getitem_54);  getitem_54 = None
    mul_30: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    mul_31: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_30, arg62_1);  mul_30 = arg62_1 = None
    add_34: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_31, arg63_1);  mul_31 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_51: "f32[2056, 512]" = torch.ops.aten.view.default(add_34, [2056, 512]);  add_34 = None
    permute_32: "f32[512, 2048]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_18: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg65_1, view_51, permute_32);  arg65_1 = view_51 = permute_32 = None
    view_52: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 257, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_52, 0.5)
    mul_33: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_52, 0.7071067811865476);  view_52 = None
    erf_4: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_35: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_34: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_14: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_14, [2056, 2048]);  clone_14 = None
    permute_33: "f32[2048, 512]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    addmm_19: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg67_1, view_53, permute_33);  arg67_1 = view_53 = permute_33 = None
    view_54: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_19, [8, 257, 512]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_15: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_36: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_32, clone_15);  add_32 = clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_55: "f32[8, 257, 1]" = var_mean_10[0]
    getitem_56: "f32[8, 257, 1]" = var_mean_10[1];  var_mean_10 = None
    add_37: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_10: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_10: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_36, getitem_56);  getitem_56 = None
    mul_35: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    mul_36: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_35, arg68_1);  mul_35 = arg68_1 = None
    add_38: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_36, arg69_1);  mul_36 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_55: "f32[2056, 512]" = torch.ops.aten.view.default(add_38, [2056, 512]);  add_38 = None
    permute_34: "f32[512, 1536]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_20: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg71_1, view_55, permute_34);  arg71_1 = view_55 = permute_34 = None
    view_56: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_20, [8, 257, 1536]);  addmm_20 = None
    view_57: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_56, [8, 257, 3, 8, 64]);  view_56 = None
    permute_35: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_57, [2, 0, 3, 1, 4]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_35);  permute_35 = None
    getitem_57: "f32[8, 8, 257, 64]" = unbind_5[0]
    getitem_58: "f32[8, 8, 257, 64]" = unbind_5[1]
    getitem_59: "f32[8, 8, 257, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_57, getitem_58, getitem_59, None, False);  getitem_57 = getitem_58 = getitem_59 = None
    getitem_60: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_36: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
    view_58: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_36, [8, 257, 512]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_59: "f32[2056, 512]" = torch.ops.aten.view.default(view_58, [2056, 512]);  view_58 = None
    permute_37: "f32[512, 512]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_21: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg73_1, view_59, permute_37);  arg73_1 = view_59 = permute_37 = None
    view_60: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_21, [8, 257, 512]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_16: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_39: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_36, clone_16);  add_36 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 257, 1]" = var_mean_11[0]
    getitem_65: "f32[8, 257, 1]" = var_mean_11[1];  var_mean_11 = None
    add_40: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_11: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_11: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_39, getitem_65);  getitem_65 = None
    mul_37: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    mul_38: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_37, arg74_1);  mul_37 = arg74_1 = None
    add_41: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_38, arg75_1);  mul_38 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_61: "f32[2056, 512]" = torch.ops.aten.view.default(add_41, [2056, 512]);  add_41 = None
    permute_38: "f32[512, 2048]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    addmm_22: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg77_1, view_61, permute_38);  arg77_1 = view_61 = permute_38 = None
    view_62: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 257, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_39: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_62, 0.5)
    mul_40: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476);  view_62 = None
    erf_5: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_42: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_41: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_39, add_42);  mul_39 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_17: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_41);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_63: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_17, [2056, 2048]);  clone_17 = None
    permute_39: "f32[2048, 512]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_23: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg79_1, view_63, permute_39);  arg79_1 = view_63 = permute_39 = None
    view_64: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_23, [8, 257, 512]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_18: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_64);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_43: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_39, clone_18);  add_39 = clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 257, 1]" = var_mean_12[0]
    getitem_67: "f32[8, 257, 1]" = var_mean_12[1];  var_mean_12 = None
    add_44: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_12: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_12: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_43, getitem_67);  getitem_67 = None
    mul_42: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    mul_43: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_42, arg80_1);  mul_42 = arg80_1 = None
    add_45: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_43, arg81_1);  mul_43 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_65: "f32[2056, 512]" = torch.ops.aten.view.default(add_45, [2056, 512]);  add_45 = None
    permute_40: "f32[512, 1536]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm_24: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg83_1, view_65, permute_40);  arg83_1 = view_65 = permute_40 = None
    view_66: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_24, [8, 257, 1536]);  addmm_24 = None
    view_67: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_66, [8, 257, 3, 8, 64]);  view_66 = None
    permute_41: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_67, [2, 0, 3, 1, 4]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_41);  permute_41 = None
    getitem_68: "f32[8, 8, 257, 64]" = unbind_6[0]
    getitem_69: "f32[8, 8, 257, 64]" = unbind_6[1]
    getitem_70: "f32[8, 8, 257, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_68, getitem_69, getitem_70, None, False);  getitem_68 = getitem_69 = getitem_70 = None
    getitem_71: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_42: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
    view_68: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_42, [8, 257, 512]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_69: "f32[2056, 512]" = torch.ops.aten.view.default(view_68, [2056, 512]);  view_68 = None
    permute_43: "f32[512, 512]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_25: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg85_1, view_69, permute_43);  arg85_1 = view_69 = permute_43 = None
    view_70: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_25, [8, 257, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_19: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_70);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_46: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_43, clone_19);  add_43 = clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_75: "f32[8, 257, 1]" = var_mean_13[0]
    getitem_76: "f32[8, 257, 1]" = var_mean_13[1];  var_mean_13 = None
    add_47: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
    rsqrt_13: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_13: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_46, getitem_76);  getitem_76 = None
    mul_44: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    mul_45: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_44, arg86_1);  mul_44 = arg86_1 = None
    add_48: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_45, arg87_1);  mul_45 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_71: "f32[2056, 512]" = torch.ops.aten.view.default(add_48, [2056, 512]);  add_48 = None
    permute_44: "f32[512, 2048]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_26: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg89_1, view_71, permute_44);  arg89_1 = view_71 = permute_44 = None
    view_72: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 257, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_72, 0.5)
    mul_47: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_72, 0.7071067811865476);  view_72 = None
    erf_6: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_49: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_48: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_73: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_20, [2056, 2048]);  clone_20 = None
    permute_45: "f32[2048, 512]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_27: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg91_1, view_73, permute_45);  arg91_1 = view_73 = permute_45 = None
    view_74: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_27, [8, 257, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_74);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_50: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_46, clone_21);  add_46 = clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 257, 1]" = var_mean_14[0]
    getitem_78: "f32[8, 257, 1]" = var_mean_14[1];  var_mean_14 = None
    add_51: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_14: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_14: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_50, getitem_78);  getitem_78 = None
    mul_49: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    mul_50: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_49, arg92_1);  mul_49 = arg92_1 = None
    add_52: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_50, arg93_1);  mul_50 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_75: "f32[2056, 512]" = torch.ops.aten.view.default(add_52, [2056, 512]);  add_52 = None
    permute_46: "f32[512, 1536]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_28: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg95_1, view_75, permute_46);  arg95_1 = view_75 = permute_46 = None
    view_76: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 257, 1536]);  addmm_28 = None
    view_77: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_76, [8, 257, 3, 8, 64]);  view_76 = None
    permute_47: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_77, [2, 0, 3, 1, 4]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
    getitem_79: "f32[8, 8, 257, 64]" = unbind_7[0]
    getitem_80: "f32[8, 8, 257, 64]" = unbind_7[1]
    getitem_81: "f32[8, 8, 257, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_79, getitem_80, getitem_81, None, False);  getitem_79 = getitem_80 = getitem_81 = None
    getitem_82: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_48: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
    view_78: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_48, [8, 257, 512]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_79: "f32[2056, 512]" = torch.ops.aten.view.default(view_78, [2056, 512]);  view_78 = None
    permute_49: "f32[512, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_29: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg97_1, view_79, permute_49);  arg97_1 = view_79 = permute_49 = None
    view_80: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_29, [8, 257, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_22: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_53: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_50, clone_22);  add_50 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 257, 1]" = var_mean_15[0]
    getitem_87: "f32[8, 257, 1]" = var_mean_15[1];  var_mean_15 = None
    add_54: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_15: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_15: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_53, getitem_87);  getitem_87 = None
    mul_51: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    mul_52: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_51, arg98_1);  mul_51 = arg98_1 = None
    add_55: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_52, arg99_1);  mul_52 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_81: "f32[2056, 512]" = torch.ops.aten.view.default(add_55, [2056, 512]);  add_55 = None
    permute_50: "f32[512, 2048]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_30: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg101_1, view_81, permute_50);  arg101_1 = view_81 = permute_50 = None
    view_82: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 257, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
    mul_54: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_82, 0.7071067811865476);  view_82 = None
    erf_7: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_56: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_55: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_53, add_56);  mul_53 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_55);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_23, [2056, 2048]);  clone_23 = None
    permute_51: "f32[2048, 512]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_31: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg103_1, view_83, permute_51);  arg103_1 = view_83 = permute_51 = None
    view_84: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_31, [8, 257, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_57: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_53, clone_24);  add_53 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 257, 1]" = var_mean_16[0]
    getitem_89: "f32[8, 257, 1]" = var_mean_16[1];  var_mean_16 = None
    add_58: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_16: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_16: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_57, getitem_89);  getitem_89 = None
    mul_56: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    mul_57: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_56, arg104_1);  mul_56 = arg104_1 = None
    add_59: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_57, arg105_1);  mul_57 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_85: "f32[2056, 512]" = torch.ops.aten.view.default(add_59, [2056, 512]);  add_59 = None
    permute_52: "f32[512, 1536]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_32: "f32[2056, 1536]" = torch.ops.aten.addmm.default(arg107_1, view_85, permute_52);  arg107_1 = view_85 = permute_52 = None
    view_86: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_32, [8, 257, 1536]);  addmm_32 = None
    view_87: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_86, [8, 257, 3, 8, 64]);  view_86 = None
    permute_53: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
    getitem_90: "f32[8, 8, 257, 64]" = unbind_8[0]
    getitem_91: "f32[8, 8, 257, 64]" = unbind_8[1]
    getitem_92: "f32[8, 8, 257, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_90, getitem_91, getitem_92, None, False);  getitem_90 = getitem_91 = getitem_92 = None
    getitem_93: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_54: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
    view_88: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_54, [8, 257, 512]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_89: "f32[2056, 512]" = torch.ops.aten.view.default(view_88, [2056, 512]);  view_88 = None
    permute_55: "f32[512, 512]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_33: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg109_1, view_89, permute_55);  arg109_1 = view_89 = permute_55 = None
    view_90: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_33, [8, 257, 512]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_25: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_60: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_57, clone_25);  add_57 = clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
    getitem_97: "f32[8, 257, 1]" = var_mean_17[0]
    getitem_98: "f32[8, 257, 1]" = var_mean_17[1];  var_mean_17 = None
    add_61: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_17: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_17: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_60, getitem_98);  getitem_98 = None
    mul_58: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    mul_59: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_58, arg110_1);  mul_58 = arg110_1 = None
    add_62: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_59, arg111_1);  mul_59 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_91: "f32[2056, 512]" = torch.ops.aten.view.default(add_62, [2056, 512]);  add_62 = None
    permute_56: "f32[512, 2048]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_34: "f32[2056, 2048]" = torch.ops.aten.addmm.default(arg113_1, view_91, permute_56);  arg113_1 = view_91 = permute_56 = None
    view_92: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 257, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_60: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_92, 0.5)
    mul_61: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476);  view_92 = None
    erf_8: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_63: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_62: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_26: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_93: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_26, [2056, 2048]);  clone_26 = None
    permute_57: "f32[2048, 512]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_35: "f32[2056, 512]" = torch.ops.aten.addmm.default(arg115_1, view_93, permute_57);  arg115_1 = view_93 = permute_57 = None
    view_94: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_35, [8, 257, 512]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_27: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_94);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_64: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_60, clone_27);  add_60 = clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    slice_5: "f32[8, 257, 512]" = torch.ops.aten.slice.Tensor(add_64, 0, 0, 9223372036854775807)
    slice_6: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 1);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:88, code: x = x[:, token_length:]
    slice_7: "f32[8, 257, 512]" = torch.ops.aten.slice.Tensor(add_64, 0, 0, 9223372036854775807);  add_64 = None
    slice_8: "f32[8, 256, 512]" = torch.ops.aten.slice.Tensor(slice_7, 1, 1, 9223372036854775807);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:89, code: x = x.transpose(1, 2).reshape(B, C, H, W)
    permute_58: "f32[8, 512, 256]" = torch.ops.aten.permute.default(slice_8, [0, 2, 1]);  slice_8 = None
    view_95: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(permute_58, [8, 512, 16, 16]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:110, code: x = self.conv(x)
    convolution_2: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(view_95, arg116_1, arg117_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 512);  view_95 = arg116_1 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    permute_59: "f32[512, 1024]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    view_96: "f32[8, 512]" = torch.ops.aten.view.default(slice_6, [8, 512]);  slice_6 = None
    mm_1: "f32[8, 1024]" = torch.ops.aten.mm.default(view_96, permute_59);  view_96 = permute_59 = None
    view_97: "f32[8, 1, 1024]" = torch.ops.aten.view.default(mm_1, [8, 1, 1024]);  mm_1 = None
    add_65: "f32[8, 1, 1024]" = torch.ops.aten.add.Tensor(view_97, arg119_1);  view_97 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    view_98: "f32[8, 1024, 64]" = torch.ops.aten.view.default(convolution_2, [8, 1024, 64]);  convolution_2 = None
    permute_60: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_2: "f32[8, 65, 1024]" = torch.ops.aten.cat.default([add_65, permute_60], 1);  add_65 = permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_99: "f32[8, 65, 1]" = var_mean_18[0]
    getitem_100: "f32[8, 65, 1]" = var_mean_18[1];  var_mean_18 = None
    add_66: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_18: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_18: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(cat_2, getitem_100);  getitem_100 = None
    mul_63: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    mul_64: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_63, arg120_1);  mul_63 = arg120_1 = None
    add_67: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_64, arg121_1);  mul_64 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_99: "f32[520, 1024]" = torch.ops.aten.view.default(add_67, [520, 1024]);  add_67 = None
    permute_61: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_36: "f32[520, 3072]" = torch.ops.aten.addmm.default(arg123_1, view_99, permute_61);  arg123_1 = view_99 = permute_61 = None
    view_100: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_36, [8, 65, 3072]);  addmm_36 = None
    view_101: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_100, [8, 65, 3, 16, 64]);  view_100 = None
    permute_62: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_101, [2, 0, 3, 1, 4]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
    getitem_101: "f32[8, 16, 65, 64]" = unbind_9[0]
    getitem_102: "f32[8, 16, 65, 64]" = unbind_9[1]
    getitem_103: "f32[8, 16, 65, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_101, getitem_102, getitem_103, None, False);  getitem_101 = getitem_102 = getitem_103 = None
    getitem_104: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_63: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
    view_102: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_63, [8, 65, 1024]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_103: "f32[520, 1024]" = torch.ops.aten.view.default(view_102, [520, 1024]);  view_102 = None
    permute_64: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_37: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg125_1, view_103, permute_64);  arg125_1 = view_103 = permute_64 = None
    view_104: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_37, [8, 65, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_28: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_68: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(cat_2, clone_28);  cat_2 = clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 65, 1]" = var_mean_19[0]
    getitem_109: "f32[8, 65, 1]" = var_mean_19[1];  var_mean_19 = None
    add_69: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_19: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_19: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_68, getitem_109);  getitem_109 = None
    mul_65: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    mul_66: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_65, arg126_1);  mul_65 = arg126_1 = None
    add_70: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_66, arg127_1);  mul_66 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[520, 1024]" = torch.ops.aten.view.default(add_70, [520, 1024]);  add_70 = None
    permute_65: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_38: "f32[520, 4096]" = torch.ops.aten.addmm.default(arg129_1, view_105, permute_65);  arg129_1 = view_105 = permute_65 = None
    view_106: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_38, [8, 65, 4096]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_68: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476);  view_106 = None
    erf_9: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_71: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_69: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_67, add_71);  mul_67 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 65, 4096]" = torch.ops.aten.clone.default(mul_69);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[520, 4096]" = torch.ops.aten.view.default(clone_29, [520, 4096]);  clone_29 = None
    permute_66: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_39: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg131_1, view_107, permute_66);  arg131_1 = view_107 = permute_66 = None
    view_108: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_39, [8, 65, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_72: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_68, clone_30);  add_68 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 65, 1]" = var_mean_20[0]
    getitem_111: "f32[8, 65, 1]" = var_mean_20[1];  var_mean_20 = None
    add_73: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_20: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_20: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_72, getitem_111);  getitem_111 = None
    mul_70: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    mul_71: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_70, arg132_1);  mul_70 = arg132_1 = None
    add_74: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_71, arg133_1);  mul_71 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_109: "f32[520, 1024]" = torch.ops.aten.view.default(add_74, [520, 1024]);  add_74 = None
    permute_67: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_40: "f32[520, 3072]" = torch.ops.aten.addmm.default(arg135_1, view_109, permute_67);  arg135_1 = view_109 = permute_67 = None
    view_110: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_40, [8, 65, 3072]);  addmm_40 = None
    view_111: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_110, [8, 65, 3, 16, 64]);  view_110 = None
    permute_68: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
    getitem_112: "f32[8, 16, 65, 64]" = unbind_10[0]
    getitem_113: "f32[8, 16, 65, 64]" = unbind_10[1]
    getitem_114: "f32[8, 16, 65, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_112, getitem_113, getitem_114, None, False);  getitem_112 = getitem_113 = getitem_114 = None
    getitem_115: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_69: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
    view_112: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_69, [8, 65, 1024]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_113: "f32[520, 1024]" = torch.ops.aten.view.default(view_112, [520, 1024]);  view_112 = None
    permute_70: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_41: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg137_1, view_113, permute_70);  arg137_1 = view_113 = permute_70 = None
    view_114: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_41, [8, 65, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_31: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_114);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_75: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_72, clone_31);  add_72 = clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_119: "f32[8, 65, 1]" = var_mean_21[0]
    getitem_120: "f32[8, 65, 1]" = var_mean_21[1];  var_mean_21 = None
    add_76: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-06);  getitem_119 = None
    rsqrt_21: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_21: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_75, getitem_120);  getitem_120 = None
    mul_72: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    mul_73: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_72, arg138_1);  mul_72 = arg138_1 = None
    add_77: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_73, arg139_1);  mul_73 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_115: "f32[520, 1024]" = torch.ops.aten.view.default(add_77, [520, 1024]);  add_77 = None
    permute_71: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_42: "f32[520, 4096]" = torch.ops.aten.addmm.default(arg141_1, view_115, permute_71);  arg141_1 = view_115 = permute_71 = None
    view_116: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_42, [8, 65, 4096]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_75: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476);  view_116 = None
    erf_10: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_78: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_76: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_74, add_78);  mul_74 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 65, 4096]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_117: "f32[520, 4096]" = torch.ops.aten.view.default(clone_32, [520, 4096]);  clone_32 = None
    permute_72: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_43: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg143_1, view_117, permute_72);  arg143_1 = view_117 = permute_72 = None
    view_118: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_43, [8, 65, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_118);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_79: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_75, clone_33);  add_75 = clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_22 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_121: "f32[8, 65, 1]" = var_mean_22[0]
    getitem_122: "f32[8, 65, 1]" = var_mean_22[1];  var_mean_22 = None
    add_80: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_22: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_22: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_79, getitem_122);  getitem_122 = None
    mul_77: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    mul_78: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_77, arg144_1);  mul_77 = arg144_1 = None
    add_81: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_78, arg145_1);  mul_78 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_119: "f32[520, 1024]" = torch.ops.aten.view.default(add_81, [520, 1024]);  add_81 = None
    permute_73: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    addmm_44: "f32[520, 3072]" = torch.ops.aten.addmm.default(arg147_1, view_119, permute_73);  arg147_1 = view_119 = permute_73 = None
    view_120: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_44, [8, 65, 3072]);  addmm_44 = None
    view_121: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_120, [8, 65, 3, 16, 64]);  view_120 = None
    permute_74: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_121, [2, 0, 3, 1, 4]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_123: "f32[8, 16, 65, 64]" = unbind_11[0]
    getitem_124: "f32[8, 16, 65, 64]" = unbind_11[1]
    getitem_125: "f32[8, 16, 65, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_123, getitem_124, getitem_125, None, False);  getitem_123 = getitem_124 = getitem_125 = None
    getitem_126: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_75: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
    view_122: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_75, [8, 65, 1024]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_123: "f32[520, 1024]" = torch.ops.aten.view.default(view_122, [520, 1024]);  view_122 = None
    permute_76: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_45: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg149_1, view_123, permute_76);  arg149_1 = view_123 = permute_76 = None
    view_124: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_45, [8, 65, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_34: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_124);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_82: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_79, clone_34);  add_79 = clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_23 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 65, 1]" = var_mean_23[0]
    getitem_131: "f32[8, 65, 1]" = var_mean_23[1];  var_mean_23 = None
    add_83: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
    rsqrt_23: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_23: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_82, getitem_131);  getitem_131 = None
    mul_79: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    mul_80: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_79, arg150_1);  mul_79 = arg150_1 = None
    add_84: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_80, arg151_1);  mul_80 = arg151_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[520, 1024]" = torch.ops.aten.view.default(add_84, [520, 1024]);  add_84 = None
    permute_77: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_46: "f32[520, 4096]" = torch.ops.aten.addmm.default(arg153_1, view_125, permute_77);  arg153_1 = view_125 = permute_77 = None
    view_126: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_46, [8, 65, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_81: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    mul_82: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476);  view_126 = None
    erf_11: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_85: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_83: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_81, add_85);  mul_81 = add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_35: "f32[8, 65, 4096]" = torch.ops.aten.clone.default(mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_127: "f32[520, 4096]" = torch.ops.aten.view.default(clone_35, [520, 4096]);  clone_35 = None
    permute_78: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_47: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg155_1, view_127, permute_78);  arg155_1 = view_127 = permute_78 = None
    view_128: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_47, [8, 65, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_36: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_128);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_86: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_82, clone_36);  add_82 = clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_24 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 65, 1]" = var_mean_24[0]
    getitem_133: "f32[8, 65, 1]" = var_mean_24[1];  var_mean_24 = None
    add_87: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_24: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_24: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_86, getitem_133);  getitem_133 = None
    mul_84: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    mul_85: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_84, arg156_1);  mul_84 = arg156_1 = None
    add_88: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_85, arg157_1);  mul_85 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_129: "f32[520, 1024]" = torch.ops.aten.view.default(add_88, [520, 1024]);  add_88 = None
    permute_79: "f32[1024, 3072]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_48: "f32[520, 3072]" = torch.ops.aten.addmm.default(arg159_1, view_129, permute_79);  arg159_1 = view_129 = permute_79 = None
    view_130: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_48, [8, 65, 3072]);  addmm_48 = None
    view_131: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_130, [8, 65, 3, 16, 64]);  view_130 = None
    permute_80: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_131, [2, 0, 3, 1, 4]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_12 = torch.ops.aten.unbind.int(permute_80);  permute_80 = None
    getitem_134: "f32[8, 16, 65, 64]" = unbind_12[0]
    getitem_135: "f32[8, 16, 65, 64]" = unbind_12[1]
    getitem_136: "f32[8, 16, 65, 64]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_134, getitem_135, getitem_136, None, False);  getitem_134 = getitem_135 = getitem_136 = None
    getitem_137: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_81: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None
    view_132: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_81, [8, 65, 1024]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_133: "f32[520, 1024]" = torch.ops.aten.view.default(view_132, [520, 1024]);  view_132 = None
    permute_82: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_49: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg161_1, view_133, permute_82);  arg161_1 = view_133 = permute_82 = None
    view_134: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_49, [8, 65, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_37: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_134);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_89: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_86, clone_37);  add_86 = clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_25 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_141: "f32[8, 65, 1]" = var_mean_25[0]
    getitem_142: "f32[8, 65, 1]" = var_mean_25[1];  var_mean_25 = None
    add_90: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-06);  getitem_141 = None
    rsqrt_25: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_25: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_89, getitem_142);  getitem_142 = None
    mul_86: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
    mul_87: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_86, arg162_1);  mul_86 = arg162_1 = None
    add_91: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_87, arg163_1);  mul_87 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_135: "f32[520, 1024]" = torch.ops.aten.view.default(add_91, [520, 1024]);  add_91 = None
    permute_83: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_50: "f32[520, 4096]" = torch.ops.aten.addmm.default(arg165_1, view_135, permute_83);  arg165_1 = view_135 = permute_83 = None
    view_136: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_50, [8, 65, 4096]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_88: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_136, 0.5)
    mul_89: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_136, 0.7071067811865476);  view_136 = None
    erf_12: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
    add_92: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_90: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_88, add_92);  mul_88 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_38: "f32[8, 65, 4096]" = torch.ops.aten.clone.default(mul_90);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_137: "f32[520, 4096]" = torch.ops.aten.view.default(clone_38, [520, 4096]);  clone_38 = None
    permute_84: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_51: "f32[520, 1024]" = torch.ops.aten.addmm.default(arg167_1, view_137, permute_84);  arg167_1 = view_137 = permute_84 = None
    view_138: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_51, [8, 65, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_39: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_138);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_93: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_89, clone_39);  add_89 = clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    slice_9: "f32[8, 65, 1024]" = torch.ops.aten.slice.Tensor(add_93, 0, 0, 9223372036854775807);  add_93 = None
    slice_10: "f32[8, 1, 1024]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 1);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:260, code: cls_tokens = self.norm(cls_tokens)
    clone_40: "f32[8, 1, 1024]" = torch.ops.aten.clone.default(slice_10, memory_format = torch.contiguous_format);  slice_10 = None
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
    getitem_143: "f32[8, 1, 1]" = var_mean_26[0]
    getitem_144: "f32[8, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_94: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-06);  getitem_143 = None
    rsqrt_26: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_26: "f32[8, 1, 1024]" = torch.ops.aten.sub.Tensor(clone_40, getitem_144);  clone_40 = getitem_144 = None
    mul_91: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
    mul_92: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_91, arg168_1);  mul_91 = arg168_1 = None
    add_95: "f32[8, 1, 1024]" = torch.ops.aten.add.Tensor(mul_92, arg169_1);  mul_92 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:280, code: x = x[:, 0]
    slice_13: "f32[8, 1, 1024]" = torch.ops.aten.slice.Tensor(add_95, 0, 0, 9223372036854775807);  add_95 = None
    select: "f32[8, 1024]" = torch.ops.aten.select.int(slice_13, 1, 0);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:281, code: x = self.head_drop(x)
    clone_41: "f32[8, 1024]" = torch.ops.aten.clone.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:283, code: x = self.head(x)
    permute_86: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_52: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg171_1, clone_41, permute_86);  arg171_1 = clone_41 = permute_86 = None
    return (addmm_52,)
    