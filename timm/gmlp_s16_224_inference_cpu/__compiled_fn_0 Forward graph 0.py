from __future__ import annotations



def forward(self, arg0_1: "f32[256, 3, 16, 16]", arg1_1: "f32[256]", arg2_1: "f32[256]", arg3_1: "f32[256]", arg4_1: "f32[1536, 256]", arg5_1: "f32[1536]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[196, 196]", arg9_1: "f32[196]", arg10_1: "f32[256, 768]", arg11_1: "f32[256]", arg12_1: "f32[256]", arg13_1: "f32[256]", arg14_1: "f32[1536, 256]", arg15_1: "f32[1536]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[196, 196]", arg19_1: "f32[196]", arg20_1: "f32[256, 768]", arg21_1: "f32[256]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[1536, 256]", arg25_1: "f32[1536]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[196, 196]", arg29_1: "f32[196]", arg30_1: "f32[256, 768]", arg31_1: "f32[256]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[1536, 256]", arg35_1: "f32[1536]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[196, 196]", arg39_1: "f32[196]", arg40_1: "f32[256, 768]", arg41_1: "f32[256]", arg42_1: "f32[256]", arg43_1: "f32[256]", arg44_1: "f32[1536, 256]", arg45_1: "f32[1536]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[196, 196]", arg49_1: "f32[196]", arg50_1: "f32[256, 768]", arg51_1: "f32[256]", arg52_1: "f32[256]", arg53_1: "f32[256]", arg54_1: "f32[1536, 256]", arg55_1: "f32[1536]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[196, 196]", arg59_1: "f32[196]", arg60_1: "f32[256, 768]", arg61_1: "f32[256]", arg62_1: "f32[256]", arg63_1: "f32[256]", arg64_1: "f32[1536, 256]", arg65_1: "f32[1536]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[196, 196]", arg69_1: "f32[196]", arg70_1: "f32[256, 768]", arg71_1: "f32[256]", arg72_1: "f32[256]", arg73_1: "f32[256]", arg74_1: "f32[1536, 256]", arg75_1: "f32[1536]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[196, 196]", arg79_1: "f32[196]", arg80_1: "f32[256, 768]", arg81_1: "f32[256]", arg82_1: "f32[256]", arg83_1: "f32[256]", arg84_1: "f32[1536, 256]", arg85_1: "f32[1536]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[196, 196]", arg89_1: "f32[196]", arg90_1: "f32[256, 768]", arg91_1: "f32[256]", arg92_1: "f32[256]", arg93_1: "f32[256]", arg94_1: "f32[1536, 256]", arg95_1: "f32[1536]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[196, 196]", arg99_1: "f32[196]", arg100_1: "f32[256, 768]", arg101_1: "f32[256]", arg102_1: "f32[256]", arg103_1: "f32[256]", arg104_1: "f32[1536, 256]", arg105_1: "f32[1536]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[196, 196]", arg109_1: "f32[196]", arg110_1: "f32[256, 768]", arg111_1: "f32[256]", arg112_1: "f32[256]", arg113_1: "f32[256]", arg114_1: "f32[1536, 256]", arg115_1: "f32[1536]", arg116_1: "f32[768]", arg117_1: "f32[768]", arg118_1: "f32[196, 196]", arg119_1: "f32[196]", arg120_1: "f32[256, 768]", arg121_1: "f32[256]", arg122_1: "f32[256]", arg123_1: "f32[256]", arg124_1: "f32[1536, 256]", arg125_1: "f32[1536]", arg126_1: "f32[768]", arg127_1: "f32[768]", arg128_1: "f32[196, 196]", arg129_1: "f32[196]", arg130_1: "f32[256, 768]", arg131_1: "f32[256]", arg132_1: "f32[256]", arg133_1: "f32[256]", arg134_1: "f32[1536, 256]", arg135_1: "f32[1536]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[196, 196]", arg139_1: "f32[196]", arg140_1: "f32[256, 768]", arg141_1: "f32[256]", arg142_1: "f32[256]", arg143_1: "f32[256]", arg144_1: "f32[1536, 256]", arg145_1: "f32[1536]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[196, 196]", arg149_1: "f32[196]", arg150_1: "f32[256, 768]", arg151_1: "f32[256]", arg152_1: "f32[256]", arg153_1: "f32[256]", arg154_1: "f32[1536, 256]", arg155_1: "f32[1536]", arg156_1: "f32[768]", arg157_1: "f32[768]", arg158_1: "f32[196, 196]", arg159_1: "f32[196]", arg160_1: "f32[256, 768]", arg161_1: "f32[256]", arg162_1: "f32[256]", arg163_1: "f32[256]", arg164_1: "f32[1536, 256]", arg165_1: "f32[1536]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[196, 196]", arg169_1: "f32[196]", arg170_1: "f32[256, 768]", arg171_1: "f32[256]", arg172_1: "f32[256]", arg173_1: "f32[256]", arg174_1: "f32[1536, 256]", arg175_1: "f32[1536]", arg176_1: "f32[768]", arg177_1: "f32[768]", arg178_1: "f32[196, 196]", arg179_1: "f32[196]", arg180_1: "f32[256, 768]", arg181_1: "f32[256]", arg182_1: "f32[256]", arg183_1: "f32[256]", arg184_1: "f32[1536, 256]", arg185_1: "f32[1536]", arg186_1: "f32[768]", arg187_1: "f32[768]", arg188_1: "f32[196, 196]", arg189_1: "f32[196]", arg190_1: "f32[256, 768]", arg191_1: "f32[256]", arg192_1: "f32[256]", arg193_1: "f32[256]", arg194_1: "f32[1536, 256]", arg195_1: "f32[1536]", arg196_1: "f32[768]", arg197_1: "f32[768]", arg198_1: "f32[196, 196]", arg199_1: "f32[196]", arg200_1: "f32[256, 768]", arg201_1: "f32[256]", arg202_1: "f32[256]", arg203_1: "f32[256]", arg204_1: "f32[1536, 256]", arg205_1: "f32[1536]", arg206_1: "f32[768]", arg207_1: "f32[768]", arg208_1: "f32[196, 196]", arg209_1: "f32[196]", arg210_1: "f32[256, 768]", arg211_1: "f32[256]", arg212_1: "f32[256]", arg213_1: "f32[256]", arg214_1: "f32[1536, 256]", arg215_1: "f32[1536]", arg216_1: "f32[768]", arg217_1: "f32[768]", arg218_1: "f32[196, 196]", arg219_1: "f32[196]", arg220_1: "f32[256, 768]", arg221_1: "f32[256]", arg222_1: "f32[256]", arg223_1: "f32[256]", arg224_1: "f32[1536, 256]", arg225_1: "f32[1536]", arg226_1: "f32[768]", arg227_1: "f32[768]", arg228_1: "f32[196, 196]", arg229_1: "f32[196]", arg230_1: "f32[256, 768]", arg231_1: "f32[256]", arg232_1: "f32[256]", arg233_1: "f32[256]", arg234_1: "f32[1536, 256]", arg235_1: "f32[1536]", arg236_1: "f32[768]", arg237_1: "f32[768]", arg238_1: "f32[196, 196]", arg239_1: "f32[196]", arg240_1: "f32[256, 768]", arg241_1: "f32[256]", arg242_1: "f32[256]", arg243_1: "f32[256]", arg244_1: "f32[1536, 256]", arg245_1: "f32[1536]", arg246_1: "f32[768]", arg247_1: "f32[768]", arg248_1: "f32[196, 196]", arg249_1: "f32[196]", arg250_1: "f32[256, 768]", arg251_1: "f32[256]", arg252_1: "f32[256]", arg253_1: "f32[256]", arg254_1: "f32[1536, 256]", arg255_1: "f32[1536]", arg256_1: "f32[768]", arg257_1: "f32[768]", arg258_1: "f32[196, 196]", arg259_1: "f32[196]", arg260_1: "f32[256, 768]", arg261_1: "f32[256]", arg262_1: "f32[256]", arg263_1: "f32[256]", arg264_1: "f32[1536, 256]", arg265_1: "f32[1536]", arg266_1: "f32[768]", arg267_1: "f32[768]", arg268_1: "f32[196, 196]", arg269_1: "f32[196]", arg270_1: "f32[256, 768]", arg271_1: "f32[256]", arg272_1: "f32[256]", arg273_1: "f32[256]", arg274_1: "f32[1536, 256]", arg275_1: "f32[1536]", arg276_1: "f32[768]", arg277_1: "f32[768]", arg278_1: "f32[196, 196]", arg279_1: "f32[196]", arg280_1: "f32[256, 768]", arg281_1: "f32[256]", arg282_1: "f32[256]", arg283_1: "f32[256]", arg284_1: "f32[1536, 256]", arg285_1: "f32[1536]", arg286_1: "f32[768]", arg287_1: "f32[768]", arg288_1: "f32[196, 196]", arg289_1: "f32[196]", arg290_1: "f32[256, 768]", arg291_1: "f32[256]", arg292_1: "f32[256]", arg293_1: "f32[256]", arg294_1: "f32[1536, 256]", arg295_1: "f32[1536]", arg296_1: "f32[768]", arg297_1: "f32[768]", arg298_1: "f32[196, 196]", arg299_1: "f32[196]", arg300_1: "f32[256, 768]", arg301_1: "f32[256]", arg302_1: "f32[256]", arg303_1: "f32[256]", arg304_1: "f32[1000, 256]", arg305_1: "f32[1000]", arg306_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(arg306_1, arg0_1, arg1_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg306_1 = arg0_1 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 256, 196]" = torch.ops.aten.view.default(convolution, [8, 256, 196]);  convolution = None
    permute: "f32[8, 196, 256]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    clone: "f32[8, 196, 256]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_1: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_1: "f32[1568, 256]" = torch.ops.aten.view.default(add_1, [1568, 256]);  add_1 = None
    permute_1: "f32[256, 1536]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg5_1, view_1, permute_1);  arg5_1 = view_1 = permute_1 = None
    view_2: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm, [8, 196, 1536]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_2: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, 0.5)
    mul_3: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, 0.7071067811865476);  view_2 = None
    erf: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
    add_2: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_4: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_2, add_2);  mul_2 = add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_1: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_4);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split = torch.ops.aten.split.Tensor(clone_1, 768, -1);  clone_1 = None
    getitem_2: "f32[8, 196, 768]" = split[0]
    getitem_3: "f32[8, 196, 768]" = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_2: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_3, memory_format = torch.contiguous_format);  getitem_3 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_2, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_1: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_2, getitem_5);  clone_2 = getitem_5 = None
    mul_5: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg6_1);  mul_5 = arg6_1 = None
    add_4: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_6, arg7_1);  mul_6 = arg7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_2: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_4, [0, 2, 1]);  add_4 = None
    permute_3: "f32[196, 196]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    clone_3: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    view_3: "f32[6144, 196]" = torch.ops.aten.view.default(clone_3, [6144, 196]);  clone_3 = None
    mm: "f32[6144, 196]" = torch.ops.aten.mm.default(view_3, permute_3);  view_3 = permute_3 = None
    view_4: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm, [8, 768, 196]);  mm = None
    add_5: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_4, arg9_1);  view_4 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_4: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_5, [0, 2, 1]);  add_5 = None
    mul_7: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_2, permute_4);  getitem_2 = permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_5: "f32[1568, 768]" = torch.ops.aten.view.default(mul_7, [1568, 768]);  mul_7 = None
    permute_5: "f32[768, 256]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm_1: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg11_1, view_5, permute_5);  arg11_1 = view_5 = permute_5 = None
    view_6: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_1, [8, 196, 256]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_4: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_6);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_6: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(permute, clone_4);  permute = clone_4 = None
    clone_5: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_5, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_7: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_5, getitem_7);  clone_5 = getitem_7 = None
    mul_8: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_9: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_8, arg12_1);  mul_8 = arg12_1 = None
    add_8: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_9, arg13_1);  mul_9 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_7: "f32[1568, 256]" = torch.ops.aten.view.default(add_8, [1568, 256]);  add_8 = None
    permute_6: "f32[256, 1536]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    addmm_2: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg15_1, view_7, permute_6);  arg15_1 = view_7 = permute_6 = None
    view_8: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_2, [8, 196, 1536]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_10: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.5)
    mul_11: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476);  view_8 = None
    erf_1: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_9: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_12: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_10, add_9);  mul_10 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_6: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_1 = torch.ops.aten.split.Tensor(clone_6, 768, -1);  clone_6 = None
    getitem_8: "f32[8, 196, 768]" = split_1[0]
    getitem_9: "f32[8, 196, 768]" = split_1[1];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_7: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_9, memory_format = torch.contiguous_format);  getitem_9 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_7, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_11: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_7, getitem_11);  clone_7 = getitem_11 = None
    mul_13: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    mul_14: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_13, arg16_1);  mul_13 = arg16_1 = None
    add_11: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_14, arg17_1);  mul_14 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_7: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_11, [0, 2, 1]);  add_11 = None
    permute_8: "f32[196, 196]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    clone_8: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_9: "f32[6144, 196]" = torch.ops.aten.view.default(clone_8, [6144, 196]);  clone_8 = None
    mm_1: "f32[6144, 196]" = torch.ops.aten.mm.default(view_9, permute_8);  view_9 = permute_8 = None
    view_10: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_1, [8, 768, 196]);  mm_1 = None
    add_12: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_10, arg19_1);  view_10 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_9: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_12, [0, 2, 1]);  add_12 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_8, permute_9);  getitem_8 = permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_11: "f32[1568, 768]" = torch.ops.aten.view.default(mul_15, [1568, 768]);  mul_15 = None
    permute_10: "f32[768, 256]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_3: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg21_1, view_11, permute_10);  arg21_1 = view_11 = permute_10 = None
    view_12: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_3, [8, 196, 256]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_9: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_13: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_6, clone_9);  add_6 = clone_9 = None
    clone_10: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_10, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_10, getitem_13);  clone_10 = getitem_13 = None
    mul_16: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    mul_17: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_16, arg22_1);  mul_16 = arg22_1 = None
    add_15: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_17, arg23_1);  mul_17 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_13: "f32[1568, 256]" = torch.ops.aten.view.default(add_15, [1568, 256]);  add_15 = None
    permute_11: "f32[256, 1536]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_4: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg25_1, view_13, permute_11);  arg25_1 = view_13 = permute_11 = None
    view_14: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_4, [8, 196, 1536]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_18: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
    mul_19: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476);  view_14 = None
    erf_2: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_16: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_18, add_16);  mul_18 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_11: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_2 = torch.ops.aten.split.Tensor(clone_11, 768, -1);  clone_11 = None
    getitem_14: "f32[8, 196, 768]" = split_2[0]
    getitem_15: "f32[8, 196, 768]" = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_12: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_15, memory_format = torch.contiguous_format);  getitem_15 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_17: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_5: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_12, getitem_17);  clone_12 = getitem_17 = None
    mul_21: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    mul_22: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg26_1);  mul_21 = arg26_1 = None
    add_18: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_22, arg27_1);  mul_22 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_12: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_18, [0, 2, 1]);  add_18 = None
    permute_13: "f32[196, 196]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
    clone_13: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_15: "f32[6144, 196]" = torch.ops.aten.view.default(clone_13, [6144, 196]);  clone_13 = None
    mm_2: "f32[6144, 196]" = torch.ops.aten.mm.default(view_15, permute_13);  view_15 = permute_13 = None
    view_16: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_2, [8, 768, 196]);  mm_2 = None
    add_19: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_16, arg29_1);  view_16 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_14: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_19, [0, 2, 1]);  add_19 = None
    mul_23: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_14, permute_14);  getitem_14 = permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_17: "f32[1568, 768]" = torch.ops.aten.view.default(mul_23, [1568, 768]);  mul_23 = None
    permute_15: "f32[768, 256]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_5: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg31_1, view_17, permute_15);  arg31_1 = view_17 = permute_15 = None
    view_18: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_5, [8, 196, 256]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_14: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_20: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_13, clone_14);  add_13 = clone_14 = None
    clone_15: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_19: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_6: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_15, getitem_19);  clone_15 = getitem_19 = None
    mul_24: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    mul_25: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_24, arg32_1);  mul_24 = arg32_1 = None
    add_22: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_25, arg33_1);  mul_25 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_19: "f32[1568, 256]" = torch.ops.aten.view.default(add_22, [1568, 256]);  add_22 = None
    permute_16: "f32[256, 1536]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
    addmm_6: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg35_1, view_19, permute_16);  arg35_1 = view_19 = permute_16 = None
    view_20: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_6, [8, 196, 1536]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_26: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, 0.5)
    mul_27: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476);  view_20 = None
    erf_3: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_23: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_26, add_23);  mul_26 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_16: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_3 = torch.ops.aten.split.Tensor(clone_16, 768, -1);  clone_16 = None
    getitem_20: "f32[8, 196, 768]" = split_3[0]
    getitem_21: "f32[8, 196, 768]" = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_17: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_21, memory_format = torch.contiguous_format);  getitem_21 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_17, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_23: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_7: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_17, getitem_23);  clone_17 = getitem_23 = None
    mul_29: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_29, arg36_1);  mul_29 = arg36_1 = None
    add_25: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_30, arg37_1);  mul_30 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_17: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_25, [0, 2, 1]);  add_25 = None
    permute_18: "f32[196, 196]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    clone_18: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_21: "f32[6144, 196]" = torch.ops.aten.view.default(clone_18, [6144, 196]);  clone_18 = None
    mm_3: "f32[6144, 196]" = torch.ops.aten.mm.default(view_21, permute_18);  view_21 = permute_18 = None
    view_22: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_3, [8, 768, 196]);  mm_3 = None
    add_26: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_22, arg39_1);  view_22 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_19: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_26, [0, 2, 1]);  add_26 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_20, permute_19);  getitem_20 = permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_23: "f32[1568, 768]" = torch.ops.aten.view.default(mul_31, [1568, 768]);  mul_31 = None
    permute_20: "f32[768, 256]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    addmm_7: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg41_1, view_23, permute_20);  arg41_1 = view_23 = permute_20 = None
    view_24: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_7, [8, 196, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_19: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_27: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_20, clone_19);  add_20 = clone_19 = None
    clone_20: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_20, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_20, getitem_25);  clone_20 = getitem_25 = None
    mul_32: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    mul_33: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_32, arg42_1);  mul_32 = arg42_1 = None
    add_29: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_33, arg43_1);  mul_33 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_25: "f32[1568, 256]" = torch.ops.aten.view.default(add_29, [1568, 256]);  add_29 = None
    permute_21: "f32[256, 1536]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    addmm_8: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg45_1, view_25, permute_21);  arg45_1 = view_25 = permute_21 = None
    view_26: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_8, [8, 196, 1536]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_34: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, 0.5)
    mul_35: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476);  view_26 = None
    erf_4: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_30: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_36: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_34, add_30);  mul_34 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_21: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_36);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_4 = torch.ops.aten.split.Tensor(clone_21, 768, -1);  clone_21 = None
    getitem_26: "f32[8, 196, 768]" = split_4[0]
    getitem_27: "f32[8, 196, 768]" = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_22: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_27, memory_format = torch.contiguous_format);  getitem_27 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_22, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_31: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_9: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_22, getitem_29);  clone_22 = getitem_29 = None
    mul_37: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    mul_38: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_37, arg46_1);  mul_37 = arg46_1 = None
    add_32: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_38, arg47_1);  mul_38 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_22: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_32, [0, 2, 1]);  add_32 = None
    permute_23: "f32[196, 196]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    clone_23: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
    view_27: "f32[6144, 196]" = torch.ops.aten.view.default(clone_23, [6144, 196]);  clone_23 = None
    mm_4: "f32[6144, 196]" = torch.ops.aten.mm.default(view_27, permute_23);  view_27 = permute_23 = None
    view_28: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_4, [8, 768, 196]);  mm_4 = None
    add_33: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_28, arg49_1);  view_28 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_24: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_33, [0, 2, 1]);  add_33 = None
    mul_39: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_26, permute_24);  getitem_26 = permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_29: "f32[1568, 768]" = torch.ops.aten.view.default(mul_39, [1568, 768]);  mul_39 = None
    permute_25: "f32[768, 256]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    addmm_9: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg51_1, view_29, permute_25);  arg51_1 = view_29 = permute_25 = None
    view_30: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_9, [8, 196, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_24: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_30);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_34: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_27, clone_24);  add_27 = clone_24 = None
    clone_25: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_31: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_10: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_25, getitem_31);  clone_25 = getitem_31 = None
    mul_40: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    mul_41: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_40, arg52_1);  mul_40 = arg52_1 = None
    add_36: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_41, arg53_1);  mul_41 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_31: "f32[1568, 256]" = torch.ops.aten.view.default(add_36, [1568, 256]);  add_36 = None
    permute_26: "f32[256, 1536]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_10: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg55_1, view_31, permute_26);  arg55_1 = view_31 = permute_26 = None
    view_32: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_10, [8, 196, 1536]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_42: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, 0.5)
    mul_43: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, 0.7071067811865476);  view_32 = None
    erf_5: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_37: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_44: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_42, add_37);  mul_42 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_26: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_5 = torch.ops.aten.split.Tensor(clone_26, 768, -1);  clone_26 = None
    getitem_32: "f32[8, 196, 768]" = split_5[0]
    getitem_33: "f32[8, 196, 768]" = split_5[1];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_27: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_33, memory_format = torch.contiguous_format);  getitem_33 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_27, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_35: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_11: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_27, getitem_35);  clone_27 = getitem_35 = None
    mul_45: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg56_1);  mul_45 = arg56_1 = None
    add_39: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_46, arg57_1);  mul_46 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_27: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_39, [0, 2, 1]);  add_39 = None
    permute_28: "f32[196, 196]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    clone_28: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    view_33: "f32[6144, 196]" = torch.ops.aten.view.default(clone_28, [6144, 196]);  clone_28 = None
    mm_5: "f32[6144, 196]" = torch.ops.aten.mm.default(view_33, permute_28);  view_33 = permute_28 = None
    view_34: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_5, [8, 768, 196]);  mm_5 = None
    add_40: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_34, arg59_1);  view_34 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_29: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    mul_47: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_32, permute_29);  getitem_32 = permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_35: "f32[1568, 768]" = torch.ops.aten.view.default(mul_47, [1568, 768]);  mul_47 = None
    permute_30: "f32[768, 256]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    addmm_11: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg61_1, view_35, permute_30);  arg61_1 = view_35 = permute_30 = None
    view_36: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_11, [8, 196, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_29: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_41: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_34, clone_29);  add_34 = clone_29 = None
    clone_30: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_30, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_30, getitem_37);  clone_30 = getitem_37 = None
    mul_48: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    mul_49: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_48, arg62_1);  mul_48 = arg62_1 = None
    add_43: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_49, arg63_1);  mul_49 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_37: "f32[1568, 256]" = torch.ops.aten.view.default(add_43, [1568, 256]);  add_43 = None
    permute_31: "f32[256, 1536]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_12: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg65_1, view_37, permute_31);  arg65_1 = view_37 = permute_31 = None
    view_38: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_12, [8, 196, 1536]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_50: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_51: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_6: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_44: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_52: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_50, add_44);  mul_50 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_31: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_6 = torch.ops.aten.split.Tensor(clone_31, 768, -1);  clone_31 = None
    getitem_38: "f32[8, 196, 768]" = split_6[0]
    getitem_39: "f32[8, 196, 768]" = split_6[1];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_32: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_39, memory_format = torch.contiguous_format);  getitem_39 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_41: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_13: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_32, getitem_41);  clone_32 = getitem_41 = None
    mul_53: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    mul_54: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_53, arg66_1);  mul_53 = arg66_1 = None
    add_46: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_54, arg67_1);  mul_54 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_32: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_46, [0, 2, 1]);  add_46 = None
    permute_33: "f32[196, 196]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    clone_33: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_39: "f32[6144, 196]" = torch.ops.aten.view.default(clone_33, [6144, 196]);  clone_33 = None
    mm_6: "f32[6144, 196]" = torch.ops.aten.mm.default(view_39, permute_33);  view_39 = permute_33 = None
    view_40: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_6, [8, 768, 196]);  mm_6 = None
    add_47: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_40, arg69_1);  view_40 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_34: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_47, [0, 2, 1]);  add_47 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_38, permute_34);  getitem_38 = permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_41: "f32[1568, 768]" = torch.ops.aten.view.default(mul_55, [1568, 768]);  mul_55 = None
    permute_35: "f32[768, 256]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_13: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg71_1, view_41, permute_35);  arg71_1 = view_41 = permute_35 = None
    view_42: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_13, [8, 196, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_34: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_42);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_48: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_41, clone_34);  add_41 = clone_34 = None
    clone_35: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_35, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_43: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_14: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_35, getitem_43);  clone_35 = getitem_43 = None
    mul_56: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    mul_57: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_56, arg72_1);  mul_56 = arg72_1 = None
    add_50: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_57, arg73_1);  mul_57 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_43: "f32[1568, 256]" = torch.ops.aten.view.default(add_50, [1568, 256]);  add_50 = None
    permute_36: "f32[256, 1536]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_14: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg75_1, view_43, permute_36);  arg75_1 = view_43 = permute_36 = None
    view_44: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_14, [8, 196, 1536]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_58: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.5)
    mul_59: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.7071067811865476);  view_44 = None
    erf_7: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_59);  mul_59 = None
    add_51: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_60: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_58, add_51);  mul_58 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_36: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_60);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_7 = torch.ops.aten.split.Tensor(clone_36, 768, -1);  clone_36 = None
    getitem_44: "f32[8, 196, 768]" = split_7[0]
    getitem_45: "f32[8, 196, 768]" = split_7[1];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_37: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_45, memory_format = torch.contiguous_format);  getitem_45 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_47: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_15: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_37, getitem_47);  clone_37 = getitem_47 = None
    mul_61: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    mul_62: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_61, arg76_1);  mul_61 = arg76_1 = None
    add_53: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_62, arg77_1);  mul_62 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_37: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_53, [0, 2, 1]);  add_53 = None
    permute_38: "f32[196, 196]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    clone_38: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_45: "f32[6144, 196]" = torch.ops.aten.view.default(clone_38, [6144, 196]);  clone_38 = None
    mm_7: "f32[6144, 196]" = torch.ops.aten.mm.default(view_45, permute_38);  view_45 = permute_38 = None
    view_46: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_7, [8, 768, 196]);  mm_7 = None
    add_54: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_46, arg79_1);  view_46 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_39: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_54, [0, 2, 1]);  add_54 = None
    mul_63: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_44, permute_39);  getitem_44 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_47: "f32[1568, 768]" = torch.ops.aten.view.default(mul_63, [1568, 768]);  mul_63 = None
    permute_40: "f32[768, 256]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_15: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg81_1, view_47, permute_40);  arg81_1 = view_47 = permute_40 = None
    view_48: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_15, [8, 196, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_39: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_55: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_48, clone_39);  add_48 = clone_39 = None
    clone_40: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_49: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_16: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_40, getitem_49);  clone_40 = getitem_49 = None
    mul_64: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    mul_65: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_64, arg82_1);  mul_64 = arg82_1 = None
    add_57: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_65, arg83_1);  mul_65 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_49: "f32[1568, 256]" = torch.ops.aten.view.default(add_57, [1568, 256]);  add_57 = None
    permute_41: "f32[256, 1536]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_16: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg85_1, view_49, permute_41);  arg85_1 = view_49 = permute_41 = None
    view_50: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 196, 1536]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_66: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
    mul_67: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476);  view_50 = None
    erf_8: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_58: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_68: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_66, add_58);  mul_66 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_41: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_8 = torch.ops.aten.split.Tensor(clone_41, 768, -1);  clone_41 = None
    getitem_50: "f32[8, 196, 768]" = split_8[0]
    getitem_51: "f32[8, 196, 768]" = split_8[1];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_42: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_51, memory_format = torch.contiguous_format);  getitem_51 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_53: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_17: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_42, getitem_53);  clone_42 = getitem_53 = None
    mul_69: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_69, arg86_1);  mul_69 = arg86_1 = None
    add_60: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_70, arg87_1);  mul_70 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_42: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_60, [0, 2, 1]);  add_60 = None
    permute_43: "f32[196, 196]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    clone_43: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    view_51: "f32[6144, 196]" = torch.ops.aten.view.default(clone_43, [6144, 196]);  clone_43 = None
    mm_8: "f32[6144, 196]" = torch.ops.aten.mm.default(view_51, permute_43);  view_51 = permute_43 = None
    view_52: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_8, [8, 768, 196]);  mm_8 = None
    add_61: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_52, arg89_1);  view_52 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_44: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_61, [0, 2, 1]);  add_61 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_50, permute_44);  getitem_50 = permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_53: "f32[1568, 768]" = torch.ops.aten.view.default(mul_71, [1568, 768]);  mul_71 = None
    permute_45: "f32[768, 256]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_17: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg91_1, view_53, permute_45);  arg91_1 = view_53 = permute_45 = None
    view_54: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_17, [8, 196, 256]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_44: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_62: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_55, clone_44);  add_55 = clone_44 = None
    clone_45: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_45, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_55: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_18: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_45, getitem_55);  clone_45 = getitem_55 = None
    mul_72: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    mul_73: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_72, arg92_1);  mul_72 = arg92_1 = None
    add_64: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_73, arg93_1);  mul_73 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_55: "f32[1568, 256]" = torch.ops.aten.view.default(add_64, [1568, 256]);  add_64 = None
    permute_46: "f32[256, 1536]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_18: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg95_1, view_55, permute_46);  arg95_1 = view_55 = permute_46 = None
    view_56: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_18, [8, 196, 1536]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_74: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, 0.5)
    mul_75: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476);  view_56 = None
    erf_9: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_65: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_76: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_74, add_65);  mul_74 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_46: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_9 = torch.ops.aten.split.Tensor(clone_46, 768, -1);  clone_46 = None
    getitem_56: "f32[8, 196, 768]" = split_9[0]
    getitem_57: "f32[8, 196, 768]" = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_47: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_57, memory_format = torch.contiguous_format);  getitem_57 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_47, [2], correction = 0, keepdim = True)
    getitem_58: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_59: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_19: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_47, getitem_59);  clone_47 = getitem_59 = None
    mul_77: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    mul_78: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_77, arg96_1);  mul_77 = arg96_1 = None
    add_67: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_78, arg97_1);  mul_78 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_47: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_67, [0, 2, 1]);  add_67 = None
    permute_48: "f32[196, 196]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    clone_48: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    view_57: "f32[6144, 196]" = torch.ops.aten.view.default(clone_48, [6144, 196]);  clone_48 = None
    mm_9: "f32[6144, 196]" = torch.ops.aten.mm.default(view_57, permute_48);  view_57 = permute_48 = None
    view_58: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_9, [8, 768, 196]);  mm_9 = None
    add_68: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_58, arg99_1);  view_58 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_49: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_68, [0, 2, 1]);  add_68 = None
    mul_79: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_56, permute_49);  getitem_56 = permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_59: "f32[1568, 768]" = torch.ops.aten.view.default(mul_79, [1568, 768]);  mul_79 = None
    permute_50: "f32[768, 256]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_19: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg101_1, view_59, permute_50);  arg101_1 = view_59 = permute_50 = None
    view_60: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_19, [8, 196, 256]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_49: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_69: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_62, clone_49);  add_62 = clone_49 = None
    clone_50: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_50, [2], correction = 0, keepdim = True)
    getitem_60: "f32[8, 196, 1]" = var_mean_20[0]
    getitem_61: "f32[8, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_20: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_20: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_50, getitem_61);  clone_50 = getitem_61 = None
    mul_80: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    mul_81: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_80, arg102_1);  mul_80 = arg102_1 = None
    add_71: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_81, arg103_1);  mul_81 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_61: "f32[1568, 256]" = torch.ops.aten.view.default(add_71, [1568, 256]);  add_71 = None
    permute_51: "f32[256, 1536]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_20: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg105_1, view_61, permute_51);  arg105_1 = view_61 = permute_51 = None
    view_62: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_20, [8, 196, 1536]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_82: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, 0.5)
    mul_83: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476);  view_62 = None
    erf_10: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_72: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_84: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_82, add_72);  mul_82 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_51: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_10 = torch.ops.aten.split.Tensor(clone_51, 768, -1);  clone_51 = None
    getitem_62: "f32[8, 196, 768]" = split_10[0]
    getitem_63: "f32[8, 196, 768]" = split_10[1];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_52: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_63, memory_format = torch.contiguous_format);  getitem_63 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_52, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 196, 1]" = var_mean_21[0]
    getitem_65: "f32[8, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_21: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_21: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_52, getitem_65);  clone_52 = getitem_65 = None
    mul_85: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg106_1);  mul_85 = arg106_1 = None
    add_74: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_86, arg107_1);  mul_86 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_52: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_74, [0, 2, 1]);  add_74 = None
    permute_53: "f32[196, 196]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    clone_53: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_63: "f32[6144, 196]" = torch.ops.aten.view.default(clone_53, [6144, 196]);  clone_53 = None
    mm_10: "f32[6144, 196]" = torch.ops.aten.mm.default(view_63, permute_53);  view_63 = permute_53 = None
    view_64: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_10, [8, 768, 196]);  mm_10 = None
    add_75: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_64, arg109_1);  view_64 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_54: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_75, [0, 2, 1]);  add_75 = None
    mul_87: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_62, permute_54);  getitem_62 = permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_65: "f32[1568, 768]" = torch.ops.aten.view.default(mul_87, [1568, 768]);  mul_87 = None
    permute_55: "f32[768, 256]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_21: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg111_1, view_65, permute_55);  arg111_1 = view_65 = permute_55 = None
    view_66: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_21, [8, 196, 256]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_54: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_66);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_76: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_69, clone_54);  add_69 = clone_54 = None
    clone_55: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_55, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 196, 1]" = var_mean_22[0]
    getitem_67: "f32[8, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_22: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_22: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_55, getitem_67);  clone_55 = getitem_67 = None
    mul_88: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    mul_89: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_88, arg112_1);  mul_88 = arg112_1 = None
    add_78: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_89, arg113_1);  mul_89 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_67: "f32[1568, 256]" = torch.ops.aten.view.default(add_78, [1568, 256]);  add_78 = None
    permute_56: "f32[256, 1536]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_22: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg115_1, view_67, permute_56);  arg115_1 = view_67 = permute_56 = None
    view_68: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_22, [8, 196, 1536]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_90: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.5)
    mul_91: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476);  view_68 = None
    erf_11: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
    add_79: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_92: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_90, add_79);  mul_90 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_56: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_11 = torch.ops.aten.split.Tensor(clone_56, 768, -1);  clone_56 = None
    getitem_68: "f32[8, 196, 768]" = split_11[0]
    getitem_69: "f32[8, 196, 768]" = split_11[1];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_57: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_69, memory_format = torch.contiguous_format);  getitem_69 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_71: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_23: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_57, getitem_71);  clone_57 = getitem_71 = None
    mul_93: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    mul_94: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_93, arg116_1);  mul_93 = arg116_1 = None
    add_81: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_94, arg117_1);  mul_94 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_57: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_81, [0, 2, 1]);  add_81 = None
    permute_58: "f32[196, 196]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    clone_58: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    view_69: "f32[6144, 196]" = torch.ops.aten.view.default(clone_58, [6144, 196]);  clone_58 = None
    mm_11: "f32[6144, 196]" = torch.ops.aten.mm.default(view_69, permute_58);  view_69 = permute_58 = None
    view_70: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_11, [8, 768, 196]);  mm_11 = None
    add_82: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_70, arg119_1);  view_70 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_59: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_82, [0, 2, 1]);  add_82 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_68, permute_59);  getitem_68 = permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_71: "f32[1568, 768]" = torch.ops.aten.view.default(mul_95, [1568, 768]);  mul_95 = None
    permute_60: "f32[768, 256]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_23: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg121_1, view_71, permute_60);  arg121_1 = view_71 = permute_60 = None
    view_72: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_23, [8, 196, 256]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_59: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_83: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_76, clone_59);  add_76 = clone_59 = None
    clone_60: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_60, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_73: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_24: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_60, getitem_73);  clone_60 = getitem_73 = None
    mul_96: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    mul_97: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_96, arg122_1);  mul_96 = arg122_1 = None
    add_85: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_97, arg123_1);  mul_97 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_73: "f32[1568, 256]" = torch.ops.aten.view.default(add_85, [1568, 256]);  add_85 = None
    permute_61: "f32[256, 1536]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_24: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg125_1, view_73, permute_61);  arg125_1 = view_73 = permute_61 = None
    view_74: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_24, [8, 196, 1536]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_98: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, 0.5)
    mul_99: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, 0.7071067811865476);  view_74 = None
    erf_12: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_86: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_100: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_98, add_86);  mul_98 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_61: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_100);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_12 = torch.ops.aten.split.Tensor(clone_61, 768, -1);  clone_61 = None
    getitem_74: "f32[8, 196, 768]" = split_12[0]
    getitem_75: "f32[8, 196, 768]" = split_12[1];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_62: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_75, memory_format = torch.contiguous_format);  getitem_75 = None
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_62, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 196, 1]" = var_mean_25[0]
    getitem_77: "f32[8, 196, 1]" = var_mean_25[1];  var_mean_25 = None
    add_87: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_25: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_25: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_62, getitem_77);  clone_62 = getitem_77 = None
    mul_101: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
    mul_102: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_101, arg126_1);  mul_101 = arg126_1 = None
    add_88: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_102, arg127_1);  mul_102 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_62: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_88, [0, 2, 1]);  add_88 = None
    permute_63: "f32[196, 196]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    clone_63: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_75: "f32[6144, 196]" = torch.ops.aten.view.default(clone_63, [6144, 196]);  clone_63 = None
    mm_12: "f32[6144, 196]" = torch.ops.aten.mm.default(view_75, permute_63);  view_75 = permute_63 = None
    view_76: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_12, [8, 768, 196]);  mm_12 = None
    add_89: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_76, arg129_1);  view_76 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_64: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_89, [0, 2, 1]);  add_89 = None
    mul_103: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_74, permute_64);  getitem_74 = permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_77: "f32[1568, 768]" = torch.ops.aten.view.default(mul_103, [1568, 768]);  mul_103 = None
    permute_65: "f32[768, 256]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_25: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg131_1, view_77, permute_65);  arg131_1 = view_77 = permute_65 = None
    view_78: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_25, [8, 196, 256]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_64: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_90: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_83, clone_64);  add_83 = clone_64 = None
    clone_65: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_65, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 196, 1]" = var_mean_26[0]
    getitem_79: "f32[8, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_26: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_26: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_65, getitem_79);  clone_65 = getitem_79 = None
    mul_104: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
    mul_105: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_104, arg132_1);  mul_104 = arg132_1 = None
    add_92: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_105, arg133_1);  mul_105 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_79: "f32[1568, 256]" = torch.ops.aten.view.default(add_92, [1568, 256]);  add_92 = None
    permute_66: "f32[256, 1536]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_26: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg135_1, view_79, permute_66);  arg135_1 = view_79 = permute_66 = None
    view_80: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_26, [8, 196, 1536]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_106: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, 0.5)
    mul_107: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476);  view_80 = None
    erf_13: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_93: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_108: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_106, add_93);  mul_106 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_66: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_108);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_13 = torch.ops.aten.split.Tensor(clone_66, 768, -1);  clone_66 = None
    getitem_80: "f32[8, 196, 768]" = split_13[0]
    getitem_81: "f32[8, 196, 768]" = split_13[1];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_67: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_81, memory_format = torch.contiguous_format);  getitem_81 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 196, 1]" = var_mean_27[0]
    getitem_83: "f32[8, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_94: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_27: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_27: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_67, getitem_83);  clone_67 = getitem_83 = None
    mul_109: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
    mul_110: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_109, arg136_1);  mul_109 = arg136_1 = None
    add_95: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_110, arg137_1);  mul_110 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_67: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_95, [0, 2, 1]);  add_95 = None
    permute_68: "f32[196, 196]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    clone_68: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_81: "f32[6144, 196]" = torch.ops.aten.view.default(clone_68, [6144, 196]);  clone_68 = None
    mm_13: "f32[6144, 196]" = torch.ops.aten.mm.default(view_81, permute_68);  view_81 = permute_68 = None
    view_82: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_13, [8, 768, 196]);  mm_13 = None
    add_96: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_82, arg139_1);  view_82 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_69: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_96, [0, 2, 1]);  add_96 = None
    mul_111: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_80, permute_69);  getitem_80 = permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_83: "f32[1568, 768]" = torch.ops.aten.view.default(mul_111, [1568, 768]);  mul_111 = None
    permute_70: "f32[768, 256]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_27: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg141_1, view_83, permute_70);  arg141_1 = view_83 = permute_70 = None
    view_84: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_27, [8, 196, 256]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_69: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_97: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_90, clone_69);  add_90 = clone_69 = None
    clone_70: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_70, [2], correction = 0, keepdim = True)
    getitem_84: "f32[8, 196, 1]" = var_mean_28[0]
    getitem_85: "f32[8, 196, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_28: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_28: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_70, getitem_85);  clone_70 = getitem_85 = None
    mul_112: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
    mul_113: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_112, arg142_1);  mul_112 = arg142_1 = None
    add_99: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_113, arg143_1);  mul_113 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_85: "f32[1568, 256]" = torch.ops.aten.view.default(add_99, [1568, 256]);  add_99 = None
    permute_71: "f32[256, 1536]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_28: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg145_1, view_85, permute_71);  arg145_1 = view_85 = permute_71 = None
    view_86: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 196, 1536]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_114: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, 0.5)
    mul_115: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476);  view_86 = None
    erf_14: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
    add_100: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_116: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_114, add_100);  mul_114 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_71: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_116);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_14 = torch.ops.aten.split.Tensor(clone_71, 768, -1);  clone_71 = None
    getitem_86: "f32[8, 196, 768]" = split_14[0]
    getitem_87: "f32[8, 196, 768]" = split_14[1];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_72: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_87, memory_format = torch.contiguous_format);  getitem_87 = None
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_72, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 196, 1]" = var_mean_29[0]
    getitem_89: "f32[8, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_101: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_29: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_29: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_72, getitem_89);  clone_72 = getitem_89 = None
    mul_117: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
    mul_118: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_117, arg146_1);  mul_117 = arg146_1 = None
    add_102: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_118, arg147_1);  mul_118 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_72: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_102, [0, 2, 1]);  add_102 = None
    permute_73: "f32[196, 196]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    clone_73: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    view_87: "f32[6144, 196]" = torch.ops.aten.view.default(clone_73, [6144, 196]);  clone_73 = None
    mm_14: "f32[6144, 196]" = torch.ops.aten.mm.default(view_87, permute_73);  view_87 = permute_73 = None
    view_88: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_14, [8, 768, 196]);  mm_14 = None
    add_103: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_88, arg149_1);  view_88 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_74: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_103, [0, 2, 1]);  add_103 = None
    mul_119: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_86, permute_74);  getitem_86 = permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_89: "f32[1568, 768]" = torch.ops.aten.view.default(mul_119, [1568, 768]);  mul_119 = None
    permute_75: "f32[768, 256]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_29: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg151_1, view_89, permute_75);  arg151_1 = view_89 = permute_75 = None
    view_90: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_29, [8, 196, 256]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_74: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_104: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_97, clone_74);  add_97 = clone_74 = None
    clone_75: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_75, [2], correction = 0, keepdim = True)
    getitem_90: "f32[8, 196, 1]" = var_mean_30[0]
    getitem_91: "f32[8, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_30: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_30: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_75, getitem_91);  clone_75 = getitem_91 = None
    mul_120: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
    mul_121: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_120, arg152_1);  mul_120 = arg152_1 = None
    add_106: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_121, arg153_1);  mul_121 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_91: "f32[1568, 256]" = torch.ops.aten.view.default(add_106, [1568, 256]);  add_106 = None
    permute_76: "f32[256, 1536]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_30: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg155_1, view_91, permute_76);  arg155_1 = view_91 = permute_76 = None
    view_92: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_30, [8, 196, 1536]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_122: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, 0.5)
    mul_123: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476);  view_92 = None
    erf_15: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_107: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_124: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_122, add_107);  mul_122 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_76: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_15 = torch.ops.aten.split.Tensor(clone_76, 768, -1);  clone_76 = None
    getitem_92: "f32[8, 196, 768]" = split_15[0]
    getitem_93: "f32[8, 196, 768]" = split_15[1];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_77: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_93, memory_format = torch.contiguous_format);  getitem_93 = None
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_77, [2], correction = 0, keepdim = True)
    getitem_94: "f32[8, 196, 1]" = var_mean_31[0]
    getitem_95: "f32[8, 196, 1]" = var_mean_31[1];  var_mean_31 = None
    add_108: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_31: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_31: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_77, getitem_95);  clone_77 = getitem_95 = None
    mul_125: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
    mul_126: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_125, arg156_1);  mul_125 = arg156_1 = None
    add_109: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_126, arg157_1);  mul_126 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_77: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_109, [0, 2, 1]);  add_109 = None
    permute_78: "f32[196, 196]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    clone_78: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_93: "f32[6144, 196]" = torch.ops.aten.view.default(clone_78, [6144, 196]);  clone_78 = None
    mm_15: "f32[6144, 196]" = torch.ops.aten.mm.default(view_93, permute_78);  view_93 = permute_78 = None
    view_94: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_15, [8, 768, 196]);  mm_15 = None
    add_110: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_94, arg159_1);  view_94 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_79: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_110, [0, 2, 1]);  add_110 = None
    mul_127: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_92, permute_79);  getitem_92 = permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_95: "f32[1568, 768]" = torch.ops.aten.view.default(mul_127, [1568, 768]);  mul_127 = None
    permute_80: "f32[768, 256]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_31: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg161_1, view_95, permute_80);  arg161_1 = view_95 = permute_80 = None
    view_96: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_31, [8, 196, 256]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_79: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_111: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_104, clone_79);  add_104 = clone_79 = None
    clone_80: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_80, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 196, 1]" = var_mean_32[0]
    getitem_97: "f32[8, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_32: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_32: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_80, getitem_97);  clone_80 = getitem_97 = None
    mul_128: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
    mul_129: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_128, arg162_1);  mul_128 = arg162_1 = None
    add_113: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_129, arg163_1);  mul_129 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_97: "f32[1568, 256]" = torch.ops.aten.view.default(add_113, [1568, 256]);  add_113 = None
    permute_81: "f32[256, 1536]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_32: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg165_1, view_97, permute_81);  arg165_1 = view_97 = permute_81 = None
    view_98: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_32, [8, 196, 1536]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_130: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_131: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_16: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_114: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_132: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_130, add_114);  mul_130 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_81: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_132);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_16 = torch.ops.aten.split.Tensor(clone_81, 768, -1);  clone_81 = None
    getitem_98: "f32[8, 196, 768]" = split_16[0]
    getitem_99: "f32[8, 196, 768]" = split_16[1];  split_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_82: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_99, memory_format = torch.contiguous_format);  getitem_99 = None
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 196, 1]" = var_mean_33[0]
    getitem_101: "f32[8, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_115: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_33: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_33: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_82, getitem_101);  clone_82 = getitem_101 = None
    mul_133: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
    mul_134: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_133, arg166_1);  mul_133 = arg166_1 = None
    add_116: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_134, arg167_1);  mul_134 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_82: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_116, [0, 2, 1]);  add_116 = None
    permute_83: "f32[196, 196]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    clone_83: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_99: "f32[6144, 196]" = torch.ops.aten.view.default(clone_83, [6144, 196]);  clone_83 = None
    mm_16: "f32[6144, 196]" = torch.ops.aten.mm.default(view_99, permute_83);  view_99 = permute_83 = None
    view_100: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_16, [8, 768, 196]);  mm_16 = None
    add_117: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_100, arg169_1);  view_100 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_84: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_117, [0, 2, 1]);  add_117 = None
    mul_135: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_98, permute_84);  getitem_98 = permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_101: "f32[1568, 768]" = torch.ops.aten.view.default(mul_135, [1568, 768]);  mul_135 = None
    permute_85: "f32[768, 256]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_33: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg171_1, view_101, permute_85);  arg171_1 = view_101 = permute_85 = None
    view_102: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_33, [8, 196, 256]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_84: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_102);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_118: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_111, clone_84);  add_111 = clone_84 = None
    clone_85: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_118, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_102: "f32[8, 196, 1]" = var_mean_34[0]
    getitem_103: "f32[8, 196, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
    rsqrt_34: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_34: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_85, getitem_103);  clone_85 = getitem_103 = None
    mul_136: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
    mul_137: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_136, arg172_1);  mul_136 = arg172_1 = None
    add_120: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_137, arg173_1);  mul_137 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_103: "f32[1568, 256]" = torch.ops.aten.view.default(add_120, [1568, 256]);  add_120 = None
    permute_86: "f32[256, 1536]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_34: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg175_1, view_103, permute_86);  arg175_1 = view_103 = permute_86 = None
    view_104: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_34, [8, 196, 1536]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_138: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, 0.5)
    mul_139: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, 0.7071067811865476);  view_104 = None
    erf_17: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_121: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_140: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_138, add_121);  mul_138 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_86: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_17 = torch.ops.aten.split.Tensor(clone_86, 768, -1);  clone_86 = None
    getitem_104: "f32[8, 196, 768]" = split_17[0]
    getitem_105: "f32[8, 196, 768]" = split_17[1];  split_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_87: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_105, memory_format = torch.contiguous_format);  getitem_105 = None
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_87, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 196, 1]" = var_mean_35[0]
    getitem_107: "f32[8, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_122: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_35: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_35: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_87, getitem_107);  clone_87 = getitem_107 = None
    mul_141: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
    mul_142: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_141, arg176_1);  mul_141 = arg176_1 = None
    add_123: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_142, arg177_1);  mul_142 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_87: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_123, [0, 2, 1]);  add_123 = None
    permute_88: "f32[196, 196]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    clone_88: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_105: "f32[6144, 196]" = torch.ops.aten.view.default(clone_88, [6144, 196]);  clone_88 = None
    mm_17: "f32[6144, 196]" = torch.ops.aten.mm.default(view_105, permute_88);  view_105 = permute_88 = None
    view_106: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_17, [8, 768, 196]);  mm_17 = None
    add_124: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_106, arg179_1);  view_106 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_89: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None
    mul_143: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_104, permute_89);  getitem_104 = permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_107: "f32[1568, 768]" = torch.ops.aten.view.default(mul_143, [1568, 768]);  mul_143 = None
    permute_90: "f32[768, 256]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_35: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg181_1, view_107, permute_90);  arg181_1 = view_107 = permute_90 = None
    view_108: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_35, [8, 196, 256]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_89: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_125: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_118, clone_89);  add_118 = clone_89 = None
    clone_90: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_90, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 196, 1]" = var_mean_36[0]
    getitem_109: "f32[8, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_36: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_36: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_90, getitem_109);  clone_90 = getitem_109 = None
    mul_144: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
    mul_145: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_144, arg182_1);  mul_144 = arg182_1 = None
    add_127: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_145, arg183_1);  mul_145 = arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_109: "f32[1568, 256]" = torch.ops.aten.view.default(add_127, [1568, 256]);  add_127 = None
    permute_91: "f32[256, 1536]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    addmm_36: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg185_1, view_109, permute_91);  arg185_1 = view_109 = permute_91 = None
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_36, [8, 196, 1536]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_146: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, 0.5)
    mul_147: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, 0.7071067811865476);  view_110 = None
    erf_18: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_128: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_148: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_146, add_128);  mul_146 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_91: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_18 = torch.ops.aten.split.Tensor(clone_91, 768, -1);  clone_91 = None
    getitem_110: "f32[8, 196, 768]" = split_18[0]
    getitem_111: "f32[8, 196, 768]" = split_18[1];  split_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_92: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_111, memory_format = torch.contiguous_format);  getitem_111 = None
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_92, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 196, 1]" = var_mean_37[0]
    getitem_113: "f32[8, 196, 1]" = var_mean_37[1];  var_mean_37 = None
    add_129: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_37: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_37: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_92, getitem_113);  clone_92 = getitem_113 = None
    mul_149: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
    mul_150: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_149, arg186_1);  mul_149 = arg186_1 = None
    add_130: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_150, arg187_1);  mul_150 = arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_92: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_130, [0, 2, 1]);  add_130 = None
    permute_93: "f32[196, 196]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    clone_93: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    view_111: "f32[6144, 196]" = torch.ops.aten.view.default(clone_93, [6144, 196]);  clone_93 = None
    mm_18: "f32[6144, 196]" = torch.ops.aten.mm.default(view_111, permute_93);  view_111 = permute_93 = None
    view_112: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_18, [8, 768, 196]);  mm_18 = None
    add_131: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_112, arg189_1);  view_112 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_94: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_131, [0, 2, 1]);  add_131 = None
    mul_151: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_110, permute_94);  getitem_110 = permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_113: "f32[1568, 768]" = torch.ops.aten.view.default(mul_151, [1568, 768]);  mul_151 = None
    permute_95: "f32[768, 256]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_37: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg191_1, view_113, permute_95);  arg191_1 = view_113 = permute_95 = None
    view_114: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_37, [8, 196, 256]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_94: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_114);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_132: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_125, clone_94);  add_125 = clone_94 = None
    clone_95: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_95, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 196, 1]" = var_mean_38[0]
    getitem_115: "f32[8, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
    rsqrt_38: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_38: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_95, getitem_115);  clone_95 = getitem_115 = None
    mul_152: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
    mul_153: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_152, arg192_1);  mul_152 = arg192_1 = None
    add_134: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_153, arg193_1);  mul_153 = arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_115: "f32[1568, 256]" = torch.ops.aten.view.default(add_134, [1568, 256]);  add_134 = None
    permute_96: "f32[256, 1536]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    addmm_38: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg195_1, view_115, permute_96);  arg195_1 = view_115 = permute_96 = None
    view_116: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_38, [8, 196, 1536]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_154: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_155: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476);  view_116 = None
    erf_19: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_135: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_156: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_154, add_135);  mul_154 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_96: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_156);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_19 = torch.ops.aten.split.Tensor(clone_96, 768, -1);  clone_96 = None
    getitem_116: "f32[8, 196, 768]" = split_19[0]
    getitem_117: "f32[8, 196, 768]" = split_19[1];  split_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_97: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_117, memory_format = torch.contiguous_format);  getitem_117 = None
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
    getitem_118: "f32[8, 196, 1]" = var_mean_39[0]
    getitem_119: "f32[8, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_136: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_39: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_39: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_97, getitem_119);  clone_97 = getitem_119 = None
    mul_157: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
    mul_158: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_157, arg196_1);  mul_157 = arg196_1 = None
    add_137: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_158, arg197_1);  mul_158 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_97: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_137, [0, 2, 1]);  add_137 = None
    permute_98: "f32[196, 196]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    clone_98: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    view_117: "f32[6144, 196]" = torch.ops.aten.view.default(clone_98, [6144, 196]);  clone_98 = None
    mm_19: "f32[6144, 196]" = torch.ops.aten.mm.default(view_117, permute_98);  view_117 = permute_98 = None
    view_118: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_19, [8, 768, 196]);  mm_19 = None
    add_138: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_118, arg199_1);  view_118 = arg199_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_99: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_138, [0, 2, 1]);  add_138 = None
    mul_159: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_116, permute_99);  getitem_116 = permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_119: "f32[1568, 768]" = torch.ops.aten.view.default(mul_159, [1568, 768]);  mul_159 = None
    permute_100: "f32[768, 256]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
    addmm_39: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg201_1, view_119, permute_100);  arg201_1 = view_119 = permute_100 = None
    view_120: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_39, [8, 196, 256]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_99: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_139: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_132, clone_99);  add_132 = clone_99 = None
    clone_100: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_100, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 196, 1]" = var_mean_40[0]
    getitem_121: "f32[8, 196, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_40: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_40: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_100, getitem_121);  clone_100 = getitem_121 = None
    mul_160: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
    mul_161: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_160, arg202_1);  mul_160 = arg202_1 = None
    add_141: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_161, arg203_1);  mul_161 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_121: "f32[1568, 256]" = torch.ops.aten.view.default(add_141, [1568, 256]);  add_141 = None
    permute_101: "f32[256, 1536]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    addmm_40: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg205_1, view_121, permute_101);  arg205_1 = view_121 = permute_101 = None
    view_122: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_40, [8, 196, 1536]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_162: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, 0.5)
    mul_163: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476);  view_122 = None
    erf_20: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
    add_142: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_164: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_162, add_142);  mul_162 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_101: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_164);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_20 = torch.ops.aten.split.Tensor(clone_101, 768, -1);  clone_101 = None
    getitem_122: "f32[8, 196, 768]" = split_20[0]
    getitem_123: "f32[8, 196, 768]" = split_20[1];  split_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_102: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_123, memory_format = torch.contiguous_format);  getitem_123 = None
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_102, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 196, 1]" = var_mean_41[0]
    getitem_125: "f32[8, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_143: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_41: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_41: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_102, getitem_125);  clone_102 = getitem_125 = None
    mul_165: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
    mul_166: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_165, arg206_1);  mul_165 = arg206_1 = None
    add_144: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_166, arg207_1);  mul_166 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_102: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_144, [0, 2, 1]);  add_144 = None
    permute_103: "f32[196, 196]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    clone_103: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_123: "f32[6144, 196]" = torch.ops.aten.view.default(clone_103, [6144, 196]);  clone_103 = None
    mm_20: "f32[6144, 196]" = torch.ops.aten.mm.default(view_123, permute_103);  view_123 = permute_103 = None
    view_124: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_20, [8, 768, 196]);  mm_20 = None
    add_145: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_124, arg209_1);  view_124 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_104: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_145, [0, 2, 1]);  add_145 = None
    mul_167: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_122, permute_104);  getitem_122 = permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_125: "f32[1568, 768]" = torch.ops.aten.view.default(mul_167, [1568, 768]);  mul_167 = None
    permute_105: "f32[768, 256]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    addmm_41: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg211_1, view_125, permute_105);  arg211_1 = view_125 = permute_105 = None
    view_126: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_41, [8, 196, 256]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_104: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_146: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_139, clone_104);  add_139 = clone_104 = None
    clone_105: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_105, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 196, 1]" = var_mean_42[0]
    getitem_127: "f32[8, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
    rsqrt_42: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_42: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_105, getitem_127);  clone_105 = getitem_127 = None
    mul_168: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
    mul_169: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_168, arg212_1);  mul_168 = arg212_1 = None
    add_148: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_169, arg213_1);  mul_169 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_127: "f32[1568, 256]" = torch.ops.aten.view.default(add_148, [1568, 256]);  add_148 = None
    permute_106: "f32[256, 1536]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    addmm_42: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg215_1, view_127, permute_106);  arg215_1 = view_127 = permute_106 = None
    view_128: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_42, [8, 196, 1536]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_170: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, 0.5)
    mul_171: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476);  view_128 = None
    erf_21: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_149: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_172: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_170, add_149);  mul_170 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_106: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_172);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_21 = torch.ops.aten.split.Tensor(clone_106, 768, -1);  clone_106 = None
    getitem_128: "f32[8, 196, 768]" = split_21[0]
    getitem_129: "f32[8, 196, 768]" = split_21[1];  split_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_107: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_129, memory_format = torch.contiguous_format);  getitem_129 = None
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_107, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 196, 1]" = var_mean_43[0]
    getitem_131: "f32[8, 196, 1]" = var_mean_43[1];  var_mean_43 = None
    add_150: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_43: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_43: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_107, getitem_131);  clone_107 = getitem_131 = None
    mul_173: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
    mul_174: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_173, arg216_1);  mul_173 = arg216_1 = None
    add_151: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_174, arg217_1);  mul_174 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_107: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_151, [0, 2, 1]);  add_151 = None
    permute_108: "f32[196, 196]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
    clone_108: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_129: "f32[6144, 196]" = torch.ops.aten.view.default(clone_108, [6144, 196]);  clone_108 = None
    mm_21: "f32[6144, 196]" = torch.ops.aten.mm.default(view_129, permute_108);  view_129 = permute_108 = None
    view_130: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_21, [8, 768, 196]);  mm_21 = None
    add_152: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_130, arg219_1);  view_130 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_109: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_152, [0, 2, 1]);  add_152 = None
    mul_175: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_128, permute_109);  getitem_128 = permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_131: "f32[1568, 768]" = torch.ops.aten.view.default(mul_175, [1568, 768]);  mul_175 = None
    permute_110: "f32[768, 256]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
    addmm_43: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg221_1, view_131, permute_110);  arg221_1 = view_131 = permute_110 = None
    view_132: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_43, [8, 196, 256]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_109: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_132);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_153: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_146, clone_109);  add_146 = clone_109 = None
    clone_110: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_110, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 196, 1]" = var_mean_44[0]
    getitem_133: "f32[8, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_44: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_44: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_110, getitem_133);  clone_110 = getitem_133 = None
    mul_176: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
    mul_177: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_176, arg222_1);  mul_176 = arg222_1 = None
    add_155: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_177, arg223_1);  mul_177 = arg223_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_133: "f32[1568, 256]" = torch.ops.aten.view.default(add_155, [1568, 256]);  add_155 = None
    permute_111: "f32[256, 1536]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    addmm_44: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg225_1, view_133, permute_111);  arg225_1 = view_133 = permute_111 = None
    view_134: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_44, [8, 196, 1536]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_178: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, 0.5)
    mul_179: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, 0.7071067811865476);  view_134 = None
    erf_22: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_179);  mul_179 = None
    add_156: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_180: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_178, add_156);  mul_178 = add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_111: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_180);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_22 = torch.ops.aten.split.Tensor(clone_111, 768, -1);  clone_111 = None
    getitem_134: "f32[8, 196, 768]" = split_22[0]
    getitem_135: "f32[8, 196, 768]" = split_22[1];  split_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_112: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_135, memory_format = torch.contiguous_format);  getitem_135 = None
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_112, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 196, 1]" = var_mean_45[0]
    getitem_137: "f32[8, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_157: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
    rsqrt_45: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_45: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_112, getitem_137);  clone_112 = getitem_137 = None
    mul_181: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
    mul_182: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_181, arg226_1);  mul_181 = arg226_1 = None
    add_158: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_182, arg227_1);  mul_182 = arg227_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_112: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_158, [0, 2, 1]);  add_158 = None
    permute_113: "f32[196, 196]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    clone_113: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    view_135: "f32[6144, 196]" = torch.ops.aten.view.default(clone_113, [6144, 196]);  clone_113 = None
    mm_22: "f32[6144, 196]" = torch.ops.aten.mm.default(view_135, permute_113);  view_135 = permute_113 = None
    view_136: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_22, [8, 768, 196]);  mm_22 = None
    add_159: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_136, arg229_1);  view_136 = arg229_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_114: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_159, [0, 2, 1]);  add_159 = None
    mul_183: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_134, permute_114);  getitem_134 = permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_137: "f32[1568, 768]" = torch.ops.aten.view.default(mul_183, [1568, 768]);  mul_183 = None
    permute_115: "f32[768, 256]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    addmm_45: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg231_1, view_137, permute_115);  arg231_1 = view_137 = permute_115 = None
    view_138: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_45, [8, 196, 256]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_114: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_138);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_160: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_153, clone_114);  add_153 = clone_114 = None
    clone_115: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_160, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_115, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 196, 1]" = var_mean_46[0]
    getitem_139: "f32[8, 196, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
    rsqrt_46: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_46: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_115, getitem_139);  clone_115 = getitem_139 = None
    mul_184: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
    mul_185: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_184, arg232_1);  mul_184 = arg232_1 = None
    add_162: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_185, arg233_1);  mul_185 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_139: "f32[1568, 256]" = torch.ops.aten.view.default(add_162, [1568, 256]);  add_162 = None
    permute_116: "f32[256, 1536]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    addmm_46: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg235_1, view_139, permute_116);  arg235_1 = view_139 = permute_116 = None
    view_140: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_46, [8, 196, 1536]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_186: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, 0.5)
    mul_187: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, 0.7071067811865476);  view_140 = None
    erf_23: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
    add_163: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_188: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_186, add_163);  mul_186 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_116: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_188);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_23 = torch.ops.aten.split.Tensor(clone_116, 768, -1);  clone_116 = None
    getitem_140: "f32[8, 196, 768]" = split_23[0]
    getitem_141: "f32[8, 196, 768]" = split_23[1];  split_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_117: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_141, memory_format = torch.contiguous_format);  getitem_141 = None
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_117, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 196, 1]" = var_mean_47[0]
    getitem_143: "f32[8, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_164: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
    rsqrt_47: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_47: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_117, getitem_143);  clone_117 = getitem_143 = None
    mul_189: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
    mul_190: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_189, arg236_1);  mul_189 = arg236_1 = None
    add_165: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_190, arg237_1);  mul_190 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_117: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_165, [0, 2, 1]);  add_165 = None
    permute_118: "f32[196, 196]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    clone_118: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_141: "f32[6144, 196]" = torch.ops.aten.view.default(clone_118, [6144, 196]);  clone_118 = None
    mm_23: "f32[6144, 196]" = torch.ops.aten.mm.default(view_141, permute_118);  view_141 = permute_118 = None
    view_142: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_23, [8, 768, 196]);  mm_23 = None
    add_166: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_142, arg239_1);  view_142 = arg239_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_119: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_166, [0, 2, 1]);  add_166 = None
    mul_191: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_140, permute_119);  getitem_140 = permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_143: "f32[1568, 768]" = torch.ops.aten.view.default(mul_191, [1568, 768]);  mul_191 = None
    permute_120: "f32[768, 256]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    addmm_47: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg241_1, view_143, permute_120);  arg241_1 = view_143 = permute_120 = None
    view_144: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_47, [8, 196, 256]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_119: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_167: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_160, clone_119);  add_160 = clone_119 = None
    clone_120: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_120, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 196, 1]" = var_mean_48[0]
    getitem_145: "f32[8, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_48: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_48: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_120, getitem_145);  clone_120 = getitem_145 = None
    mul_192: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
    mul_193: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_192, arg242_1);  mul_192 = arg242_1 = None
    add_169: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_193, arg243_1);  mul_193 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_145: "f32[1568, 256]" = torch.ops.aten.view.default(add_169, [1568, 256]);  add_169 = None
    permute_121: "f32[256, 1536]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    addmm_48: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg245_1, view_145, permute_121);  arg245_1 = view_145 = permute_121 = None
    view_146: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_48, [8, 196, 1536]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_194: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, 0.5)
    mul_195: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476);  view_146 = None
    erf_24: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_170: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_196: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_194, add_170);  mul_194 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_121: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_196);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_24 = torch.ops.aten.split.Tensor(clone_121, 768, -1);  clone_121 = None
    getitem_146: "f32[8, 196, 768]" = split_24[0]
    getitem_147: "f32[8, 196, 768]" = split_24[1];  split_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_122: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_147, memory_format = torch.contiguous_format);  getitem_147 = None
    var_mean_49 = torch.ops.aten.var_mean.correction(clone_122, [2], correction = 0, keepdim = True)
    getitem_148: "f32[8, 196, 1]" = var_mean_49[0]
    getitem_149: "f32[8, 196, 1]" = var_mean_49[1];  var_mean_49 = None
    add_171: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
    rsqrt_49: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_49: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_122, getitem_149);  clone_122 = getitem_149 = None
    mul_197: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
    mul_198: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_197, arg246_1);  mul_197 = arg246_1 = None
    add_172: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_198, arg247_1);  mul_198 = arg247_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_122: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_172, [0, 2, 1]);  add_172 = None
    permute_123: "f32[196, 196]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
    clone_123: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_147: "f32[6144, 196]" = torch.ops.aten.view.default(clone_123, [6144, 196]);  clone_123 = None
    mm_24: "f32[6144, 196]" = torch.ops.aten.mm.default(view_147, permute_123);  view_147 = permute_123 = None
    view_148: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_24, [8, 768, 196]);  mm_24 = None
    add_173: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_148, arg249_1);  view_148 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_124: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_173, [0, 2, 1]);  add_173 = None
    mul_199: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_146, permute_124);  getitem_146 = permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_149: "f32[1568, 768]" = torch.ops.aten.view.default(mul_199, [1568, 768]);  mul_199 = None
    permute_125: "f32[768, 256]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    addmm_49: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg251_1, view_149, permute_125);  arg251_1 = view_149 = permute_125 = None
    view_150: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_49, [8, 196, 256]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_124: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_150);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_174: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_167, clone_124);  add_167 = clone_124 = None
    clone_125: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_174, memory_format = torch.contiguous_format)
    var_mean_50 = torch.ops.aten.var_mean.correction(clone_125, [2], correction = 0, keepdim = True)
    getitem_150: "f32[8, 196, 1]" = var_mean_50[0]
    getitem_151: "f32[8, 196, 1]" = var_mean_50[1];  var_mean_50 = None
    add_175: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-06);  getitem_150 = None
    rsqrt_50: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_50: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_125, getitem_151);  clone_125 = getitem_151 = None
    mul_200: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
    mul_201: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_200, arg252_1);  mul_200 = arg252_1 = None
    add_176: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_201, arg253_1);  mul_201 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_151: "f32[1568, 256]" = torch.ops.aten.view.default(add_176, [1568, 256]);  add_176 = None
    permute_126: "f32[256, 1536]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_50: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg255_1, view_151, permute_126);  arg255_1 = view_151 = permute_126 = None
    view_152: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_50, [8, 196, 1536]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_202: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, 0.5)
    mul_203: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, 0.7071067811865476);  view_152 = None
    erf_25: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_203);  mul_203 = None
    add_177: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_204: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_202, add_177);  mul_202 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_126: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_204);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_25 = torch.ops.aten.split.Tensor(clone_126, 768, -1);  clone_126 = None
    getitem_152: "f32[8, 196, 768]" = split_25[0]
    getitem_153: "f32[8, 196, 768]" = split_25[1];  split_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_127: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_153, memory_format = torch.contiguous_format);  getitem_153 = None
    var_mean_51 = torch.ops.aten.var_mean.correction(clone_127, [2], correction = 0, keepdim = True)
    getitem_154: "f32[8, 196, 1]" = var_mean_51[0]
    getitem_155: "f32[8, 196, 1]" = var_mean_51[1];  var_mean_51 = None
    add_178: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
    rsqrt_51: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_51: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_127, getitem_155);  clone_127 = getitem_155 = None
    mul_205: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
    mul_206: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_205, arg256_1);  mul_205 = arg256_1 = None
    add_179: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_206, arg257_1);  mul_206 = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_127: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_179, [0, 2, 1]);  add_179 = None
    permute_128: "f32[196, 196]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
    clone_128: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_153: "f32[6144, 196]" = torch.ops.aten.view.default(clone_128, [6144, 196]);  clone_128 = None
    mm_25: "f32[6144, 196]" = torch.ops.aten.mm.default(view_153, permute_128);  view_153 = permute_128 = None
    view_154: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_25, [8, 768, 196]);  mm_25 = None
    add_180: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_154, arg259_1);  view_154 = arg259_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_129: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_180, [0, 2, 1]);  add_180 = None
    mul_207: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_152, permute_129);  getitem_152 = permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_155: "f32[1568, 768]" = torch.ops.aten.view.default(mul_207, [1568, 768]);  mul_207 = None
    permute_130: "f32[768, 256]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    addmm_51: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg261_1, view_155, permute_130);  arg261_1 = view_155 = permute_130 = None
    view_156: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_51, [8, 196, 256]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_129: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_181: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_174, clone_129);  add_174 = clone_129 = None
    clone_130: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_181, memory_format = torch.contiguous_format)
    var_mean_52 = torch.ops.aten.var_mean.correction(clone_130, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 196, 1]" = var_mean_52[0]
    getitem_157: "f32[8, 196, 1]" = var_mean_52[1];  var_mean_52 = None
    add_182: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
    rsqrt_52: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_52: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_130, getitem_157);  clone_130 = getitem_157 = None
    mul_208: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
    mul_209: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_208, arg262_1);  mul_208 = arg262_1 = None
    add_183: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_209, arg263_1);  mul_209 = arg263_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_157: "f32[1568, 256]" = torch.ops.aten.view.default(add_183, [1568, 256]);  add_183 = None
    permute_131: "f32[256, 1536]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
    addmm_52: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg265_1, view_157, permute_131);  arg265_1 = view_157 = permute_131 = None
    view_158: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_52, [8, 196, 1536]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_210: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_211: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_26: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
    add_184: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_212: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_210, add_184);  mul_210 = add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_131: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_212);  mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_26 = torch.ops.aten.split.Tensor(clone_131, 768, -1);  clone_131 = None
    getitem_158: "f32[8, 196, 768]" = split_26[0]
    getitem_159: "f32[8, 196, 768]" = split_26[1];  split_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_132: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_159, memory_format = torch.contiguous_format);  getitem_159 = None
    var_mean_53 = torch.ops.aten.var_mean.correction(clone_132, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 196, 1]" = var_mean_53[0]
    getitem_161: "f32[8, 196, 1]" = var_mean_53[1];  var_mean_53 = None
    add_185: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
    rsqrt_53: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_53: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_132, getitem_161);  clone_132 = getitem_161 = None
    mul_213: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
    mul_214: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_213, arg266_1);  mul_213 = arg266_1 = None
    add_186: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_214, arg267_1);  mul_214 = arg267_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_132: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_186, [0, 2, 1]);  add_186 = None
    permute_133: "f32[196, 196]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
    clone_133: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_159: "f32[6144, 196]" = torch.ops.aten.view.default(clone_133, [6144, 196]);  clone_133 = None
    mm_26: "f32[6144, 196]" = torch.ops.aten.mm.default(view_159, permute_133);  view_159 = permute_133 = None
    view_160: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_26, [8, 768, 196]);  mm_26 = None
    add_187: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_160, arg269_1);  view_160 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_134: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_187, [0, 2, 1]);  add_187 = None
    mul_215: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_158, permute_134);  getitem_158 = permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_161: "f32[1568, 768]" = torch.ops.aten.view.default(mul_215, [1568, 768]);  mul_215 = None
    permute_135: "f32[768, 256]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
    addmm_53: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg271_1, view_161, permute_135);  arg271_1 = view_161 = permute_135 = None
    view_162: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_53, [8, 196, 256]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_134: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_188: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_181, clone_134);  add_181 = clone_134 = None
    clone_135: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_188, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_135, [2], correction = 0, keepdim = True)
    getitem_162: "f32[8, 196, 1]" = var_mean_54[0]
    getitem_163: "f32[8, 196, 1]" = var_mean_54[1];  var_mean_54 = None
    add_189: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_54: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_54: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_135, getitem_163);  clone_135 = getitem_163 = None
    mul_216: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None
    mul_217: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_216, arg272_1);  mul_216 = arg272_1 = None
    add_190: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_217, arg273_1);  mul_217 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_163: "f32[1568, 256]" = torch.ops.aten.view.default(add_190, [1568, 256]);  add_190 = None
    permute_136: "f32[256, 1536]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
    addmm_54: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg275_1, view_163, permute_136);  arg275_1 = view_163 = permute_136 = None
    view_164: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_54, [8, 196, 1536]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_218: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, 0.5)
    mul_219: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476);  view_164 = None
    erf_27: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_191: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_220: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_218, add_191);  mul_218 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_136: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_220);  mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_27 = torch.ops.aten.split.Tensor(clone_136, 768, -1);  clone_136 = None
    getitem_164: "f32[8, 196, 768]" = split_27[0]
    getitem_165: "f32[8, 196, 768]" = split_27[1];  split_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_137: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_165, memory_format = torch.contiguous_format);  getitem_165 = None
    var_mean_55 = torch.ops.aten.var_mean.correction(clone_137, [2], correction = 0, keepdim = True)
    getitem_166: "f32[8, 196, 1]" = var_mean_55[0]
    getitem_167: "f32[8, 196, 1]" = var_mean_55[1];  var_mean_55 = None
    add_192: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
    rsqrt_55: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_55: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_137, getitem_167);  clone_137 = getitem_167 = None
    mul_221: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
    mul_222: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_221, arg276_1);  mul_221 = arg276_1 = None
    add_193: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_222, arg277_1);  mul_222 = arg277_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_137: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_193, [0, 2, 1]);  add_193 = None
    permute_138: "f32[196, 196]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
    clone_138: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    view_165: "f32[6144, 196]" = torch.ops.aten.view.default(clone_138, [6144, 196]);  clone_138 = None
    mm_27: "f32[6144, 196]" = torch.ops.aten.mm.default(view_165, permute_138);  view_165 = permute_138 = None
    view_166: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_27, [8, 768, 196]);  mm_27 = None
    add_194: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_166, arg279_1);  view_166 = arg279_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_139: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_194, [0, 2, 1]);  add_194 = None
    mul_223: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_164, permute_139);  getitem_164 = permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_167: "f32[1568, 768]" = torch.ops.aten.view.default(mul_223, [1568, 768]);  mul_223 = None
    permute_140: "f32[768, 256]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
    addmm_55: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg281_1, view_167, permute_140);  arg281_1 = view_167 = permute_140 = None
    view_168: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_55, [8, 196, 256]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_139: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_168);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_195: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_188, clone_139);  add_188 = clone_139 = None
    clone_140: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_195, memory_format = torch.contiguous_format)
    var_mean_56 = torch.ops.aten.var_mean.correction(clone_140, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 196, 1]" = var_mean_56[0]
    getitem_169: "f32[8, 196, 1]" = var_mean_56[1];  var_mean_56 = None
    add_196: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_56: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_56: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_140, getitem_169);  clone_140 = getitem_169 = None
    mul_224: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None
    mul_225: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_224, arg282_1);  mul_224 = arg282_1 = None
    add_197: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_225, arg283_1);  mul_225 = arg283_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_169: "f32[1568, 256]" = torch.ops.aten.view.default(add_197, [1568, 256]);  add_197 = None
    permute_141: "f32[256, 1536]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
    addmm_56: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg285_1, view_169, permute_141);  arg285_1 = view_169 = permute_141 = None
    view_170: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_56, [8, 196, 1536]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_226: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, 0.5)
    mul_227: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, 0.7071067811865476);  view_170 = None
    erf_28: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_227);  mul_227 = None
    add_198: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_228: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_226, add_198);  mul_226 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_141: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_228);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_28 = torch.ops.aten.split.Tensor(clone_141, 768, -1);  clone_141 = None
    getitem_170: "f32[8, 196, 768]" = split_28[0]
    getitem_171: "f32[8, 196, 768]" = split_28[1];  split_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_142: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_171, memory_format = torch.contiguous_format);  getitem_171 = None
    var_mean_57 = torch.ops.aten.var_mean.correction(clone_142, [2], correction = 0, keepdim = True)
    getitem_172: "f32[8, 196, 1]" = var_mean_57[0]
    getitem_173: "f32[8, 196, 1]" = var_mean_57[1];  var_mean_57 = None
    add_199: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
    rsqrt_57: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_57: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_142, getitem_173);  clone_142 = getitem_173 = None
    mul_229: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
    mul_230: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_229, arg286_1);  mul_229 = arg286_1 = None
    add_200: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_230, arg287_1);  mul_230 = arg287_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_142: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_200, [0, 2, 1]);  add_200 = None
    permute_143: "f32[196, 196]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
    clone_143: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_171: "f32[6144, 196]" = torch.ops.aten.view.default(clone_143, [6144, 196]);  clone_143 = None
    mm_28: "f32[6144, 196]" = torch.ops.aten.mm.default(view_171, permute_143);  view_171 = permute_143 = None
    view_172: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_28, [8, 768, 196]);  mm_28 = None
    add_201: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_172, arg289_1);  view_172 = arg289_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_144: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_201, [0, 2, 1]);  add_201 = None
    mul_231: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_170, permute_144);  getitem_170 = permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_173: "f32[1568, 768]" = torch.ops.aten.view.default(mul_231, [1568, 768]);  mul_231 = None
    permute_145: "f32[768, 256]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
    addmm_57: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg291_1, view_173, permute_145);  arg291_1 = view_173 = permute_145 = None
    view_174: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_57, [8, 196, 256]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_144: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_174);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_202: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_195, clone_144);  add_195 = clone_144 = None
    clone_145: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_202, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_145, [2], correction = 0, keepdim = True)
    getitem_174: "f32[8, 196, 1]" = var_mean_58[0]
    getitem_175: "f32[8, 196, 1]" = var_mean_58[1];  var_mean_58 = None
    add_203: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_58: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_58: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_145, getitem_175);  clone_145 = getitem_175 = None
    mul_232: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
    mul_233: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_232, arg292_1);  mul_232 = arg292_1 = None
    add_204: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_233, arg293_1);  mul_233 = arg293_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_175: "f32[1568, 256]" = torch.ops.aten.view.default(add_204, [1568, 256]);  add_204 = None
    permute_146: "f32[256, 1536]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
    addmm_58: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg295_1, view_175, permute_146);  arg295_1 = view_175 = permute_146 = None
    view_176: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_58, [8, 196, 1536]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_234: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, 0.5)
    mul_235: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476);  view_176 = None
    erf_29: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_235);  mul_235 = None
    add_205: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_236: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_234, add_205);  mul_234 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:187, code: x = self.drop1(x)
    clone_146: "f32[8, 196, 1536]" = torch.ops.aten.clone.default(mul_236);  mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    split_29 = torch.ops.aten.split.Tensor(clone_146, 768, -1);  clone_146 = None
    getitem_176: "f32[8, 196, 768]" = split_29[0]
    getitem_177: "f32[8, 196, 768]" = split_29[1];  split_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_147: "f32[8, 196, 768]" = torch.ops.aten.clone.default(getitem_177, memory_format = torch.contiguous_format);  getitem_177 = None
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_147, [2], correction = 0, keepdim = True)
    getitem_178: "f32[8, 196, 1]" = var_mean_59[0]
    getitem_179: "f32[8, 196, 1]" = var_mean_59[1];  var_mean_59 = None
    add_206: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
    rsqrt_59: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    sub_59: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(clone_147, getitem_179);  clone_147 = getitem_179 = None
    mul_237: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
    mul_238: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_237, arg296_1);  mul_237 = arg296_1 = None
    add_207: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_238, arg297_1);  mul_238 = arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    permute_147: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_207, [0, 2, 1]);  add_207 = None
    permute_148: "f32[196, 196]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
    clone_148: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    view_177: "f32[6144, 196]" = torch.ops.aten.view.default(clone_148, [6144, 196]);  clone_148 = None
    mm_29: "f32[6144, 196]" = torch.ops.aten.mm.default(view_177, permute_148);  view_177 = permute_148 = None
    view_178: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_29, [8, 768, 196]);  mm_29 = None
    add_208: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_178, arg299_1);  view_178 = arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_149: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_208, [0, 2, 1]);  add_208 = None
    mul_239: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_176, permute_149);  getitem_176 = permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_179: "f32[1568, 768]" = torch.ops.aten.view.default(mul_239, [1568, 768]);  mul_239 = None
    permute_150: "f32[768, 256]" = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
    addmm_59: "f32[1568, 256]" = torch.ops.aten.addmm.default(arg301_1, view_179, permute_150);  arg301_1 = view_179 = permute_150 = None
    view_180: "f32[8, 196, 256]" = torch.ops.aten.view.default(addmm_59, [8, 196, 256]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:191, code: x = self.drop2(x)
    clone_149: "f32[8, 196, 256]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_209: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_202, clone_149);  add_202 = clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_150: "f32[8, 196, 256]" = torch.ops.aten.clone.default(add_209, memory_format = torch.contiguous_format);  add_209 = None
    var_mean_60 = torch.ops.aten.var_mean.correction(clone_150, [2], correction = 0, keepdim = True)
    getitem_180: "f32[8, 196, 1]" = var_mean_60[0]
    getitem_181: "f32[8, 196, 1]" = var_mean_60[1];  var_mean_60 = None
    add_210: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
    rsqrt_60: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_60: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(clone_150, getitem_181);  clone_150 = getitem_181 = None
    mul_240: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
    mul_241: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_240, arg302_1);  mul_240 = arg302_1 = None
    add_211: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_241, arg303_1);  mul_241 = arg303_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 256]" = torch.ops.aten.mean.dim(add_211, [1]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    clone_151: "f32[8, 256]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_151: "f32[256, 1000]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
    addmm_60: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg305_1, clone_151, permute_151);  arg305_1 = clone_151 = permute_151 = None
    return (addmm_60,)
    