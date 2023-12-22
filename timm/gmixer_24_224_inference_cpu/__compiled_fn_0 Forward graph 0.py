from __future__ import annotations



def forward(self, arg0_1: "f32[384, 3, 16, 16]", arg1_1: "f32[384]", arg2_1: "f32[384]", arg3_1: "f32[384]", arg4_1: "f32[384, 196]", arg5_1: "f32[384]", arg6_1: "f32[196, 192]", arg7_1: "f32[196]", arg8_1: "f32[384]", arg9_1: "f32[384]", arg10_1: "f32[1536, 384]", arg11_1: "f32[1536]", arg12_1: "f32[384, 768]", arg13_1: "f32[384]", arg14_1: "f32[384]", arg15_1: "f32[384]", arg16_1: "f32[384, 196]", arg17_1: "f32[384]", arg18_1: "f32[196, 192]", arg19_1: "f32[196]", arg20_1: "f32[384]", arg21_1: "f32[384]", arg22_1: "f32[1536, 384]", arg23_1: "f32[1536]", arg24_1: "f32[384, 768]", arg25_1: "f32[384]", arg26_1: "f32[384]", arg27_1: "f32[384]", arg28_1: "f32[384, 196]", arg29_1: "f32[384]", arg30_1: "f32[196, 192]", arg31_1: "f32[196]", arg32_1: "f32[384]", arg33_1: "f32[384]", arg34_1: "f32[1536, 384]", arg35_1: "f32[1536]", arg36_1: "f32[384, 768]", arg37_1: "f32[384]", arg38_1: "f32[384]", arg39_1: "f32[384]", arg40_1: "f32[384, 196]", arg41_1: "f32[384]", arg42_1: "f32[196, 192]", arg43_1: "f32[196]", arg44_1: "f32[384]", arg45_1: "f32[384]", arg46_1: "f32[1536, 384]", arg47_1: "f32[1536]", arg48_1: "f32[384, 768]", arg49_1: "f32[384]", arg50_1: "f32[384]", arg51_1: "f32[384]", arg52_1: "f32[384, 196]", arg53_1: "f32[384]", arg54_1: "f32[196, 192]", arg55_1: "f32[196]", arg56_1: "f32[384]", arg57_1: "f32[384]", arg58_1: "f32[1536, 384]", arg59_1: "f32[1536]", arg60_1: "f32[384, 768]", arg61_1: "f32[384]", arg62_1: "f32[384]", arg63_1: "f32[384]", arg64_1: "f32[384, 196]", arg65_1: "f32[384]", arg66_1: "f32[196, 192]", arg67_1: "f32[196]", arg68_1: "f32[384]", arg69_1: "f32[384]", arg70_1: "f32[1536, 384]", arg71_1: "f32[1536]", arg72_1: "f32[384, 768]", arg73_1: "f32[384]", arg74_1: "f32[384]", arg75_1: "f32[384]", arg76_1: "f32[384, 196]", arg77_1: "f32[384]", arg78_1: "f32[196, 192]", arg79_1: "f32[196]", arg80_1: "f32[384]", arg81_1: "f32[384]", arg82_1: "f32[1536, 384]", arg83_1: "f32[1536]", arg84_1: "f32[384, 768]", arg85_1: "f32[384]", arg86_1: "f32[384]", arg87_1: "f32[384]", arg88_1: "f32[384, 196]", arg89_1: "f32[384]", arg90_1: "f32[196, 192]", arg91_1: "f32[196]", arg92_1: "f32[384]", arg93_1: "f32[384]", arg94_1: "f32[1536, 384]", arg95_1: "f32[1536]", arg96_1: "f32[384, 768]", arg97_1: "f32[384]", arg98_1: "f32[384]", arg99_1: "f32[384]", arg100_1: "f32[384, 196]", arg101_1: "f32[384]", arg102_1: "f32[196, 192]", arg103_1: "f32[196]", arg104_1: "f32[384]", arg105_1: "f32[384]", arg106_1: "f32[1536, 384]", arg107_1: "f32[1536]", arg108_1: "f32[384, 768]", arg109_1: "f32[384]", arg110_1: "f32[384]", arg111_1: "f32[384]", arg112_1: "f32[384, 196]", arg113_1: "f32[384]", arg114_1: "f32[196, 192]", arg115_1: "f32[196]", arg116_1: "f32[384]", arg117_1: "f32[384]", arg118_1: "f32[1536, 384]", arg119_1: "f32[1536]", arg120_1: "f32[384, 768]", arg121_1: "f32[384]", arg122_1: "f32[384]", arg123_1: "f32[384]", arg124_1: "f32[384, 196]", arg125_1: "f32[384]", arg126_1: "f32[196, 192]", arg127_1: "f32[196]", arg128_1: "f32[384]", arg129_1: "f32[384]", arg130_1: "f32[1536, 384]", arg131_1: "f32[1536]", arg132_1: "f32[384, 768]", arg133_1: "f32[384]", arg134_1: "f32[384]", arg135_1: "f32[384]", arg136_1: "f32[384, 196]", arg137_1: "f32[384]", arg138_1: "f32[196, 192]", arg139_1: "f32[196]", arg140_1: "f32[384]", arg141_1: "f32[384]", arg142_1: "f32[1536, 384]", arg143_1: "f32[1536]", arg144_1: "f32[384, 768]", arg145_1: "f32[384]", arg146_1: "f32[384]", arg147_1: "f32[384]", arg148_1: "f32[384, 196]", arg149_1: "f32[384]", arg150_1: "f32[196, 192]", arg151_1: "f32[196]", arg152_1: "f32[384]", arg153_1: "f32[384]", arg154_1: "f32[1536, 384]", arg155_1: "f32[1536]", arg156_1: "f32[384, 768]", arg157_1: "f32[384]", arg158_1: "f32[384]", arg159_1: "f32[384]", arg160_1: "f32[384, 196]", arg161_1: "f32[384]", arg162_1: "f32[196, 192]", arg163_1: "f32[196]", arg164_1: "f32[384]", arg165_1: "f32[384]", arg166_1: "f32[1536, 384]", arg167_1: "f32[1536]", arg168_1: "f32[384, 768]", arg169_1: "f32[384]", arg170_1: "f32[384]", arg171_1: "f32[384]", arg172_1: "f32[384, 196]", arg173_1: "f32[384]", arg174_1: "f32[196, 192]", arg175_1: "f32[196]", arg176_1: "f32[384]", arg177_1: "f32[384]", arg178_1: "f32[1536, 384]", arg179_1: "f32[1536]", arg180_1: "f32[384, 768]", arg181_1: "f32[384]", arg182_1: "f32[384]", arg183_1: "f32[384]", arg184_1: "f32[384, 196]", arg185_1: "f32[384]", arg186_1: "f32[196, 192]", arg187_1: "f32[196]", arg188_1: "f32[384]", arg189_1: "f32[384]", arg190_1: "f32[1536, 384]", arg191_1: "f32[1536]", arg192_1: "f32[384, 768]", arg193_1: "f32[384]", arg194_1: "f32[384]", arg195_1: "f32[384]", arg196_1: "f32[384, 196]", arg197_1: "f32[384]", arg198_1: "f32[196, 192]", arg199_1: "f32[196]", arg200_1: "f32[384]", arg201_1: "f32[384]", arg202_1: "f32[1536, 384]", arg203_1: "f32[1536]", arg204_1: "f32[384, 768]", arg205_1: "f32[384]", arg206_1: "f32[384]", arg207_1: "f32[384]", arg208_1: "f32[384, 196]", arg209_1: "f32[384]", arg210_1: "f32[196, 192]", arg211_1: "f32[196]", arg212_1: "f32[384]", arg213_1: "f32[384]", arg214_1: "f32[1536, 384]", arg215_1: "f32[1536]", arg216_1: "f32[384, 768]", arg217_1: "f32[384]", arg218_1: "f32[384]", arg219_1: "f32[384]", arg220_1: "f32[384, 196]", arg221_1: "f32[384]", arg222_1: "f32[196, 192]", arg223_1: "f32[196]", arg224_1: "f32[384]", arg225_1: "f32[384]", arg226_1: "f32[1536, 384]", arg227_1: "f32[1536]", arg228_1: "f32[384, 768]", arg229_1: "f32[384]", arg230_1: "f32[384]", arg231_1: "f32[384]", arg232_1: "f32[384, 196]", arg233_1: "f32[384]", arg234_1: "f32[196, 192]", arg235_1: "f32[196]", arg236_1: "f32[384]", arg237_1: "f32[384]", arg238_1: "f32[1536, 384]", arg239_1: "f32[1536]", arg240_1: "f32[384, 768]", arg241_1: "f32[384]", arg242_1: "f32[384]", arg243_1: "f32[384]", arg244_1: "f32[384, 196]", arg245_1: "f32[384]", arg246_1: "f32[196, 192]", arg247_1: "f32[196]", arg248_1: "f32[384]", arg249_1: "f32[384]", arg250_1: "f32[1536, 384]", arg251_1: "f32[1536]", arg252_1: "f32[384, 768]", arg253_1: "f32[384]", arg254_1: "f32[384]", arg255_1: "f32[384]", arg256_1: "f32[384, 196]", arg257_1: "f32[384]", arg258_1: "f32[196, 192]", arg259_1: "f32[196]", arg260_1: "f32[384]", arg261_1: "f32[384]", arg262_1: "f32[1536, 384]", arg263_1: "f32[1536]", arg264_1: "f32[384, 768]", arg265_1: "f32[384]", arg266_1: "f32[384]", arg267_1: "f32[384]", arg268_1: "f32[384, 196]", arg269_1: "f32[384]", arg270_1: "f32[196, 192]", arg271_1: "f32[196]", arg272_1: "f32[384]", arg273_1: "f32[384]", arg274_1: "f32[1536, 384]", arg275_1: "f32[1536]", arg276_1: "f32[384, 768]", arg277_1: "f32[384]", arg278_1: "f32[384]", arg279_1: "f32[384]", arg280_1: "f32[384, 196]", arg281_1: "f32[384]", arg282_1: "f32[196, 192]", arg283_1: "f32[196]", arg284_1: "f32[384]", arg285_1: "f32[384]", arg286_1: "f32[1536, 384]", arg287_1: "f32[1536]", arg288_1: "f32[384, 768]", arg289_1: "f32[384]", arg290_1: "f32[384]", arg291_1: "f32[384]", arg292_1: "f32[1000, 384]", arg293_1: "f32[1000]", arg294_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(arg294_1, arg0_1, arg1_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  arg294_1 = arg0_1 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 384, 196]" = torch.ops.aten.view.default(convolution, [8, 384, 196]);  convolution = None
    permute: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_1: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    permute_1: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_1, [0, 2, 1]);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_2: "f32[196, 384]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    clone_1: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[3072, 196]" = torch.ops.aten.view.default(clone_1, [3072, 196]);  clone_1 = None
    mm: "f32[3072, 384]" = torch.ops.aten.mm.default(view_1, permute_2);  view_1 = permute_2 = None
    view_2: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm, [8, 384, 384]);  mm = None
    add_2: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_2, arg5_1);  view_2 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split = torch.ops.aten.split.Tensor(add_2, 192, -1);  add_2 = None
    getitem_2: "f32[8, 384, 192]" = split[0]
    getitem_3: "f32[8, 384, 192]" = split[1];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_3)
    mul_2: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_3, sigmoid);  getitem_3 = sigmoid = None
    mul_3: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_2, mul_2);  getitem_2 = mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_2: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_3);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_3: "f32[3072, 192]" = torch.ops.aten.view.default(clone_2, [3072, 192]);  clone_2 = None
    permute_3: "f32[192, 196]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg7_1, view_3, permute_3);  arg7_1 = view_3 = permute_3 = None
    view_4: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm, [8, 384, 196]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_3: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_4);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_4: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_3, [0, 2, 1]);  clone_3 = None
    add_3: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(permute, permute_4);  permute = permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_4: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_3, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_5: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_4, getitem_5);  clone_4 = getitem_5 = None
    mul_4: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_5: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_4, arg8_1);  mul_4 = arg8_1 = None
    add_5: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_5, arg9_1);  mul_5 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_5: "f32[1568, 384]" = torch.ops.aten.view.default(add_5, [1568, 384]);  add_5 = None
    permute_5: "f32[384, 1536]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm_1: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg11_1, view_5, permute_5);  arg11_1 = view_5 = permute_5 = None
    view_6: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_1, [8, 196, 1536]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_1 = torch.ops.aten.split.Tensor(view_6, 768, -1);  view_6 = None
    getitem_6: "f32[8, 196, 768]" = split_1[0]
    getitem_7: "f32[8, 196, 768]" = split_1[1];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_1: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_7)
    mul_6: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_7, sigmoid_1);  getitem_7 = sigmoid_1 = None
    mul_7: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_6, mul_6);  getitem_6 = mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_5: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_7: "f32[1568, 768]" = torch.ops.aten.view.default(clone_5, [1568, 768]);  clone_5 = None
    permute_6: "f32[768, 384]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    addmm_2: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg13_1, view_7, permute_6);  arg13_1 = view_7 = permute_6 = None
    view_8: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_2, [8, 196, 384]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_6: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_6: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_3, clone_6);  add_3 = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_7: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_6, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_7, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 196, 1]" = var_mean_2[0]
    getitem_9: "f32[8, 196, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_2: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_7, getitem_9);  clone_7 = getitem_9 = None
    mul_8: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_9: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_8, arg14_1);  mul_8 = arg14_1 = None
    add_8: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_9, arg15_1);  mul_9 = arg15_1 = None
    permute_7: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_8, [0, 2, 1]);  add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_8: "f32[196, 384]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    clone_8: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_9: "f32[3072, 196]" = torch.ops.aten.view.default(clone_8, [3072, 196]);  clone_8 = None
    mm_1: "f32[3072, 384]" = torch.ops.aten.mm.default(view_9, permute_8);  view_9 = permute_8 = None
    view_10: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_1, [8, 384, 384]);  mm_1 = None
    add_9: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_10, arg17_1);  view_10 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_2 = torch.ops.aten.split.Tensor(add_9, 192, -1);  add_9 = None
    getitem_10: "f32[8, 384, 192]" = split_2[0]
    getitem_11: "f32[8, 384, 192]" = split_2[1];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_2: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_11)
    mul_10: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_11, sigmoid_2);  getitem_11 = sigmoid_2 = None
    mul_11: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_10, mul_10);  getitem_10 = mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_9: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_11);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_11: "f32[3072, 192]" = torch.ops.aten.view.default(clone_9, [3072, 192]);  clone_9 = None
    permute_9: "f32[192, 196]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    addmm_3: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg19_1, view_11, permute_9);  arg19_1 = view_11 = permute_9 = None
    view_12: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_3, [8, 384, 196]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_10: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_10: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_10, [0, 2, 1]);  clone_10 = None
    add_10: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_6, permute_10);  add_6 = permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_11: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 196, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 196, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_11, getitem_13);  clone_11 = getitem_13 = None
    mul_12: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    mul_13: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_12, arg20_1);  mul_12 = arg20_1 = None
    add_12: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_13, arg21_1);  mul_13 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_13: "f32[1568, 384]" = torch.ops.aten.view.default(add_12, [1568, 384]);  add_12 = None
    permute_11: "f32[384, 1536]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    addmm_4: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg23_1, view_13, permute_11);  arg23_1 = view_13 = permute_11 = None
    view_14: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_4, [8, 196, 1536]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_3 = torch.ops.aten.split.Tensor(view_14, 768, -1);  view_14 = None
    getitem_14: "f32[8, 196, 768]" = split_3[0]
    getitem_15: "f32[8, 196, 768]" = split_3[1];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_3: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_15)
    mul_14: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_15, sigmoid_3);  getitem_15 = sigmoid_3 = None
    mul_15: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_14, mul_14);  getitem_14 = mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_12: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_15);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_15: "f32[1568, 768]" = torch.ops.aten.view.default(clone_12, [1568, 768]);  clone_12 = None
    permute_12: "f32[768, 384]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_5: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg25_1, view_15, permute_12);  arg25_1 = view_15 = permute_12 = None
    view_16: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_5, [8, 196, 384]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_13: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_13: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_10, clone_13);  add_10 = clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_14: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 196, 1]" = var_mean_4[0]
    getitem_17: "f32[8, 196, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_4: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_14, getitem_17);  clone_14 = getitem_17 = None
    mul_16: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    mul_17: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_16, arg26_1);  mul_16 = arg26_1 = None
    add_15: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_17, arg27_1);  mul_17 = arg27_1 = None
    permute_13: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_15, [0, 2, 1]);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_14: "f32[196, 384]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
    clone_15: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_17: "f32[3072, 196]" = torch.ops.aten.view.default(clone_15, [3072, 196]);  clone_15 = None
    mm_2: "f32[3072, 384]" = torch.ops.aten.mm.default(view_17, permute_14);  view_17 = permute_14 = None
    view_18: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_2, [8, 384, 384]);  mm_2 = None
    add_16: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_18, arg29_1);  view_18 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_4 = torch.ops.aten.split.Tensor(add_16, 192, -1);  add_16 = None
    getitem_18: "f32[8, 384, 192]" = split_4[0]
    getitem_19: "f32[8, 384, 192]" = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_4: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_19)
    mul_18: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_19, sigmoid_4);  getitem_19 = sigmoid_4 = None
    mul_19: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_18, mul_18);  getitem_18 = mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_16: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_19: "f32[3072, 192]" = torch.ops.aten.view.default(clone_16, [3072, 192]);  clone_16 = None
    permute_15: "f32[192, 196]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_6: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg31_1, view_19, permute_15);  arg31_1 = view_19 = permute_15 = None
    view_20: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_6, [8, 384, 196]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_17: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_16: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_17, [0, 2, 1]);  clone_17 = None
    add_17: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_13, permute_16);  add_13 = permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_18: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_18, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 196, 1]" = var_mean_5[0]
    getitem_21: "f32[8, 196, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_5: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_5: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_18, getitem_21);  clone_18 = getitem_21 = None
    mul_20: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    mul_21: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_20, arg32_1);  mul_20 = arg32_1 = None
    add_19: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_21, arg33_1);  mul_21 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_21: "f32[1568, 384]" = torch.ops.aten.view.default(add_19, [1568, 384]);  add_19 = None
    permute_17: "f32[384, 1536]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
    addmm_7: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg35_1, view_21, permute_17);  arg35_1 = view_21 = permute_17 = None
    view_22: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_7, [8, 196, 1536]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_5 = torch.ops.aten.split.Tensor(view_22, 768, -1);  view_22 = None
    getitem_22: "f32[8, 196, 768]" = split_5[0]
    getitem_23: "f32[8, 196, 768]" = split_5[1];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_5: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_23)
    mul_22: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_23, sigmoid_5);  getitem_23 = sigmoid_5 = None
    mul_23: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_22, mul_22);  getitem_22 = mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_19: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_23);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_23: "f32[1568, 768]" = torch.ops.aten.view.default(clone_19, [1568, 768]);  clone_19 = None
    permute_18: "f32[768, 384]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_8: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg37_1, view_23, permute_18);  arg37_1 = view_23 = permute_18 = None
    view_24: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_8, [8, 196, 384]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_20: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_20: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_17, clone_20);  add_17 = clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_21: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_21, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 196, 1]" = var_mean_6[0]
    getitem_25: "f32[8, 196, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_6: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_6: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_21, getitem_25);  clone_21 = getitem_25 = None
    mul_24: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    mul_25: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_24, arg38_1);  mul_24 = arg38_1 = None
    add_22: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_25, arg39_1);  mul_25 = arg39_1 = None
    permute_19: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_22, [0, 2, 1]);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_20: "f32[196, 384]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    clone_22: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_25: "f32[3072, 196]" = torch.ops.aten.view.default(clone_22, [3072, 196]);  clone_22 = None
    mm_3: "f32[3072, 384]" = torch.ops.aten.mm.default(view_25, permute_20);  view_25 = permute_20 = None
    view_26: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_3, [8, 384, 384]);  mm_3 = None
    add_23: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_26, arg41_1);  view_26 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_6 = torch.ops.aten.split.Tensor(add_23, 192, -1);  add_23 = None
    getitem_26: "f32[8, 384, 192]" = split_6[0]
    getitem_27: "f32[8, 384, 192]" = split_6[1];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_6: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_27)
    mul_26: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_27, sigmoid_6);  getitem_27 = sigmoid_6 = None
    mul_27: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_26, mul_26);  getitem_26 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_23: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_27);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_27: "f32[3072, 192]" = torch.ops.aten.view.default(clone_23, [3072, 192]);  clone_23 = None
    permute_21: "f32[192, 196]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_9: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg43_1, view_27, permute_21);  arg43_1 = view_27 = permute_21 = None
    view_28: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_9, [8, 384, 196]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_24: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_28);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_22: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_24, [0, 2, 1]);  clone_24 = None
    add_24: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_20, permute_22);  add_20 = permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_25: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_24, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_25, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 196, 1]" = var_mean_7[0]
    getitem_29: "f32[8, 196, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_7: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_7: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_25, getitem_29);  clone_25 = getitem_29 = None
    mul_28: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    mul_29: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_28, arg44_1);  mul_28 = arg44_1 = None
    add_26: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_29, arg45_1);  mul_29 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_29: "f32[1568, 384]" = torch.ops.aten.view.default(add_26, [1568, 384]);  add_26 = None
    permute_23: "f32[384, 1536]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_10: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg47_1, view_29, permute_23);  arg47_1 = view_29 = permute_23 = None
    view_30: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_10, [8, 196, 1536]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_7 = torch.ops.aten.split.Tensor(view_30, 768, -1);  view_30 = None
    getitem_30: "f32[8, 196, 768]" = split_7[0]
    getitem_31: "f32[8, 196, 768]" = split_7[1];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_7: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_31)
    mul_30: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_31, sigmoid_7);  getitem_31 = sigmoid_7 = None
    mul_31: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_30, mul_30);  getitem_30 = mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_26: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_31);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_31: "f32[1568, 768]" = torch.ops.aten.view.default(clone_26, [1568, 768]);  clone_26 = None
    permute_24: "f32[768, 384]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_11: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg49_1, view_31, permute_24);  arg49_1 = view_31 = permute_24 = None
    view_32: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_11, [8, 196, 384]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_27: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_32);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_27: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_24, clone_27);  add_24 = clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_28: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_28, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 196, 1]" = var_mean_8[0]
    getitem_33: "f32[8, 196, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_8: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_28, getitem_33);  clone_28 = getitem_33 = None
    mul_32: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    mul_33: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_32, arg50_1);  mul_32 = arg50_1 = None
    add_29: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_33, arg51_1);  mul_33 = arg51_1 = None
    permute_25: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_29, [0, 2, 1]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_26: "f32[196, 384]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    clone_29: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_33: "f32[3072, 196]" = torch.ops.aten.view.default(clone_29, [3072, 196]);  clone_29 = None
    mm_4: "f32[3072, 384]" = torch.ops.aten.mm.default(view_33, permute_26);  view_33 = permute_26 = None
    view_34: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_4, [8, 384, 384]);  mm_4 = None
    add_30: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_34, arg53_1);  view_34 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_8 = torch.ops.aten.split.Tensor(add_30, 192, -1);  add_30 = None
    getitem_34: "f32[8, 384, 192]" = split_8[0]
    getitem_35: "f32[8, 384, 192]" = split_8[1];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_8: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_35)
    mul_34: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_35, sigmoid_8);  getitem_35 = sigmoid_8 = None
    mul_35: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_34, mul_34);  getitem_34 = mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_30: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_35);  mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_35: "f32[3072, 192]" = torch.ops.aten.view.default(clone_30, [3072, 192]);  clone_30 = None
    permute_27: "f32[192, 196]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_12: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg55_1, view_35, permute_27);  arg55_1 = view_35 = permute_27 = None
    view_36: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_12, [8, 384, 196]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_31: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_28: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_31, [0, 2, 1]);  clone_31 = None
    add_31: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_27, permute_28);  add_27 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_32: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_32, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 196, 1]" = var_mean_9[0]
    getitem_37: "f32[8, 196, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_9: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_9: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_32, getitem_37);  clone_32 = getitem_37 = None
    mul_36: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    mul_37: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_36, arg56_1);  mul_36 = arg56_1 = None
    add_33: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_37, arg57_1);  mul_37 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_37: "f32[1568, 384]" = torch.ops.aten.view.default(add_33, [1568, 384]);  add_33 = None
    permute_29: "f32[384, 1536]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_13: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg59_1, view_37, permute_29);  arg59_1 = view_37 = permute_29 = None
    view_38: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_13, [8, 196, 1536]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_9 = torch.ops.aten.split.Tensor(view_38, 768, -1);  view_38 = None
    getitem_38: "f32[8, 196, 768]" = split_9[0]
    getitem_39: "f32[8, 196, 768]" = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_9: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_39)
    mul_38: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_39, sigmoid_9);  getitem_39 = sigmoid_9 = None
    mul_39: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_38, mul_38);  getitem_38 = mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_33: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_39: "f32[1568, 768]" = torch.ops.aten.view.default(clone_33, [1568, 768]);  clone_33 = None
    permute_30: "f32[768, 384]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    addmm_14: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg61_1, view_39, permute_30);  arg61_1 = view_39 = permute_30 = None
    view_40: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_14, [8, 196, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_34: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_34: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_31, clone_34);  add_31 = clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_35: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_35, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 196, 1]" = var_mean_10[0]
    getitem_41: "f32[8, 196, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_10: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_10: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_35, getitem_41);  clone_35 = getitem_41 = None
    mul_40: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    mul_41: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_40, arg62_1);  mul_40 = arg62_1 = None
    add_36: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_41, arg63_1);  mul_41 = arg63_1 = None
    permute_31: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_36, [0, 2, 1]);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_32: "f32[196, 384]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    clone_36: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_41: "f32[3072, 196]" = torch.ops.aten.view.default(clone_36, [3072, 196]);  clone_36 = None
    mm_5: "f32[3072, 384]" = torch.ops.aten.mm.default(view_41, permute_32);  view_41 = permute_32 = None
    view_42: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_5, [8, 384, 384]);  mm_5 = None
    add_37: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_42, arg65_1);  view_42 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_10 = torch.ops.aten.split.Tensor(add_37, 192, -1);  add_37 = None
    getitem_42: "f32[8, 384, 192]" = split_10[0]
    getitem_43: "f32[8, 384, 192]" = split_10[1];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_10: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_43)
    mul_42: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_43, sigmoid_10);  getitem_43 = sigmoid_10 = None
    mul_43: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_42, mul_42);  getitem_42 = mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_37: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_43);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_43: "f32[3072, 192]" = torch.ops.aten.view.default(clone_37, [3072, 192]);  clone_37 = None
    permute_33: "f32[192, 196]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    addmm_15: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg67_1, view_43, permute_33);  arg67_1 = view_43 = permute_33 = None
    view_44: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_15, [8, 384, 196]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_38: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_34: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_38, [0, 2, 1]);  clone_38 = None
    add_38: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_34, permute_34);  add_34 = permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_39: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_39, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 196, 1]" = var_mean_11[0]
    getitem_45: "f32[8, 196, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_11: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_11: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_39, getitem_45);  clone_39 = getitem_45 = None
    mul_44: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    mul_45: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_44, arg68_1);  mul_44 = arg68_1 = None
    add_40: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_45, arg69_1);  mul_45 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_45: "f32[1568, 384]" = torch.ops.aten.view.default(add_40, [1568, 384]);  add_40 = None
    permute_35: "f32[384, 1536]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_16: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg71_1, view_45, permute_35);  arg71_1 = view_45 = permute_35 = None
    view_46: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 196, 1536]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_11 = torch.ops.aten.split.Tensor(view_46, 768, -1);  view_46 = None
    getitem_46: "f32[8, 196, 768]" = split_11[0]
    getitem_47: "f32[8, 196, 768]" = split_11[1];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_11: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_47)
    mul_46: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_47, sigmoid_11);  getitem_47 = sigmoid_11 = None
    mul_47: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_46, mul_46);  getitem_46 = mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_40: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_47);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_47: "f32[1568, 768]" = torch.ops.aten.view.default(clone_40, [1568, 768]);  clone_40 = None
    permute_36: "f32[768, 384]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_17: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg73_1, view_47, permute_36);  arg73_1 = view_47 = permute_36 = None
    view_48: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_17, [8, 196, 384]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_41: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_41: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_38, clone_41);  add_38 = clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_42: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 196, 1]" = var_mean_12[0]
    getitem_49: "f32[8, 196, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_12: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_42, getitem_49);  clone_42 = getitem_49 = None
    mul_48: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    mul_49: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_48, arg74_1);  mul_48 = arg74_1 = None
    add_43: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_49, arg75_1);  mul_49 = arg75_1 = None
    permute_37: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_43, [0, 2, 1]);  add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_38: "f32[196, 384]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    clone_43: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_49: "f32[3072, 196]" = torch.ops.aten.view.default(clone_43, [3072, 196]);  clone_43 = None
    mm_6: "f32[3072, 384]" = torch.ops.aten.mm.default(view_49, permute_38);  view_49 = permute_38 = None
    view_50: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_6, [8, 384, 384]);  mm_6 = None
    add_44: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_50, arg77_1);  view_50 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_12 = torch.ops.aten.split.Tensor(add_44, 192, -1);  add_44 = None
    getitem_50: "f32[8, 384, 192]" = split_12[0]
    getitem_51: "f32[8, 384, 192]" = split_12[1];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_12: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_51)
    mul_50: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_51, sigmoid_12);  getitem_51 = sigmoid_12 = None
    mul_51: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_50, mul_50);  getitem_50 = mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_44: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_51);  mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_51: "f32[3072, 192]" = torch.ops.aten.view.default(clone_44, [3072, 192]);  clone_44 = None
    permute_39: "f32[192, 196]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_18: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg79_1, view_51, permute_39);  arg79_1 = view_51 = permute_39 = None
    view_52: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_18, [8, 384, 196]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_45: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_52);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_40: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_45, [0, 2, 1]);  clone_45 = None
    add_45: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_41, permute_40);  add_41 = permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_46: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_45, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_46, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 196, 1]" = var_mean_13[0]
    getitem_53: "f32[8, 196, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_13: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_13: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_46, getitem_53);  clone_46 = getitem_53 = None
    mul_52: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    mul_53: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_52, arg80_1);  mul_52 = arg80_1 = None
    add_47: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_53, arg81_1);  mul_53 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_53: "f32[1568, 384]" = torch.ops.aten.view.default(add_47, [1568, 384]);  add_47 = None
    permute_41: "f32[384, 1536]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm_19: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg83_1, view_53, permute_41);  arg83_1 = view_53 = permute_41 = None
    view_54: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_19, [8, 196, 1536]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_13 = torch.ops.aten.split.Tensor(view_54, 768, -1);  view_54 = None
    getitem_54: "f32[8, 196, 768]" = split_13[0]
    getitem_55: "f32[8, 196, 768]" = split_13[1];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_13: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_55)
    mul_54: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_55, sigmoid_13);  getitem_55 = sigmoid_13 = None
    mul_55: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_54, mul_54);  getitem_54 = mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_47: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_55);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_55: "f32[1568, 768]" = torch.ops.aten.view.default(clone_47, [1568, 768]);  clone_47 = None
    permute_42: "f32[768, 384]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_20: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg85_1, view_55, permute_42);  arg85_1 = view_55 = permute_42 = None
    view_56: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_20, [8, 196, 384]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_48: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_48: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_45, clone_48);  add_45 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_49: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_49, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 196, 1]" = var_mean_14[0]
    getitem_57: "f32[8, 196, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_14: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_14: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_49, getitem_57);  clone_49 = getitem_57 = None
    mul_56: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    mul_57: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_56, arg86_1);  mul_56 = arg86_1 = None
    add_50: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_57, arg87_1);  mul_57 = arg87_1 = None
    permute_43: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_50, [0, 2, 1]);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_44: "f32[196, 384]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    clone_50: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_57: "f32[3072, 196]" = torch.ops.aten.view.default(clone_50, [3072, 196]);  clone_50 = None
    mm_7: "f32[3072, 384]" = torch.ops.aten.mm.default(view_57, permute_44);  view_57 = permute_44 = None
    view_58: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_7, [8, 384, 384]);  mm_7 = None
    add_51: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_58, arg89_1);  view_58 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_14 = torch.ops.aten.split.Tensor(add_51, 192, -1);  add_51 = None
    getitem_58: "f32[8, 384, 192]" = split_14[0]
    getitem_59: "f32[8, 384, 192]" = split_14[1];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_14: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_59)
    mul_58: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_59, sigmoid_14);  getitem_59 = sigmoid_14 = None
    mul_59: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_58, mul_58);  getitem_58 = mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_51: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_59);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_59: "f32[3072, 192]" = torch.ops.aten.view.default(clone_51, [3072, 192]);  clone_51 = None
    permute_45: "f32[192, 196]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_21: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg91_1, view_59, permute_45);  arg91_1 = view_59 = permute_45 = None
    view_60: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_21, [8, 384, 196]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_52: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_46: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_52, [0, 2, 1]);  clone_52 = None
    add_52: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_48, permute_46);  add_48 = permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_53: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_52, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_53, [2], correction = 0, keepdim = True)
    getitem_60: "f32[8, 196, 1]" = var_mean_15[0]
    getitem_61: "f32[8, 196, 1]" = var_mean_15[1];  var_mean_15 = None
    add_53: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_15: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_15: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_53, getitem_61);  clone_53 = getitem_61 = None
    mul_60: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    mul_61: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_60, arg92_1);  mul_60 = arg92_1 = None
    add_54: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_61, arg93_1);  mul_61 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_61: "f32[1568, 384]" = torch.ops.aten.view.default(add_54, [1568, 384]);  add_54 = None
    permute_47: "f32[384, 1536]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_22: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg95_1, view_61, permute_47);  arg95_1 = view_61 = permute_47 = None
    view_62: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_22, [8, 196, 1536]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_15 = torch.ops.aten.split.Tensor(view_62, 768, -1);  view_62 = None
    getitem_62: "f32[8, 196, 768]" = split_15[0]
    getitem_63: "f32[8, 196, 768]" = split_15[1];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_15: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_63)
    mul_62: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_63, sigmoid_15);  getitem_63 = sigmoid_15 = None
    mul_63: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_62, mul_62);  getitem_62 = mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_54: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_63);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_63: "f32[1568, 768]" = torch.ops.aten.view.default(clone_54, [1568, 768]);  clone_54 = None
    permute_48: "f32[768, 384]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_23: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg97_1, view_63, permute_48);  arg97_1 = view_63 = permute_48 = None
    view_64: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_23, [8, 196, 384]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_55: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_64);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_55: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_52, clone_55);  add_52 = clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_56: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_55, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_56, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 196, 1]" = var_mean_16[0]
    getitem_65: "f32[8, 196, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_16: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_16: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_56, getitem_65);  clone_56 = getitem_65 = None
    mul_64: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    mul_65: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_64, arg98_1);  mul_64 = arg98_1 = None
    add_57: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_65, arg99_1);  mul_65 = arg99_1 = None
    permute_49: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_57, [0, 2, 1]);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_50: "f32[196, 384]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    clone_57: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_65: "f32[3072, 196]" = torch.ops.aten.view.default(clone_57, [3072, 196]);  clone_57 = None
    mm_8: "f32[3072, 384]" = torch.ops.aten.mm.default(view_65, permute_50);  view_65 = permute_50 = None
    view_66: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_8, [8, 384, 384]);  mm_8 = None
    add_58: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_66, arg101_1);  view_66 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_16 = torch.ops.aten.split.Tensor(add_58, 192, -1);  add_58 = None
    getitem_66: "f32[8, 384, 192]" = split_16[0]
    getitem_67: "f32[8, 384, 192]" = split_16[1];  split_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_16: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_67)
    mul_66: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_67, sigmoid_16);  getitem_67 = sigmoid_16 = None
    mul_67: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_66, mul_66);  getitem_66 = mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_58: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_67);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_67: "f32[3072, 192]" = torch.ops.aten.view.default(clone_58, [3072, 192]);  clone_58 = None
    permute_51: "f32[192, 196]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_24: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg103_1, view_67, permute_51);  arg103_1 = view_67 = permute_51 = None
    view_68: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_24, [8, 384, 196]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_59: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_52: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_59, [0, 2, 1]);  clone_59 = None
    add_59: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_55, permute_52);  add_55 = permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_60: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_60, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 196, 1]" = var_mean_17[0]
    getitem_69: "f32[8, 196, 1]" = var_mean_17[1];  var_mean_17 = None
    add_60: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_17: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_17: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_60, getitem_69);  clone_60 = getitem_69 = None
    mul_68: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    mul_69: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_68, arg104_1);  mul_68 = arg104_1 = None
    add_61: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_69, arg105_1);  mul_69 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_69: "f32[1568, 384]" = torch.ops.aten.view.default(add_61, [1568, 384]);  add_61 = None
    permute_53: "f32[384, 1536]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_25: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg107_1, view_69, permute_53);  arg107_1 = view_69 = permute_53 = None
    view_70: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_25, [8, 196, 1536]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_17 = torch.ops.aten.split.Tensor(view_70, 768, -1);  view_70 = None
    getitem_70: "f32[8, 196, 768]" = split_17[0]
    getitem_71: "f32[8, 196, 768]" = split_17[1];  split_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_17: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_71)
    mul_70: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_71, sigmoid_17);  getitem_71 = sigmoid_17 = None
    mul_71: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_70, mul_70);  getitem_70 = mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_61: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_71);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_71: "f32[1568, 768]" = torch.ops.aten.view.default(clone_61, [1568, 768]);  clone_61 = None
    permute_54: "f32[768, 384]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_26: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg109_1, view_71, permute_54);  arg109_1 = view_71 = permute_54 = None
    view_72: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_26, [8, 196, 384]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_62: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_62: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_59, clone_62);  add_59 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_63: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_62, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_63, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 196, 1]" = var_mean_18[0]
    getitem_73: "f32[8, 196, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_18: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_18: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_63, getitem_73);  clone_63 = getitem_73 = None
    mul_72: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    mul_73: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_72, arg110_1);  mul_72 = arg110_1 = None
    add_64: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_73, arg111_1);  mul_73 = arg111_1 = None
    permute_55: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_56: "f32[196, 384]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    clone_64: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    view_73: "f32[3072, 196]" = torch.ops.aten.view.default(clone_64, [3072, 196]);  clone_64 = None
    mm_9: "f32[3072, 384]" = torch.ops.aten.mm.default(view_73, permute_56);  view_73 = permute_56 = None
    view_74: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_9, [8, 384, 384]);  mm_9 = None
    add_65: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_74, arg113_1);  view_74 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_18 = torch.ops.aten.split.Tensor(add_65, 192, -1);  add_65 = None
    getitem_74: "f32[8, 384, 192]" = split_18[0]
    getitem_75: "f32[8, 384, 192]" = split_18[1];  split_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_18: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_75)
    mul_74: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_75, sigmoid_18);  getitem_75 = sigmoid_18 = None
    mul_75: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_74, mul_74);  getitem_74 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_65: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_75);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_75: "f32[3072, 192]" = torch.ops.aten.view.default(clone_65, [3072, 192]);  clone_65 = None
    permute_57: "f32[192, 196]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_27: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg115_1, view_75, permute_57);  arg115_1 = view_75 = permute_57 = None
    view_76: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_27, [8, 384, 196]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_66: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_76);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_58: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_66, [0, 2, 1]);  clone_66 = None
    add_66: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_62, permute_58);  add_62 = permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_67: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 196, 1]" = var_mean_19[0]
    getitem_77: "f32[8, 196, 1]" = var_mean_19[1];  var_mean_19 = None
    add_67: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_19: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_19: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_67, getitem_77);  clone_67 = getitem_77 = None
    mul_76: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    mul_77: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_76, arg116_1);  mul_76 = arg116_1 = None
    add_68: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_77, arg117_1);  mul_77 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_77: "f32[1568, 384]" = torch.ops.aten.view.default(add_68, [1568, 384]);  add_68 = None
    permute_59: "f32[384, 1536]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_28: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg119_1, view_77, permute_59);  arg119_1 = view_77 = permute_59 = None
    view_78: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 196, 1536]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_19 = torch.ops.aten.split.Tensor(view_78, 768, -1);  view_78 = None
    getitem_78: "f32[8, 196, 768]" = split_19[0]
    getitem_79: "f32[8, 196, 768]" = split_19[1];  split_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_19: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_79)
    mul_78: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_79, sigmoid_19);  getitem_79 = sigmoid_19 = None
    mul_79: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_78, mul_78);  getitem_78 = mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_68: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_79: "f32[1568, 768]" = torch.ops.aten.view.default(clone_68, [1568, 768]);  clone_68 = None
    permute_60: "f32[768, 384]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_29: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg121_1, view_79, permute_60);  arg121_1 = view_79 = permute_60 = None
    view_80: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_29, [8, 196, 384]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_69: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_69: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_66, clone_69);  add_66 = clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_70: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_70, [2], correction = 0, keepdim = True)
    getitem_80: "f32[8, 196, 1]" = var_mean_20[0]
    getitem_81: "f32[8, 196, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_20: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_20: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_70, getitem_81);  clone_70 = getitem_81 = None
    mul_80: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    mul_81: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_80, arg122_1);  mul_80 = arg122_1 = None
    add_71: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_81, arg123_1);  mul_81 = arg123_1 = None
    permute_61: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_71, [0, 2, 1]);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_62: "f32[196, 384]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    clone_71: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    view_81: "f32[3072, 196]" = torch.ops.aten.view.default(clone_71, [3072, 196]);  clone_71 = None
    mm_10: "f32[3072, 384]" = torch.ops.aten.mm.default(view_81, permute_62);  view_81 = permute_62 = None
    view_82: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_10, [8, 384, 384]);  mm_10 = None
    add_72: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_82, arg125_1);  view_82 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_20 = torch.ops.aten.split.Tensor(add_72, 192, -1);  add_72 = None
    getitem_82: "f32[8, 384, 192]" = split_20[0]
    getitem_83: "f32[8, 384, 192]" = split_20[1];  split_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_20: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_83)
    mul_82: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_83, sigmoid_20);  getitem_83 = sigmoid_20 = None
    mul_83: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_82, mul_82);  getitem_82 = mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_72: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_83: "f32[3072, 192]" = torch.ops.aten.view.default(clone_72, [3072, 192]);  clone_72 = None
    permute_63: "f32[192, 196]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_30: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg127_1, view_83, permute_63);  arg127_1 = view_83 = permute_63 = None
    view_84: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_30, [8, 384, 196]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_73: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_64: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_73, [0, 2, 1]);  clone_73 = None
    add_73: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_69, permute_64);  add_69 = permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_74: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_74, [2], correction = 0, keepdim = True)
    getitem_84: "f32[8, 196, 1]" = var_mean_21[0]
    getitem_85: "f32[8, 196, 1]" = var_mean_21[1];  var_mean_21 = None
    add_74: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_21: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_21: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_74, getitem_85);  clone_74 = getitem_85 = None
    mul_84: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    mul_85: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_84, arg128_1);  mul_84 = arg128_1 = None
    add_75: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_85, arg129_1);  mul_85 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_85: "f32[1568, 384]" = torch.ops.aten.view.default(add_75, [1568, 384]);  add_75 = None
    permute_65: "f32[384, 1536]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_31: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg131_1, view_85, permute_65);  arg131_1 = view_85 = permute_65 = None
    view_86: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_31, [8, 196, 1536]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_21 = torch.ops.aten.split.Tensor(view_86, 768, -1);  view_86 = None
    getitem_86: "f32[8, 196, 768]" = split_21[0]
    getitem_87: "f32[8, 196, 768]" = split_21[1];  split_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_21: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_87)
    mul_86: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_87, sigmoid_21);  getitem_87 = sigmoid_21 = None
    mul_87: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_86, mul_86);  getitem_86 = mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_75: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_87);  mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_87: "f32[1568, 768]" = torch.ops.aten.view.default(clone_75, [1568, 768]);  clone_75 = None
    permute_66: "f32[768, 384]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_32: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg133_1, view_87, permute_66);  arg133_1 = view_87 = permute_66 = None
    view_88: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_32, [8, 196, 384]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_76: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_88);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_76: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_73, clone_76);  add_73 = clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_77: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_77, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 196, 1]" = var_mean_22[0]
    getitem_89: "f32[8, 196, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_22: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_22: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_77, getitem_89);  clone_77 = getitem_89 = None
    mul_88: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    mul_89: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_88, arg134_1);  mul_88 = arg134_1 = None
    add_78: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_89, arg135_1);  mul_89 = arg135_1 = None
    permute_67: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_78, [0, 2, 1]);  add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_68: "f32[196, 384]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    clone_78: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_89: "f32[3072, 196]" = torch.ops.aten.view.default(clone_78, [3072, 196]);  clone_78 = None
    mm_11: "f32[3072, 384]" = torch.ops.aten.mm.default(view_89, permute_68);  view_89 = permute_68 = None
    view_90: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_11, [8, 384, 384]);  mm_11 = None
    add_79: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_90, arg137_1);  view_90 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_22 = torch.ops.aten.split.Tensor(add_79, 192, -1);  add_79 = None
    getitem_90: "f32[8, 384, 192]" = split_22[0]
    getitem_91: "f32[8, 384, 192]" = split_22[1];  split_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_22: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_91)
    mul_90: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_91, sigmoid_22);  getitem_91 = sigmoid_22 = None
    mul_91: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_90, mul_90);  getitem_90 = mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_79: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_91);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_91: "f32[3072, 192]" = torch.ops.aten.view.default(clone_79, [3072, 192]);  clone_79 = None
    permute_69: "f32[192, 196]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_33: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg139_1, view_91, permute_69);  arg139_1 = view_91 = permute_69 = None
    view_92: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_33, [8, 384, 196]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_80: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_92);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_70: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_80, [0, 2, 1]);  clone_80 = None
    add_80: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_76, permute_70);  add_76 = permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_81: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_81, [2], correction = 0, keepdim = True)
    getitem_92: "f32[8, 196, 1]" = var_mean_23[0]
    getitem_93: "f32[8, 196, 1]" = var_mean_23[1];  var_mean_23 = None
    add_81: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_23: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_23: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_81, getitem_93);  clone_81 = getitem_93 = None
    mul_92: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    mul_93: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_92, arg140_1);  mul_92 = arg140_1 = None
    add_82: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_93, arg141_1);  mul_93 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_93: "f32[1568, 384]" = torch.ops.aten.view.default(add_82, [1568, 384]);  add_82 = None
    permute_71: "f32[384, 1536]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_34: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg143_1, view_93, permute_71);  arg143_1 = view_93 = permute_71 = None
    view_94: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_34, [8, 196, 1536]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_23 = torch.ops.aten.split.Tensor(view_94, 768, -1);  view_94 = None
    getitem_94: "f32[8, 196, 768]" = split_23[0]
    getitem_95: "f32[8, 196, 768]" = split_23[1];  split_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_23: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_95)
    mul_94: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_95, sigmoid_23);  getitem_95 = sigmoid_23 = None
    mul_95: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_94, mul_94);  getitem_94 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_82: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_95);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_95: "f32[1568, 768]" = torch.ops.aten.view.default(clone_82, [1568, 768]);  clone_82 = None
    permute_72: "f32[768, 384]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_35: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg145_1, view_95, permute_72);  arg145_1 = view_95 = permute_72 = None
    view_96: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_35, [8, 196, 384]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_83: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_83: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_80, clone_83);  add_80 = clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_84: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_83, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 196, 1]" = var_mean_24[0]
    getitem_97: "f32[8, 196, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_24: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_24: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_84, getitem_97);  clone_84 = getitem_97 = None
    mul_96: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    mul_97: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_96, arg146_1);  mul_96 = arg146_1 = None
    add_85: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_97, arg147_1);  mul_97 = arg147_1 = None
    permute_73: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_85, [0, 2, 1]);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_74: "f32[196, 384]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    clone_85: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_97: "f32[3072, 196]" = torch.ops.aten.view.default(clone_85, [3072, 196]);  clone_85 = None
    mm_12: "f32[3072, 384]" = torch.ops.aten.mm.default(view_97, permute_74);  view_97 = permute_74 = None
    view_98: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_12, [8, 384, 384]);  mm_12 = None
    add_86: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_98, arg149_1);  view_98 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_24 = torch.ops.aten.split.Tensor(add_86, 192, -1);  add_86 = None
    getitem_98: "f32[8, 384, 192]" = split_24[0]
    getitem_99: "f32[8, 384, 192]" = split_24[1];  split_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_24: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_99)
    mul_98: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_99, sigmoid_24);  getitem_99 = sigmoid_24 = None
    mul_99: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_98, mul_98);  getitem_98 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_86: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_99);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_99: "f32[3072, 192]" = torch.ops.aten.view.default(clone_86, [3072, 192]);  clone_86 = None
    permute_75: "f32[192, 196]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_36: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg151_1, view_99, permute_75);  arg151_1 = view_99 = permute_75 = None
    view_100: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_36, [8, 384, 196]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_87: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_76: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_87, [0, 2, 1]);  clone_87 = None
    add_87: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_83, permute_76);  add_83 = permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_88: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_88, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 196, 1]" = var_mean_25[0]
    getitem_101: "f32[8, 196, 1]" = var_mean_25[1];  var_mean_25 = None
    add_88: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
    rsqrt_25: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_25: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_88, getitem_101);  clone_88 = getitem_101 = None
    mul_100: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
    mul_101: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_100, arg152_1);  mul_100 = arg152_1 = None
    add_89: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_101, arg153_1);  mul_101 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_101: "f32[1568, 384]" = torch.ops.aten.view.default(add_89, [1568, 384]);  add_89 = None
    permute_77: "f32[384, 1536]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_37: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg155_1, view_101, permute_77);  arg155_1 = view_101 = permute_77 = None
    view_102: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_37, [8, 196, 1536]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_25 = torch.ops.aten.split.Tensor(view_102, 768, -1);  view_102 = None
    getitem_102: "f32[8, 196, 768]" = split_25[0]
    getitem_103: "f32[8, 196, 768]" = split_25[1];  split_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_25: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_103)
    mul_102: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_103, sigmoid_25);  getitem_103 = sigmoid_25 = None
    mul_103: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_102, mul_102);  getitem_102 = mul_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_89: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_103);  mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_103: "f32[1568, 768]" = torch.ops.aten.view.default(clone_89, [1568, 768]);  clone_89 = None
    permute_78: "f32[768, 384]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_38: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg157_1, view_103, permute_78);  arg157_1 = view_103 = permute_78 = None
    view_104: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_38, [8, 196, 384]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_90: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_90: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_87, clone_90);  add_87 = clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_91: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_91, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 196, 1]" = var_mean_26[0]
    getitem_105: "f32[8, 196, 1]" = var_mean_26[1];  var_mean_26 = None
    add_91: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_26: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_26: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_91, getitem_105);  clone_91 = getitem_105 = None
    mul_104: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
    mul_105: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_104, arg158_1);  mul_104 = arg158_1 = None
    add_92: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_105, arg159_1);  mul_105 = arg159_1 = None
    permute_79: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_92, [0, 2, 1]);  add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_80: "f32[196, 384]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    clone_92: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    view_105: "f32[3072, 196]" = torch.ops.aten.view.default(clone_92, [3072, 196]);  clone_92 = None
    mm_13: "f32[3072, 384]" = torch.ops.aten.mm.default(view_105, permute_80);  view_105 = permute_80 = None
    view_106: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_13, [8, 384, 384]);  mm_13 = None
    add_93: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_106, arg161_1);  view_106 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_26 = torch.ops.aten.split.Tensor(add_93, 192, -1);  add_93 = None
    getitem_106: "f32[8, 384, 192]" = split_26[0]
    getitem_107: "f32[8, 384, 192]" = split_26[1];  split_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_26: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_107)
    mul_106: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_107, sigmoid_26);  getitem_107 = sigmoid_26 = None
    mul_107: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_106, mul_106);  getitem_106 = mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_93: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_107);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_107: "f32[3072, 192]" = torch.ops.aten.view.default(clone_93, [3072, 192]);  clone_93 = None
    permute_81: "f32[192, 196]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    addmm_39: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg163_1, view_107, permute_81);  arg163_1 = view_107 = permute_81 = None
    view_108: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_39, [8, 384, 196]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_94: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_82: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_94, [0, 2, 1]);  clone_94 = None
    add_94: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_90, permute_82);  add_90 = permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_95: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_95, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 196, 1]" = var_mean_27[0]
    getitem_109: "f32[8, 196, 1]" = var_mean_27[1];  var_mean_27 = None
    add_95: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_27: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_27: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_95, getitem_109);  clone_95 = getitem_109 = None
    mul_108: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
    mul_109: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_108, arg164_1);  mul_108 = arg164_1 = None
    add_96: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_109, arg165_1);  mul_109 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_109: "f32[1568, 384]" = torch.ops.aten.view.default(add_96, [1568, 384]);  add_96 = None
    permute_83: "f32[384, 1536]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_40: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg167_1, view_109, permute_83);  arg167_1 = view_109 = permute_83 = None
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_40, [8, 196, 1536]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_27 = torch.ops.aten.split.Tensor(view_110, 768, -1);  view_110 = None
    getitem_110: "f32[8, 196, 768]" = split_27[0]
    getitem_111: "f32[8, 196, 768]" = split_27[1];  split_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_27: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_111)
    mul_110: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_111, sigmoid_27);  getitem_111 = sigmoid_27 = None
    mul_111: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_110, mul_110);  getitem_110 = mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_96: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_111);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_111: "f32[1568, 768]" = torch.ops.aten.view.default(clone_96, [1568, 768]);  clone_96 = None
    permute_84: "f32[768, 384]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_41: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg169_1, view_111, permute_84);  arg169_1 = view_111 = permute_84 = None
    view_112: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_41, [8, 196, 384]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_97: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_112);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_97: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_94, clone_97);  add_94 = clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_98: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_98, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 196, 1]" = var_mean_28[0]
    getitem_113: "f32[8, 196, 1]" = var_mean_28[1];  var_mean_28 = None
    add_98: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_28: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_28: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_98, getitem_113);  clone_98 = getitem_113 = None
    mul_112: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
    mul_113: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_112, arg170_1);  mul_112 = arg170_1 = None
    add_99: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_113, arg171_1);  mul_113 = arg171_1 = None
    permute_85: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_99, [0, 2, 1]);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_86: "f32[196, 384]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    clone_99: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_113: "f32[3072, 196]" = torch.ops.aten.view.default(clone_99, [3072, 196]);  clone_99 = None
    mm_14: "f32[3072, 384]" = torch.ops.aten.mm.default(view_113, permute_86);  view_113 = permute_86 = None
    view_114: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_14, [8, 384, 384]);  mm_14 = None
    add_100: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_114, arg173_1);  view_114 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_28 = torch.ops.aten.split.Tensor(add_100, 192, -1);  add_100 = None
    getitem_114: "f32[8, 384, 192]" = split_28[0]
    getitem_115: "f32[8, 384, 192]" = split_28[1];  split_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_28: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_115)
    mul_114: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_115, sigmoid_28);  getitem_115 = sigmoid_28 = None
    mul_115: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_114, mul_114);  getitem_114 = mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_100: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_115);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_115: "f32[3072, 192]" = torch.ops.aten.view.default(clone_100, [3072, 192]);  clone_100 = None
    permute_87: "f32[192, 196]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_42: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg175_1, view_115, permute_87);  arg175_1 = view_115 = permute_87 = None
    view_116: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_42, [8, 384, 196]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_101: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_88: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_101, [0, 2, 1]);  clone_101 = None
    add_101: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_97, permute_88);  add_97 = permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_102: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_101, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_102, [2], correction = 0, keepdim = True)
    getitem_116: "f32[8, 196, 1]" = var_mean_29[0]
    getitem_117: "f32[8, 196, 1]" = var_mean_29[1];  var_mean_29 = None
    add_102: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-06);  getitem_116 = None
    rsqrt_29: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_29: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_102, getitem_117);  clone_102 = getitem_117 = None
    mul_116: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
    mul_117: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_116, arg176_1);  mul_116 = arg176_1 = None
    add_103: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_117, arg177_1);  mul_117 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_117: "f32[1568, 384]" = torch.ops.aten.view.default(add_103, [1568, 384]);  add_103 = None
    permute_89: "f32[384, 1536]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_43: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg179_1, view_117, permute_89);  arg179_1 = view_117 = permute_89 = None
    view_118: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_43, [8, 196, 1536]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_29 = torch.ops.aten.split.Tensor(view_118, 768, -1);  view_118 = None
    getitem_118: "f32[8, 196, 768]" = split_29[0]
    getitem_119: "f32[8, 196, 768]" = split_29[1];  split_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_29: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_119)
    mul_118: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_119, sigmoid_29);  getitem_119 = sigmoid_29 = None
    mul_119: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_118, mul_118);  getitem_118 = mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_103: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_119);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_119: "f32[1568, 768]" = torch.ops.aten.view.default(clone_103, [1568, 768]);  clone_103 = None
    permute_90: "f32[768, 384]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_44: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg181_1, view_119, permute_90);  arg181_1 = view_119 = permute_90 = None
    view_120: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_44, [8, 196, 384]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_104: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_104: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_101, clone_104);  add_101 = clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_105: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_105, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 196, 1]" = var_mean_30[0]
    getitem_121: "f32[8, 196, 1]" = var_mean_30[1];  var_mean_30 = None
    add_105: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_30: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_30: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_105, getitem_121);  clone_105 = getitem_121 = None
    mul_120: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
    mul_121: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_120, arg182_1);  mul_120 = arg182_1 = None
    add_106: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_121, arg183_1);  mul_121 = arg183_1 = None
    permute_91: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_106, [0, 2, 1]);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_92: "f32[196, 384]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    clone_106: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    view_121: "f32[3072, 196]" = torch.ops.aten.view.default(clone_106, [3072, 196]);  clone_106 = None
    mm_15: "f32[3072, 384]" = torch.ops.aten.mm.default(view_121, permute_92);  view_121 = permute_92 = None
    view_122: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_15, [8, 384, 384]);  mm_15 = None
    add_107: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_122, arg185_1);  view_122 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_30 = torch.ops.aten.split.Tensor(add_107, 192, -1);  add_107 = None
    getitem_122: "f32[8, 384, 192]" = split_30[0]
    getitem_123: "f32[8, 384, 192]" = split_30[1];  split_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_30: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_123)
    mul_122: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_123, sigmoid_30);  getitem_123 = sigmoid_30 = None
    mul_123: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_122, mul_122);  getitem_122 = mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_107: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_123);  mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_123: "f32[3072, 192]" = torch.ops.aten.view.default(clone_107, [3072, 192]);  clone_107 = None
    permute_93: "f32[192, 196]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_45: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg187_1, view_123, permute_93);  arg187_1 = view_123 = permute_93 = None
    view_124: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_45, [8, 384, 196]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_108: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_124);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_94: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_108, [0, 2, 1]);  clone_108 = None
    add_108: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_104, permute_94);  add_104 = permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_109: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_108, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_109, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 196, 1]" = var_mean_31[0]
    getitem_125: "f32[8, 196, 1]" = var_mean_31[1];  var_mean_31 = None
    add_109: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
    rsqrt_31: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_31: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_109, getitem_125);  clone_109 = getitem_125 = None
    mul_124: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
    mul_125: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_124, arg188_1);  mul_124 = arg188_1 = None
    add_110: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_125, arg189_1);  mul_125 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_125: "f32[1568, 384]" = torch.ops.aten.view.default(add_110, [1568, 384]);  add_110 = None
    permute_95: "f32[384, 1536]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_46: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg191_1, view_125, permute_95);  arg191_1 = view_125 = permute_95 = None
    view_126: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_46, [8, 196, 1536]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_31 = torch.ops.aten.split.Tensor(view_126, 768, -1);  view_126 = None
    getitem_126: "f32[8, 196, 768]" = split_31[0]
    getitem_127: "f32[8, 196, 768]" = split_31[1];  split_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_31: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_127)
    mul_126: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_127, sigmoid_31);  getitem_127 = sigmoid_31 = None
    mul_127: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_126, mul_126);  getitem_126 = mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_110: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_127);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_127: "f32[1568, 768]" = torch.ops.aten.view.default(clone_110, [1568, 768]);  clone_110 = None
    permute_96: "f32[768, 384]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_47: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg193_1, view_127, permute_96);  arg193_1 = view_127 = permute_96 = None
    view_128: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_47, [8, 196, 384]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_111: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_128);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_111: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_108, clone_111);  add_108 = clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_112: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_112, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 196, 1]" = var_mean_32[0]
    getitem_129: "f32[8, 196, 1]" = var_mean_32[1];  var_mean_32 = None
    add_112: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
    rsqrt_32: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_32: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_112, getitem_129);  clone_112 = getitem_129 = None
    mul_128: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
    mul_129: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_128, arg194_1);  mul_128 = arg194_1 = None
    add_113: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_129, arg195_1);  mul_129 = arg195_1 = None
    permute_97: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_113, [0, 2, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_98: "f32[196, 384]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    clone_113: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    view_129: "f32[3072, 196]" = torch.ops.aten.view.default(clone_113, [3072, 196]);  clone_113 = None
    mm_16: "f32[3072, 384]" = torch.ops.aten.mm.default(view_129, permute_98);  view_129 = permute_98 = None
    view_130: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_16, [8, 384, 384]);  mm_16 = None
    add_114: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_130, arg197_1);  view_130 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_32 = torch.ops.aten.split.Tensor(add_114, 192, -1);  add_114 = None
    getitem_130: "f32[8, 384, 192]" = split_32[0]
    getitem_131: "f32[8, 384, 192]" = split_32[1];  split_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_32: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_131)
    mul_130: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_131, sigmoid_32);  getitem_131 = sigmoid_32 = None
    mul_131: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_130, mul_130);  getitem_130 = mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_114: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_131);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_131: "f32[3072, 192]" = torch.ops.aten.view.default(clone_114, [3072, 192]);  clone_114 = None
    permute_99: "f32[192, 196]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    addmm_48: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg199_1, view_131, permute_99);  arg199_1 = view_131 = permute_99 = None
    view_132: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_48, [8, 384, 196]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_115: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_132);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_100: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_115, [0, 2, 1]);  clone_115 = None
    add_115: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_111, permute_100);  add_111 = permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_116: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_116, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 196, 1]" = var_mean_33[0]
    getitem_133: "f32[8, 196, 1]" = var_mean_33[1];  var_mean_33 = None
    add_116: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_33: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_33: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_116, getitem_133);  clone_116 = getitem_133 = None
    mul_132: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
    mul_133: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_132, arg200_1);  mul_132 = arg200_1 = None
    add_117: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_133, arg201_1);  mul_133 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_133: "f32[1568, 384]" = torch.ops.aten.view.default(add_117, [1568, 384]);  add_117 = None
    permute_101: "f32[384, 1536]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
    addmm_49: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg203_1, view_133, permute_101);  arg203_1 = view_133 = permute_101 = None
    view_134: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_49, [8, 196, 1536]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_33 = torch.ops.aten.split.Tensor(view_134, 768, -1);  view_134 = None
    getitem_134: "f32[8, 196, 768]" = split_33[0]
    getitem_135: "f32[8, 196, 768]" = split_33[1];  split_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_33: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_135)
    mul_134: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_135, sigmoid_33);  getitem_135 = sigmoid_33 = None
    mul_135: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_134, mul_134);  getitem_134 = mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_117: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_135);  mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_135: "f32[1568, 768]" = torch.ops.aten.view.default(clone_117, [1568, 768]);  clone_117 = None
    permute_102: "f32[768, 384]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    addmm_50: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg205_1, view_135, permute_102);  arg205_1 = view_135 = permute_102 = None
    view_136: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_50, [8, 196, 384]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_118: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_136);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_118: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_115, clone_118);  add_115 = clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_119: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_118, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_119, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 196, 1]" = var_mean_34[0]
    getitem_137: "f32[8, 196, 1]" = var_mean_34[1];  var_mean_34 = None
    add_119: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
    rsqrt_34: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_34: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_119, getitem_137);  clone_119 = getitem_137 = None
    mul_136: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
    mul_137: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_136, arg206_1);  mul_136 = arg206_1 = None
    add_120: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_137, arg207_1);  mul_137 = arg207_1 = None
    permute_103: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_120, [0, 2, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_104: "f32[196, 384]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    clone_120: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    view_137: "f32[3072, 196]" = torch.ops.aten.view.default(clone_120, [3072, 196]);  clone_120 = None
    mm_17: "f32[3072, 384]" = torch.ops.aten.mm.default(view_137, permute_104);  view_137 = permute_104 = None
    view_138: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_17, [8, 384, 384]);  mm_17 = None
    add_121: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_138, arg209_1);  view_138 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_34 = torch.ops.aten.split.Tensor(add_121, 192, -1);  add_121 = None
    getitem_138: "f32[8, 384, 192]" = split_34[0]
    getitem_139: "f32[8, 384, 192]" = split_34[1];  split_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_34: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_139)
    mul_138: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_139, sigmoid_34);  getitem_139 = sigmoid_34 = None
    mul_139: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_138, mul_138);  getitem_138 = mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_121: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_139);  mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_139: "f32[3072, 192]" = torch.ops.aten.view.default(clone_121, [3072, 192]);  clone_121 = None
    permute_105: "f32[192, 196]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    addmm_51: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg211_1, view_139, permute_105);  arg211_1 = view_139 = permute_105 = None
    view_140: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_51, [8, 384, 196]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_122: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_106: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_122, [0, 2, 1]);  clone_122 = None
    add_122: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_118, permute_106);  add_118 = permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_123: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_123, [2], correction = 0, keepdim = True)
    getitem_140: "f32[8, 196, 1]" = var_mean_35[0]
    getitem_141: "f32[8, 196, 1]" = var_mean_35[1];  var_mean_35 = None
    add_123: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
    rsqrt_35: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_35: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_123, getitem_141);  clone_123 = getitem_141 = None
    mul_140: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
    mul_141: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_140, arg212_1);  mul_140 = arg212_1 = None
    add_124: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_141, arg213_1);  mul_141 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_141: "f32[1568, 384]" = torch.ops.aten.view.default(add_124, [1568, 384]);  add_124 = None
    permute_107: "f32[384, 1536]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    addmm_52: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg215_1, view_141, permute_107);  arg215_1 = view_141 = permute_107 = None
    view_142: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_52, [8, 196, 1536]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_35 = torch.ops.aten.split.Tensor(view_142, 768, -1);  view_142 = None
    getitem_142: "f32[8, 196, 768]" = split_35[0]
    getitem_143: "f32[8, 196, 768]" = split_35[1];  split_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_35: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_143)
    mul_142: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_143, sigmoid_35);  getitem_143 = sigmoid_35 = None
    mul_143: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_142, mul_142);  getitem_142 = mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_124: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_143);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_143: "f32[1568, 768]" = torch.ops.aten.view.default(clone_124, [1568, 768]);  clone_124 = None
    permute_108: "f32[768, 384]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
    addmm_53: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg217_1, view_143, permute_108);  arg217_1 = view_143 = permute_108 = None
    view_144: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_53, [8, 196, 384]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_125: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_125: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_122, clone_125);  add_122 = clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_126: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_126, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 196, 1]" = var_mean_36[0]
    getitem_145: "f32[8, 196, 1]" = var_mean_36[1];  var_mean_36 = None
    add_126: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_36: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_36: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_126, getitem_145);  clone_126 = getitem_145 = None
    mul_144: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
    mul_145: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_144, arg218_1);  mul_144 = arg218_1 = None
    add_127: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_145, arg219_1);  mul_145 = arg219_1 = None
    permute_109: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_127, [0, 2, 1]);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_110: "f32[196, 384]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
    clone_127: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
    view_145: "f32[3072, 196]" = torch.ops.aten.view.default(clone_127, [3072, 196]);  clone_127 = None
    mm_18: "f32[3072, 384]" = torch.ops.aten.mm.default(view_145, permute_110);  view_145 = permute_110 = None
    view_146: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_18, [8, 384, 384]);  mm_18 = None
    add_128: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_146, arg221_1);  view_146 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_36 = torch.ops.aten.split.Tensor(add_128, 192, -1);  add_128 = None
    getitem_146: "f32[8, 384, 192]" = split_36[0]
    getitem_147: "f32[8, 384, 192]" = split_36[1];  split_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_36: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_147)
    mul_146: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_147, sigmoid_36);  getitem_147 = sigmoid_36 = None
    mul_147: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_146, mul_146);  getitem_146 = mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_128: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_147);  mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_147: "f32[3072, 192]" = torch.ops.aten.view.default(clone_128, [3072, 192]);  clone_128 = None
    permute_111: "f32[192, 196]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
    addmm_54: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg223_1, view_147, permute_111);  arg223_1 = view_147 = permute_111 = None
    view_148: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_54, [8, 384, 196]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_129: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_148);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_112: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_129, [0, 2, 1]);  clone_129 = None
    add_129: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_125, permute_112);  add_125 = permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_130: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_130, [2], correction = 0, keepdim = True)
    getitem_148: "f32[8, 196, 1]" = var_mean_37[0]
    getitem_149: "f32[8, 196, 1]" = var_mean_37[1];  var_mean_37 = None
    add_130: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
    rsqrt_37: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_37: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_130, getitem_149);  clone_130 = getitem_149 = None
    mul_148: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
    mul_149: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_148, arg224_1);  mul_148 = arg224_1 = None
    add_131: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_149, arg225_1);  mul_149 = arg225_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_149: "f32[1568, 384]" = torch.ops.aten.view.default(add_131, [1568, 384]);  add_131 = None
    permute_113: "f32[384, 1536]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
    addmm_55: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg227_1, view_149, permute_113);  arg227_1 = view_149 = permute_113 = None
    view_150: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_55, [8, 196, 1536]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_37 = torch.ops.aten.split.Tensor(view_150, 768, -1);  view_150 = None
    getitem_150: "f32[8, 196, 768]" = split_37[0]
    getitem_151: "f32[8, 196, 768]" = split_37[1];  split_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_37: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_151)
    mul_150: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_151, sigmoid_37);  getitem_151 = sigmoid_37 = None
    mul_151: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_150, mul_150);  getitem_150 = mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_131: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_151);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_151: "f32[1568, 768]" = torch.ops.aten.view.default(clone_131, [1568, 768]);  clone_131 = None
    permute_114: "f32[768, 384]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    addmm_56: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg229_1, view_151, permute_114);  arg229_1 = view_151 = permute_114 = None
    view_152: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_56, [8, 196, 384]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_132: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_152);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_132: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_129, clone_132);  add_129 = clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_133: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_132, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_133, [2], correction = 0, keepdim = True)
    getitem_152: "f32[8, 196, 1]" = var_mean_38[0]
    getitem_153: "f32[8, 196, 1]" = var_mean_38[1];  var_mean_38 = None
    add_133: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-06);  getitem_152 = None
    rsqrt_38: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_38: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_133, getitem_153);  clone_133 = getitem_153 = None
    mul_152: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
    mul_153: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_152, arg230_1);  mul_152 = arg230_1 = None
    add_134: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_153, arg231_1);  mul_153 = arg231_1 = None
    permute_115: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_134, [0, 2, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_116: "f32[196, 384]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
    clone_134: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_153: "f32[3072, 196]" = torch.ops.aten.view.default(clone_134, [3072, 196]);  clone_134 = None
    mm_19: "f32[3072, 384]" = torch.ops.aten.mm.default(view_153, permute_116);  view_153 = permute_116 = None
    view_154: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_19, [8, 384, 384]);  mm_19 = None
    add_135: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_154, arg233_1);  view_154 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_38 = torch.ops.aten.split.Tensor(add_135, 192, -1);  add_135 = None
    getitem_154: "f32[8, 384, 192]" = split_38[0]
    getitem_155: "f32[8, 384, 192]" = split_38[1];  split_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_38: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_155)
    mul_154: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_155, sigmoid_38);  getitem_155 = sigmoid_38 = None
    mul_155: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_154, mul_154);  getitem_154 = mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_135: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_155);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_155: "f32[3072, 192]" = torch.ops.aten.view.default(clone_135, [3072, 192]);  clone_135 = None
    permute_117: "f32[192, 196]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    addmm_57: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg235_1, view_155, permute_117);  arg235_1 = view_155 = permute_117 = None
    view_156: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_57, [8, 384, 196]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_136: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_118: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_136, [0, 2, 1]);  clone_136 = None
    add_136: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_132, permute_118);  add_132 = permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_137: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_136, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_137, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 196, 1]" = var_mean_39[0]
    getitem_157: "f32[8, 196, 1]" = var_mean_39[1];  var_mean_39 = None
    add_137: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
    rsqrt_39: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_39: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_137, getitem_157);  clone_137 = getitem_157 = None
    mul_156: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
    mul_157: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_156, arg236_1);  mul_156 = arg236_1 = None
    add_138: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_157, arg237_1);  mul_157 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_157: "f32[1568, 384]" = torch.ops.aten.view.default(add_138, [1568, 384]);  add_138 = None
    permute_119: "f32[384, 1536]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    addmm_58: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg239_1, view_157, permute_119);  arg239_1 = view_157 = permute_119 = None
    view_158: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_58, [8, 196, 1536]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_39 = torch.ops.aten.split.Tensor(view_158, 768, -1);  view_158 = None
    getitem_158: "f32[8, 196, 768]" = split_39[0]
    getitem_159: "f32[8, 196, 768]" = split_39[1];  split_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_39: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_159)
    mul_158: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_159, sigmoid_39);  getitem_159 = sigmoid_39 = None
    mul_159: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_158, mul_158);  getitem_158 = mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_138: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_159);  mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_159: "f32[1568, 768]" = torch.ops.aten.view.default(clone_138, [1568, 768]);  clone_138 = None
    permute_120: "f32[768, 384]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    addmm_59: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg241_1, view_159, permute_120);  arg241_1 = view_159 = permute_120 = None
    view_160: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_59, [8, 196, 384]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_139: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_160);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_139: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_136, clone_139);  add_136 = clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_140: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_139, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_140, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 196, 1]" = var_mean_40[0]
    getitem_161: "f32[8, 196, 1]" = var_mean_40[1];  var_mean_40 = None
    add_140: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_40: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_40: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_140, getitem_161);  clone_140 = getitem_161 = None
    mul_160: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
    mul_161: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_160, arg242_1);  mul_160 = arg242_1 = None
    add_141: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_161, arg243_1);  mul_161 = arg243_1 = None
    permute_121: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_141, [0, 2, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_122: "f32[196, 384]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    clone_141: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_161: "f32[3072, 196]" = torch.ops.aten.view.default(clone_141, [3072, 196]);  clone_141 = None
    mm_20: "f32[3072, 384]" = torch.ops.aten.mm.default(view_161, permute_122);  view_161 = permute_122 = None
    view_162: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_20, [8, 384, 384]);  mm_20 = None
    add_142: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_162, arg245_1);  view_162 = arg245_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_40 = torch.ops.aten.split.Tensor(add_142, 192, -1);  add_142 = None
    getitem_162: "f32[8, 384, 192]" = split_40[0]
    getitem_163: "f32[8, 384, 192]" = split_40[1];  split_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_40: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_163)
    mul_162: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_163, sigmoid_40);  getitem_163 = sigmoid_40 = None
    mul_163: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_162, mul_162);  getitem_162 = mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_142: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_163);  mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_163: "f32[3072, 192]" = torch.ops.aten.view.default(clone_142, [3072, 192]);  clone_142 = None
    permute_123: "f32[192, 196]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
    addmm_60: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg247_1, view_163, permute_123);  arg247_1 = view_163 = permute_123 = None
    view_164: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_60, [8, 384, 196]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_143: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_164);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_124: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_143, [0, 2, 1]);  clone_143 = None
    add_143: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_139, permute_124);  add_139 = permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_144: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format)
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_144, [2], correction = 0, keepdim = True)
    getitem_164: "f32[8, 196, 1]" = var_mean_41[0]
    getitem_165: "f32[8, 196, 1]" = var_mean_41[1];  var_mean_41 = None
    add_144: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-06);  getitem_164 = None
    rsqrt_41: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_41: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_144, getitem_165);  clone_144 = getitem_165 = None
    mul_164: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
    mul_165: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_164, arg248_1);  mul_164 = arg248_1 = None
    add_145: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_165, arg249_1);  mul_165 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_165: "f32[1568, 384]" = torch.ops.aten.view.default(add_145, [1568, 384]);  add_145 = None
    permute_125: "f32[384, 1536]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    addmm_61: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg251_1, view_165, permute_125);  arg251_1 = view_165 = permute_125 = None
    view_166: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_61, [8, 196, 1536]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_41 = torch.ops.aten.split.Tensor(view_166, 768, -1);  view_166 = None
    getitem_166: "f32[8, 196, 768]" = split_41[0]
    getitem_167: "f32[8, 196, 768]" = split_41[1];  split_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_41: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_167)
    mul_166: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_167, sigmoid_41);  getitem_167 = sigmoid_41 = None
    mul_167: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_166, mul_166);  getitem_166 = mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_145: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_167);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_167: "f32[1568, 768]" = torch.ops.aten.view.default(clone_145, [1568, 768]);  clone_145 = None
    permute_126: "f32[768, 384]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
    addmm_62: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg253_1, view_167, permute_126);  arg253_1 = view_167 = permute_126 = None
    view_168: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_62, [8, 196, 384]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_146: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_168);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_146: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_143, clone_146);  add_143 = clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_147: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_147, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 196, 1]" = var_mean_42[0]
    getitem_169: "f32[8, 196, 1]" = var_mean_42[1];  var_mean_42 = None
    add_147: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_42: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_42: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_147, getitem_169);  clone_147 = getitem_169 = None
    mul_168: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
    mul_169: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_168, arg254_1);  mul_168 = arg254_1 = None
    add_148: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_169, arg255_1);  mul_169 = arg255_1 = None
    permute_127: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_148, [0, 2, 1]);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_128: "f32[196, 384]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
    clone_148: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_169: "f32[3072, 196]" = torch.ops.aten.view.default(clone_148, [3072, 196]);  clone_148 = None
    mm_21: "f32[3072, 384]" = torch.ops.aten.mm.default(view_169, permute_128);  view_169 = permute_128 = None
    view_170: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_21, [8, 384, 384]);  mm_21 = None
    add_149: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_170, arg257_1);  view_170 = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_42 = torch.ops.aten.split.Tensor(add_149, 192, -1);  add_149 = None
    getitem_170: "f32[8, 384, 192]" = split_42[0]
    getitem_171: "f32[8, 384, 192]" = split_42[1];  split_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_42: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_171)
    mul_170: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_171, sigmoid_42);  getitem_171 = sigmoid_42 = None
    mul_171: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_170, mul_170);  getitem_170 = mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_149: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_171);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_171: "f32[3072, 192]" = torch.ops.aten.view.default(clone_149, [3072, 192]);  clone_149 = None
    permute_129: "f32[192, 196]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
    addmm_63: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg259_1, view_171, permute_129);  arg259_1 = view_171 = permute_129 = None
    view_172: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_63, [8, 384, 196]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_150: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_172);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_130: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_150, [0, 2, 1]);  clone_150 = None
    add_150: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_146, permute_130);  add_146 = permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_151: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_151, [2], correction = 0, keepdim = True)
    getitem_172: "f32[8, 196, 1]" = var_mean_43[0]
    getitem_173: "f32[8, 196, 1]" = var_mean_43[1];  var_mean_43 = None
    add_151: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-06);  getitem_172 = None
    rsqrt_43: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_43: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_151, getitem_173);  clone_151 = getitem_173 = None
    mul_172: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
    mul_173: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_172, arg260_1);  mul_172 = arg260_1 = None
    add_152: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_173, arg261_1);  mul_173 = arg261_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_173: "f32[1568, 384]" = torch.ops.aten.view.default(add_152, [1568, 384]);  add_152 = None
    permute_131: "f32[384, 1536]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
    addmm_64: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg263_1, view_173, permute_131);  arg263_1 = view_173 = permute_131 = None
    view_174: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_64, [8, 196, 1536]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_43 = torch.ops.aten.split.Tensor(view_174, 768, -1);  view_174 = None
    getitem_174: "f32[8, 196, 768]" = split_43[0]
    getitem_175: "f32[8, 196, 768]" = split_43[1];  split_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_43: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_175)
    mul_174: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_175, sigmoid_43);  getitem_175 = sigmoid_43 = None
    mul_175: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_174, mul_174);  getitem_174 = mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_152: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_175);  mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_175: "f32[1568, 768]" = torch.ops.aten.view.default(clone_152, [1568, 768]);  clone_152 = None
    permute_132: "f32[768, 384]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
    addmm_65: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg265_1, view_175, permute_132);  arg265_1 = view_175 = permute_132 = None
    view_176: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_65, [8, 196, 384]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_153: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_176);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_153: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_150, clone_153);  add_150 = clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_154: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_154, [2], correction = 0, keepdim = True)
    getitem_176: "f32[8, 196, 1]" = var_mean_44[0]
    getitem_177: "f32[8, 196, 1]" = var_mean_44[1];  var_mean_44 = None
    add_154: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_44: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_44: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_154, getitem_177);  clone_154 = getitem_177 = None
    mul_176: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
    mul_177: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_176, arg266_1);  mul_176 = arg266_1 = None
    add_155: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_177, arg267_1);  mul_177 = arg267_1 = None
    permute_133: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_155, [0, 2, 1]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_134: "f32[196, 384]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
    clone_155: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_177: "f32[3072, 196]" = torch.ops.aten.view.default(clone_155, [3072, 196]);  clone_155 = None
    mm_22: "f32[3072, 384]" = torch.ops.aten.mm.default(view_177, permute_134);  view_177 = permute_134 = None
    view_178: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_22, [8, 384, 384]);  mm_22 = None
    add_156: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_178, arg269_1);  view_178 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_44 = torch.ops.aten.split.Tensor(add_156, 192, -1);  add_156 = None
    getitem_178: "f32[8, 384, 192]" = split_44[0]
    getitem_179: "f32[8, 384, 192]" = split_44[1];  split_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_44: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_179)
    mul_178: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_179, sigmoid_44);  getitem_179 = sigmoid_44 = None
    mul_179: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_178, mul_178);  getitem_178 = mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_156: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_179);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_179: "f32[3072, 192]" = torch.ops.aten.view.default(clone_156, [3072, 192]);  clone_156 = None
    permute_135: "f32[192, 196]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
    addmm_66: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg271_1, view_179, permute_135);  arg271_1 = view_179 = permute_135 = None
    view_180: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_66, [8, 384, 196]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_157: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_136: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_157, [0, 2, 1]);  clone_157 = None
    add_157: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_153, permute_136);  add_153 = permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_158: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format)
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_158, [2], correction = 0, keepdim = True)
    getitem_180: "f32[8, 196, 1]" = var_mean_45[0]
    getitem_181: "f32[8, 196, 1]" = var_mean_45[1];  var_mean_45 = None
    add_158: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
    rsqrt_45: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_45: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_158, getitem_181);  clone_158 = getitem_181 = None
    mul_180: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
    mul_181: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_180, arg272_1);  mul_180 = arg272_1 = None
    add_159: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_181, arg273_1);  mul_181 = arg273_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_181: "f32[1568, 384]" = torch.ops.aten.view.default(add_159, [1568, 384]);  add_159 = None
    permute_137: "f32[384, 1536]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
    addmm_67: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg275_1, view_181, permute_137);  arg275_1 = view_181 = permute_137 = None
    view_182: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_67, [8, 196, 1536]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_45 = torch.ops.aten.split.Tensor(view_182, 768, -1);  view_182 = None
    getitem_182: "f32[8, 196, 768]" = split_45[0]
    getitem_183: "f32[8, 196, 768]" = split_45[1];  split_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_45: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_183)
    mul_182: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_183, sigmoid_45);  getitem_183 = sigmoid_45 = None
    mul_183: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_182, mul_182);  getitem_182 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_159: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_183);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_183: "f32[1568, 768]" = torch.ops.aten.view.default(clone_159, [1568, 768]);  clone_159 = None
    permute_138: "f32[768, 384]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
    addmm_68: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg277_1, view_183, permute_138);  arg277_1 = view_183 = permute_138 = None
    view_184: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_68, [8, 196, 384]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_160: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_184);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_160: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_157, clone_160);  add_157 = clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    clone_161: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_160, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_161, [2], correction = 0, keepdim = True)
    getitem_184: "f32[8, 196, 1]" = var_mean_46[0]
    getitem_185: "f32[8, 196, 1]" = var_mean_46[1];  var_mean_46 = None
    add_161: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-06);  getitem_184 = None
    rsqrt_46: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_46: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_161, getitem_185);  clone_161 = getitem_185 = None
    mul_184: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
    mul_185: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_184, arg278_1);  mul_184 = arg278_1 = None
    add_162: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_185, arg279_1);  mul_185 = arg279_1 = None
    permute_139: "f32[8, 384, 196]" = torch.ops.aten.permute.default(add_162, [0, 2, 1]);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    permute_140: "f32[196, 384]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
    clone_162: "f32[8, 384, 196]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_185: "f32[3072, 196]" = torch.ops.aten.view.default(clone_162, [3072, 196]);  clone_162 = None
    mm_23: "f32[3072, 384]" = torch.ops.aten.mm.default(view_185, permute_140);  view_185 = permute_140 = None
    view_186: "f32[8, 384, 384]" = torch.ops.aten.view.default(mm_23, [8, 384, 384]);  mm_23 = None
    add_163: "f32[8, 384, 384]" = torch.ops.aten.add.Tensor(view_186, arg281_1);  view_186 = arg281_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_46 = torch.ops.aten.split.Tensor(add_163, 192, -1);  add_163 = None
    getitem_186: "f32[8, 384, 192]" = split_46[0]
    getitem_187: "f32[8, 384, 192]" = split_46[1];  split_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_46: "f32[8, 384, 192]" = torch.ops.aten.sigmoid.default(getitem_187)
    mul_186: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_187, sigmoid_46);  getitem_187 = sigmoid_46 = None
    mul_187: "f32[8, 384, 192]" = torch.ops.aten.mul.Tensor(getitem_186, mul_186);  getitem_186 = mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_163: "f32[8, 384, 192]" = torch.ops.aten.clone.default(mul_187);  mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_187: "f32[3072, 192]" = torch.ops.aten.view.default(clone_163, [3072, 192]);  clone_163 = None
    permute_141: "f32[192, 196]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
    addmm_69: "f32[3072, 196]" = torch.ops.aten.addmm.default(arg283_1, view_187, permute_141);  arg283_1 = view_187 = permute_141 = None
    view_188: "f32[8, 384, 196]" = torch.ops.aten.view.default(addmm_69, [8, 384, 196]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_164: "f32[8, 384, 196]" = torch.ops.aten.clone.default(view_188);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_142: "f32[8, 196, 384]" = torch.ops.aten.permute.default(clone_164, [0, 2, 1]);  clone_164 = None
    add_164: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_160, permute_142);  add_160 = permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    clone_165: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_164, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_165, [2], correction = 0, keepdim = True)
    getitem_188: "f32[8, 196, 1]" = var_mean_47[0]
    getitem_189: "f32[8, 196, 1]" = var_mean_47[1];  var_mean_47 = None
    add_165: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-06);  getitem_188 = None
    rsqrt_47: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_47: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_165, getitem_189);  clone_165 = getitem_189 = None
    mul_188: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
    mul_189: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_188, arg284_1);  mul_188 = arg284_1 = None
    add_166: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_189, arg285_1);  mul_189 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:91, code: x = self.fc1(x)
    view_189: "f32[1568, 384]" = torch.ops.aten.view.default(add_166, [1568, 384]);  add_166 = None
    permute_143: "f32[384, 1536]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
    addmm_70: "f32[1568, 1536]" = torch.ops.aten.addmm.default(arg287_1, view_189, permute_143);  arg287_1 = view_189 = permute_143 = None
    view_190: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_70, [8, 196, 1536]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:92, code: x1, x2 = x.chunk(2, dim=self.chunk_dim)
    split_47 = torch.ops.aten.split.Tensor(view_190, 768, -1);  view_190 = None
    getitem_190: "f32[8, 196, 768]" = split_47[0]
    getitem_191: "f32[8, 196, 768]" = split_47[1];  split_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:93, code: x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    sigmoid_47: "f32[8, 196, 768]" = torch.ops.aten.sigmoid.default(getitem_191)
    mul_190: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_191, sigmoid_47);  getitem_191 = sigmoid_47 = None
    mul_191: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(getitem_190, mul_190);  getitem_190 = mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:94, code: x = self.drop1(x)
    clone_166: "f32[8, 196, 768]" = torch.ops.aten.clone.default(mul_191);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:96, code: x = self.fc2(x)
    view_191: "f32[1568, 768]" = torch.ops.aten.view.default(clone_166, [1568, 768]);  clone_166 = None
    permute_144: "f32[768, 384]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
    addmm_71: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg289_1, view_191, permute_144);  arg289_1 = view_191 = permute_144 = None
    view_192: "f32[8, 196, 384]" = torch.ops.aten.view.default(addmm_71, [8, 196, 384]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:97, code: x = self.drop2(x)
    clone_167: "f32[8, 196, 384]" = torch.ops.aten.clone.default(view_192);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_167: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(add_164, clone_167);  add_164 = clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    clone_168: "f32[8, 196, 384]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format);  add_167 = None
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_168, [2], correction = 0, keepdim = True)
    getitem_192: "f32[8, 196, 1]" = var_mean_48[0]
    getitem_193: "f32[8, 196, 1]" = var_mean_48[1];  var_mean_48 = None
    add_168: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
    rsqrt_48: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_48: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(clone_168, getitem_193);  clone_168 = getitem_193 = None
    mul_192: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
    mul_193: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_192, arg290_1);  mul_192 = arg290_1 = None
    add_169: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_193, arg291_1);  mul_193 = arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    mean: "f32[8, 384]" = torch.ops.aten.mean.dim(add_169, [1]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:272, code: x = self.head_drop(x)
    clone_169: "f32[8, 384]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    permute_145: "f32[384, 1000]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
    addmm_72: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg293_1, clone_169, permute_145);  arg293_1 = clone_169 = permute_145 = None
    return (addmm_72,)
    