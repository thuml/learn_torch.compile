from __future__ import annotations



def forward(self, arg0_1: "f32[1, 14, 14, 384]", arg1_1: "f32[1, 1, 384]", arg2_1: "f32[64, 3, 7, 7]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64, 64, 3, 3]", arg6_1: "f32[64]", arg7_1: "f32[64]", arg8_1: "f32[64, 64, 3, 3]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[192, 64, 4, 4]", arg12_1: "f32[192]", arg13_1: "f32[192]", arg14_1: "f32[192]", arg15_1: "f32[192, 192]", arg16_1: "f32[486, 192]", arg17_1: "f32[486]", arg18_1: "f32[192, 192]", arg19_1: "f32[192]", arg20_1: "f32[192]", arg21_1: "f32[192]", arg22_1: "f32[576, 192]", arg23_1: "f32[576]", arg24_1: "f32[192, 576]", arg25_1: "f32[192]", arg26_1: "f32[192]", arg27_1: "f32[192]", arg28_1: "f32[192, 192]", arg29_1: "f32[486, 192]", arg30_1: "f32[486]", arg31_1: "f32[192, 192]", arg32_1: "f32[192]", arg33_1: "f32[192]", arg34_1: "f32[192]", arg35_1: "f32[576, 192]", arg36_1: "f32[576]", arg37_1: "f32[192, 576]", arg38_1: "f32[192]", arg39_1: "f32[192]", arg40_1: "f32[192]", arg41_1: "f32[192, 192]", arg42_1: "f32[486, 192]", arg43_1: "f32[486]", arg44_1: "f32[192, 192]", arg45_1: "f32[192]", arg46_1: "f32[192]", arg47_1: "f32[192]", arg48_1: "f32[576, 192]", arg49_1: "f32[576]", arg50_1: "f32[192, 576]", arg51_1: "f32[192]", arg52_1: "f32[192]", arg53_1: "f32[192]", arg54_1: "f32[192, 192]", arg55_1: "f32[486, 192]", arg56_1: "f32[486]", arg57_1: "f32[192, 192]", arg58_1: "f32[192]", arg59_1: "f32[192]", arg60_1: "f32[192]", arg61_1: "f32[576, 192]", arg62_1: "f32[576]", arg63_1: "f32[192, 576]", arg64_1: "f32[192]", arg65_1: "f32[384, 192, 2, 2]", arg66_1: "f32[384]", arg67_1: "f32[384]", arg68_1: "f32[384]", arg69_1: "f32[1152, 384]", arg70_1: "f32[384, 384]", arg71_1: "f32[384]", arg72_1: "f32[384]", arg73_1: "f32[384]", arg74_1: "f32[1152, 384]", arg75_1: "f32[1152]", arg76_1: "f32[384, 1152]", arg77_1: "f32[384]", arg78_1: "f32[384]", arg79_1: "f32[384]", arg80_1: "f32[1152, 384]", arg81_1: "f32[384, 384]", arg82_1: "f32[384]", arg83_1: "f32[384]", arg84_1: "f32[384]", arg85_1: "f32[1152, 384]", arg86_1: "f32[1152]", arg87_1: "f32[384, 1152]", arg88_1: "f32[384]", arg89_1: "f32[384]", arg90_1: "f32[384]", arg91_1: "f32[1152, 384]", arg92_1: "f32[384, 384]", arg93_1: "f32[384]", arg94_1: "f32[384]", arg95_1: "f32[384]", arg96_1: "f32[1152, 384]", arg97_1: "f32[1152]", arg98_1: "f32[384, 1152]", arg99_1: "f32[384]", arg100_1: "f32[384]", arg101_1: "f32[384]", arg102_1: "f32[1152, 384]", arg103_1: "f32[384, 384]", arg104_1: "f32[384]", arg105_1: "f32[384]", arg106_1: "f32[384]", arg107_1: "f32[1152, 384]", arg108_1: "f32[1152]", arg109_1: "f32[384, 1152]", arg110_1: "f32[384]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[1152, 384]", arg114_1: "f32[384, 384]", arg115_1: "f32[384]", arg116_1: "f32[384]", arg117_1: "f32[384]", arg118_1: "f32[1152, 384]", arg119_1: "f32[1152]", arg120_1: "f32[384, 1152]", arg121_1: "f32[384]", arg122_1: "f32[384]", arg123_1: "f32[384]", arg124_1: "f32[1152, 384]", arg125_1: "f32[384, 384]", arg126_1: "f32[384]", arg127_1: "f32[384]", arg128_1: "f32[384]", arg129_1: "f32[1152, 384]", arg130_1: "f32[1152]", arg131_1: "f32[384, 1152]", arg132_1: "f32[384]", arg133_1: "f32[384]", arg134_1: "f32[384]", arg135_1: "f32[1152, 384]", arg136_1: "f32[384, 384]", arg137_1: "f32[384]", arg138_1: "f32[384]", arg139_1: "f32[384]", arg140_1: "f32[1152, 384]", arg141_1: "f32[1152]", arg142_1: "f32[384, 1152]", arg143_1: "f32[384]", arg144_1: "f32[384]", arg145_1: "f32[384]", arg146_1: "f32[1152, 384]", arg147_1: "f32[384, 384]", arg148_1: "f32[384]", arg149_1: "f32[384]", arg150_1: "f32[384]", arg151_1: "f32[1152, 384]", arg152_1: "f32[1152]", arg153_1: "f32[384, 1152]", arg154_1: "f32[384]", arg155_1: "f32[384]", arg156_1: "f32[384]", arg157_1: "f32[1152, 384]", arg158_1: "f32[384, 384]", arg159_1: "f32[384]", arg160_1: "f32[384]", arg161_1: "f32[384]", arg162_1: "f32[1152, 384]", arg163_1: "f32[1152]", arg164_1: "f32[384, 1152]", arg165_1: "f32[384]", arg166_1: "f32[384]", arg167_1: "f32[384]", arg168_1: "f32[1152, 384]", arg169_1: "f32[384, 384]", arg170_1: "f32[384]", arg171_1: "f32[384]", arg172_1: "f32[384]", arg173_1: "f32[1152, 384]", arg174_1: "f32[1152]", arg175_1: "f32[384, 1152]", arg176_1: "f32[384]", arg177_1: "f32[384]", arg178_1: "f32[384]", arg179_1: "f32[1152, 384]", arg180_1: "f32[384, 384]", arg181_1: "f32[384]", arg182_1: "f32[384]", arg183_1: "f32[384]", arg184_1: "f32[1152, 384]", arg185_1: "f32[1152]", arg186_1: "f32[384, 1152]", arg187_1: "f32[384]", arg188_1: "f32[384]", arg189_1: "f32[384]", arg190_1: "f32[1152, 384]", arg191_1: "f32[384, 384]", arg192_1: "f32[384]", arg193_1: "f32[384]", arg194_1: "f32[384]", arg195_1: "f32[1152, 384]", arg196_1: "f32[1152]", arg197_1: "f32[384, 1152]", arg198_1: "f32[384]", arg199_1: "f32[384]", arg200_1: "f32[384]", arg201_1: "f32[1152, 384]", arg202_1: "f32[384, 384]", arg203_1: "f32[384]", arg204_1: "f32[384]", arg205_1: "f32[384]", arg206_1: "f32[1152, 384]", arg207_1: "f32[1152]", arg208_1: "f32[384, 1152]", arg209_1: "f32[384]", arg210_1: "f32[384]", arg211_1: "f32[384]", arg212_1: "f32[1152, 384]", arg213_1: "f32[384, 384]", arg214_1: "f32[384]", arg215_1: "f32[384]", arg216_1: "f32[384]", arg217_1: "f32[1152, 384]", arg218_1: "f32[1152]", arg219_1: "f32[384, 1152]", arg220_1: "f32[384]", arg221_1: "f32[384]", arg222_1: "f32[384]", arg223_1: "f32[768, 384]", arg224_1: "f32[384, 384]", arg225_1: "f32[384, 384]", arg226_1: "f32[384]", arg227_1: "f32[384]", arg228_1: "f32[384]", arg229_1: "f32[1152, 384]", arg230_1: "f32[1152]", arg231_1: "f32[384, 1152]", arg232_1: "f32[384]", arg233_1: "f32[384]", arg234_1: "f32[384]", arg235_1: "f32[768, 384]", arg236_1: "f32[384, 384]", arg237_1: "f32[384, 384]", arg238_1: "f32[384]", arg239_1: "f32[384]", arg240_1: "f32[384]", arg241_1: "f32[1152, 384]", arg242_1: "f32[1152]", arg243_1: "f32[384, 1152]", arg244_1: "f32[384]", arg245_1: "f32[384]", arg246_1: "f32[384]", arg247_1: "f32[1000, 384]", arg248_1: "f32[1000]", arg249_1: "f32[1000, 384]", arg250_1: "f32[1000]", arg251_1: "f32[64]", arg252_1: "f32[64]", arg253_1: "i64[]", arg254_1: "f32[64]", arg255_1: "f32[64]", arg256_1: "i64[]", arg257_1: "f32[64]", arg258_1: "f32[64]", arg259_1: "i64[]", arg260_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:357, code: x = self.conv(x)
    convolution: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(arg260_1, arg2_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg260_1 = arg2_1 = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    sub: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    add: "f32[64]" = torch.ops.aten.add.Tensor(arg252_1, 1e-05);  arg252_1 = None
    sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    mul_1: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    relu: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    convolution_1: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu, arg5_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg5_1 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    sub_1: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    mul_4: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    relu_1: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    convolution_2: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu_1, arg8_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg8_1 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    sub_2: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    add_4: "f32[64]" = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    mul_7: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    relu_2: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:358, code: x = self.proj(x)  # B, C, H, W
    convolution_3: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_2, arg11_1, arg12_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg11_1 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:695, code: x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C
    permute: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone, [3], correction = 0, keepdim = True)
    getitem: "f32[8, 28, 28, 1]" = var_mean[0]
    getitem_1: "f32[8, 28, 28, 1]" = var_mean[1];  var_mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    full_default: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_4: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_30: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
    iota_5: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_31: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
    add_10: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_30, unsqueeze_31);  unsqueeze_30 = unsqueeze_31 = None
    unsqueeze_32: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_10, -1);  add_10 = None
    unsqueeze_33: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    iota_6: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_34: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_6, 0);  iota_6 = None
    iota_7: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_35: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
    add_11: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_34, unsqueeze_35);  unsqueeze_34 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sub_3: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    add_6: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    mul_9: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = rsqrt = None
    mul_10: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_9, arg13_1);  mul_9 = arg13_1 = None
    add_7: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_10, arg14_1);  mul_10 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_5: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_7, [0, 3, 1, 2])
    avg_pool2d: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_5, [2, 2], [2, 2], [0, 0], True);  permute_5 = None
    permute_6: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d, [0, 2, 3, 1]);  avg_pool2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_4: "f32[1568, 192]" = torch.ops.aten.reshape.default(permute_6, [1568, 192]);  permute_6 = None
    permute_7: "f32[192, 486]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    
    # No stacktrace found for following nodes
    mm_default_60: "f32[1568, 486]" = torch.ops.aten.mm.default(view_4, permute_7);  view_4 = permute_7 = None
    add_tensor_60: "f32[1568, 486]" = torch.ops.aten.add.Tensor(mm_default_60, arg17_1);  mm_default_60 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_5: "f32[8, 14, 14, 486]" = torch.ops.aten.reshape.default(add_tensor_60, [8, 14, 14, 486]);  add_tensor_60 = None
    view_6: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.reshape.default(view_5, [8, 196, 6, 9, 9]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_8: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3, 4]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_11: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_8, 0.1767766952966369);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_11, memory_format = torch.contiguous_format);  mul_11 = None
    amax: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_2, [-1], True)
    sub_4: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_2, amax);  clone_2 = amax = None
    exp: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_1: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(div, [8, 6, 196, 9, 9]);  div = None
    view_7: "f32[9408, 9, 9]" = torch.ops.aten.reshape.default(expand, [9408, 9, 9]);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    view: "f32[6272, 192]" = torch.ops.aten.reshape.default(add_7, [6272, 192]);  add_7 = None
    permute_1: "f32[192, 192]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    mm: "f32[6272, 192]" = torch.ops.aten.mm.default(view, permute_1);  view = permute_1 = None
    view_1: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(mm, [8, 28, 28, 192]);  mm = None
    permute_2: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_1, [0, 3, 1, 2]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    constant_pad_nd: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_2, [1, 1, 1, 1], 0.0);  permute_2 = None
    iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_24: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    iota_1: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_25: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    add_8: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_24, unsqueeze_25);  unsqueeze_24 = unsqueeze_25 = None
    unsqueeze_28: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_8, -1);  add_8 = None
    unsqueeze_29: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    iota_2: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_26: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
    iota_3: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_27: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
    add_9: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_26, unsqueeze_27);  unsqueeze_26 = unsqueeze_27 = None
    index: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(constant_pad_nd, [None, None, unsqueeze_29, add_9]);  constant_pad_nd = unsqueeze_29 = add_9 = None
    permute_3: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
    clone_1: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    view_2: "f32[8, 1728, 196]" = torch.ops.aten.reshape.default(clone_1, [8, 1728, 196]);  clone_1 = None
    view_3: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.reshape.default(view_2, [8, 6, 32, 9, 196]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_4: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_3, [0, 1, 4, 3, 2]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_1: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_4, [8, 6, 196, 9, 32]);  permute_4 = None
    clone_4: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_8: "f32[9408, 9, 32]" = torch.ops.aten.reshape.default(clone_4, [9408, 9, 32]);  clone_4 = None
    bmm: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_7, view_8);  view_7 = view_8 = None
    view_9: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.reshape.default(bmm, [8, 6, 196, 9, 32]);  bmm = None
    permute_9: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_9, [0, 1, 4, 3, 2]);  view_9 = None
    clone_5: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    view_10: "f32[8, 1728, 196]" = torch.ops.aten.reshape.default(clone_5, [8, 1728, 196]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_11: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.reshape.default(view_10, [8, 192, 3, 3, 14, 14]);  view_10 = None
    permute_10: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_11, [0, 1, 2, 4, 3, 5]);  view_11 = None
    _unsafe_index_put: "f32[8, 192, 30, 30]" = torch.ops.prims._unsafe_index_put_.default(full_default, [None, None, unsqueeze_33, add_11], permute_10, True);  full_default = unsqueeze_33 = add_11 = permute_10 = None
    constant_pad_nd_1: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put, [-1, -1, -1, -1], 0.0);  _unsafe_index_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_11: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_1, [0, 2, 3, 1]);  constant_pad_nd_1 = None
    clone_6: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    view_12: "f32[6272, 192]" = torch.ops.aten.reshape.default(clone_6, [6272, 192]);  clone_6 = None
    permute_12: "f32[192, 192]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    mm_1: "f32[6272, 192]" = torch.ops.aten.mm.default(view_12, permute_12);  view_12 = permute_12 = None
    view_13: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(mm_1, [8, 28, 28, 192]);  mm_1 = None
    add_12: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_13, arg19_1);  view_13 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_13: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute, add_12);  permute = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_8: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_13, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_8, [3], correction = 0, keepdim = True)
    getitem_2: "f32[8, 28, 28, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 28, 28, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_5: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_8, getitem_3);  clone_8 = getitem_3 = None
    add_14: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    mul_12: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = rsqrt_1 = None
    mul_13: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_12, arg20_1);  mul_12 = arg20_1 = None
    add_15: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_13, arg21_1);  mul_13 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_14: "f32[6272, 192]" = torch.ops.aten.reshape.default(add_15, [6272, 192]);  add_15 = None
    permute_13: "f32[192, 576]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[6272, 576]" = torch.ops.aten.mm.default(view_14, permute_13);  view_14 = permute_13 = None
    add_tensor_59: "f32[6272, 576]" = torch.ops.aten.add.Tensor(mm_default_59, arg23_1);  mm_default_59 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_15: "f32[8, 28, 28, 576]" = torch.ops.aten.reshape.default(add_tensor_59, [8, 28, 28, 576]);  add_tensor_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_14: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    mul_15: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
    erf: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_16: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_16: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_14, add_16);  mul_14 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_16: "f32[6272, 576]" = torch.ops.aten.reshape.default(mul_16, [6272, 576]);  mul_16 = None
    permute_14: "f32[576, 192]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[6272, 192]" = torch.ops.aten.mm.default(view_16, permute_14);  view_16 = permute_14 = None
    add_tensor_58: "f32[6272, 192]" = torch.ops.aten.add.Tensor(mm_default_58, arg25_1);  mm_default_58 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(add_tensor_58, [8, 28, 28, 192]);  add_tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_17: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_13, view_17);  add_13 = view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_11: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_11, [3], correction = 0, keepdim = True)
    getitem_4: "f32[8, 28, 28, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 28, 28, 1]" = var_mean_2[1];  var_mean_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    full_default_1: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_12: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_42: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_12, 0);  iota_12 = None
    iota_13: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_43: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
    add_22: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_42, unsqueeze_43);  unsqueeze_42 = unsqueeze_43 = None
    unsqueeze_44: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_22, -1);  add_22 = None
    unsqueeze_45: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    iota_14: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_46: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_14, 0);  iota_14 = None
    iota_15: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_47: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
    add_23: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_46, unsqueeze_47);  unsqueeze_46 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sub_6: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_11, getitem_5);  clone_11 = getitem_5 = None
    add_18: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_17: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = rsqrt_2 = None
    mul_18: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_17, arg26_1);  mul_17 = arg26_1 = None
    add_19: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_18, arg27_1);  mul_18 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_19: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_19, [0, 3, 1, 2])
    avg_pool2d_1: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_19, [2, 2], [2, 2], [0, 0], True);  permute_19 = None
    permute_20: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_1, [0, 2, 3, 1]);  avg_pool2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_22: "f32[1568, 192]" = torch.ops.aten.reshape.default(permute_20, [1568, 192]);  permute_20 = None
    permute_21: "f32[192, 486]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[1568, 486]" = torch.ops.aten.mm.default(view_22, permute_21);  view_22 = permute_21 = None
    add_tensor_57: "f32[1568, 486]" = torch.ops.aten.add.Tensor(mm_default_57, arg30_1);  mm_default_57 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_23: "f32[8, 14, 14, 486]" = torch.ops.aten.reshape.default(add_tensor_57, [8, 14, 14, 486]);  add_tensor_57 = None
    view_24: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.reshape.default(view_23, [8, 196, 6, 9, 9]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_22: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3, 4]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_19: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_22, 0.1767766952966369);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_13: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_19, memory_format = torch.contiguous_format);  mul_19 = None
    amax_1: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_13, [-1], True)
    sub_7: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_13, amax_1);  clone_13 = amax_1 = None
    exp_1: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_2: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(div_1, [8, 6, 196, 9, 9]);  div_1 = None
    view_25: "f32[9408, 9, 9]" = torch.ops.aten.reshape.default(expand_2, [9408, 9, 9]);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    view_18: "f32[6272, 192]" = torch.ops.aten.reshape.default(add_19, [6272, 192]);  add_19 = None
    permute_15: "f32[192, 192]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
    mm_2: "f32[6272, 192]" = torch.ops.aten.mm.default(view_18, permute_15);  view_18 = permute_15 = None
    view_19: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(mm_2, [8, 28, 28, 192]);  mm_2 = None
    permute_16: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_19, [0, 3, 1, 2]);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    constant_pad_nd_2: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_16, [1, 1, 1, 1], 0.0);  permute_16 = None
    iota_8: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_36: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_8, 0);  iota_8 = None
    iota_9: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_37: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
    add_20: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_36, unsqueeze_37);  unsqueeze_36 = unsqueeze_37 = None
    unsqueeze_40: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_20, -1);  add_20 = None
    unsqueeze_41: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    iota_10: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_38: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_10, 0);  iota_10 = None
    iota_11: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_39: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
    add_21: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_38, unsqueeze_39);  unsqueeze_38 = unsqueeze_39 = None
    index_1: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(constant_pad_nd_2, [None, None, unsqueeze_41, add_21]);  constant_pad_nd_2 = unsqueeze_41 = add_21 = None
    permute_17: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
    clone_12: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_20: "f32[8, 1728, 196]" = torch.ops.aten.reshape.default(clone_12, [8, 1728, 196]);  clone_12 = None
    view_21: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.reshape.default(view_20, [8, 6, 32, 9, 196]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_18: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_21, [0, 1, 4, 3, 2]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_3: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_18, [8, 6, 196, 9, 32]);  permute_18 = None
    clone_15: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_26: "f32[9408, 9, 32]" = torch.ops.aten.reshape.default(clone_15, [9408, 9, 32]);  clone_15 = None
    bmm_1: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_25, view_26);  view_25 = view_26 = None
    view_27: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.reshape.default(bmm_1, [8, 6, 196, 9, 32]);  bmm_1 = None
    permute_23: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_27, [0, 1, 4, 3, 2]);  view_27 = None
    clone_16: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_28: "f32[8, 1728, 196]" = torch.ops.aten.reshape.default(clone_16, [8, 1728, 196]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_29: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.reshape.default(view_28, [8, 192, 3, 3, 14, 14]);  view_28 = None
    permute_24: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_29, [0, 1, 2, 4, 3, 5]);  view_29 = None
    _unsafe_index_put_1: "f32[8, 192, 30, 30]" = torch.ops.prims._unsafe_index_put_.default(full_default_1, [None, None, unsqueeze_45, add_23], permute_24, True);  full_default_1 = unsqueeze_45 = add_23 = permute_24 = None
    constant_pad_nd_3: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_1, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_25: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_3, [0, 2, 3, 1]);  constant_pad_nd_3 = None
    clone_17: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_30: "f32[6272, 192]" = torch.ops.aten.reshape.default(clone_17, [6272, 192]);  clone_17 = None
    permute_26: "f32[192, 192]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    mm_3: "f32[6272, 192]" = torch.ops.aten.mm.default(view_30, permute_26);  view_30 = permute_26 = None
    view_31: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(mm_3, [8, 28, 28, 192]);  mm_3 = None
    add_24: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_31, arg32_1);  view_31 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_25: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_17, add_24);  add_17 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_19: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_25, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_19, [3], correction = 0, keepdim = True)
    getitem_6: "f32[8, 28, 28, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 28, 28, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_8: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_19, getitem_7);  clone_19 = getitem_7 = None
    add_26: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    mul_20: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = rsqrt_3 = None
    mul_21: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_20, arg33_1);  mul_20 = arg33_1 = None
    add_27: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_21, arg34_1);  mul_21 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_32: "f32[6272, 192]" = torch.ops.aten.reshape.default(add_27, [6272, 192]);  add_27 = None
    permute_27: "f32[192, 576]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[6272, 576]" = torch.ops.aten.mm.default(view_32, permute_27);  view_32 = permute_27 = None
    add_tensor_56: "f32[6272, 576]" = torch.ops.aten.add.Tensor(mm_default_56, arg36_1);  mm_default_56 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_33: "f32[8, 28, 28, 576]" = torch.ops.aten.reshape.default(add_tensor_56, [8, 28, 28, 576]);  add_tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_22: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_23: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476);  view_33 = None
    erf_1: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_28: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_24: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_22, add_28);  mul_22 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_34: "f32[6272, 576]" = torch.ops.aten.reshape.default(mul_24, [6272, 576]);  mul_24 = None
    permute_28: "f32[576, 192]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[6272, 192]" = torch.ops.aten.mm.default(view_34, permute_28);  view_34 = permute_28 = None
    add_tensor_55: "f32[6272, 192]" = torch.ops.aten.add.Tensor(mm_default_55, arg38_1);  mm_default_55 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(add_tensor_55, [8, 28, 28, 192]);  add_tensor_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_29: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_25, view_35);  add_25 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_22: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_29, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_22, [3], correction = 0, keepdim = True)
    getitem_8: "f32[8, 28, 28, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 28, 28, 1]" = var_mean_4[1];  var_mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    full_default_2: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_20: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_54: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_20, 0);  iota_20 = None
    iota_21: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_55: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
    add_34: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_54, unsqueeze_55);  unsqueeze_54 = unsqueeze_55 = None
    unsqueeze_56: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_34, -1);  add_34 = None
    unsqueeze_57: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    iota_22: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_58: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_22, 0);  iota_22 = None
    iota_23: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_59: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
    add_35: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_58, unsqueeze_59);  unsqueeze_58 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sub_9: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_22, getitem_9);  clone_22 = getitem_9 = None
    add_30: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    mul_25: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = rsqrt_4 = None
    mul_26: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_25, arg39_1);  mul_25 = arg39_1 = None
    add_31: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_26, arg40_1);  mul_26 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_33: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_31, [0, 3, 1, 2])
    avg_pool2d_2: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_33, [2, 2], [2, 2], [0, 0], True);  permute_33 = None
    permute_34: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_2, [0, 2, 3, 1]);  avg_pool2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_40: "f32[1568, 192]" = torch.ops.aten.reshape.default(permute_34, [1568, 192]);  permute_34 = None
    permute_35: "f32[192, 486]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[1568, 486]" = torch.ops.aten.mm.default(view_40, permute_35);  view_40 = permute_35 = None
    add_tensor_54: "f32[1568, 486]" = torch.ops.aten.add.Tensor(mm_default_54, arg43_1);  mm_default_54 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_41: "f32[8, 14, 14, 486]" = torch.ops.aten.reshape.default(add_tensor_54, [8, 14, 14, 486]);  add_tensor_54 = None
    view_42: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.reshape.default(view_41, [8, 196, 6, 9, 9]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_36: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3, 4]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_27: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_36, 0.1767766952966369);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_24: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_27, memory_format = torch.contiguous_format);  mul_27 = None
    amax_2: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_24, [-1], True)
    sub_10: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_24, amax_2);  clone_24 = amax_2 = None
    exp_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_3: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_4: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(div_2, [8, 6, 196, 9, 9]);  div_2 = None
    view_43: "f32[9408, 9, 9]" = torch.ops.aten.reshape.default(expand_4, [9408, 9, 9]);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    view_36: "f32[6272, 192]" = torch.ops.aten.reshape.default(add_31, [6272, 192]);  add_31 = None
    permute_29: "f32[192, 192]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    mm_4: "f32[6272, 192]" = torch.ops.aten.mm.default(view_36, permute_29);  view_36 = permute_29 = None
    view_37: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(mm_4, [8, 28, 28, 192]);  mm_4 = None
    permute_30: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_37, [0, 3, 1, 2]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    constant_pad_nd_4: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_30, [1, 1, 1, 1], 0.0);  permute_30 = None
    iota_16: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_48: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_16, 0);  iota_16 = None
    iota_17: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_49: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
    add_32: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_48, unsqueeze_49);  unsqueeze_48 = unsqueeze_49 = None
    unsqueeze_52: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_32, -1);  add_32 = None
    unsqueeze_53: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    iota_18: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_50: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_18, 0);  iota_18 = None
    iota_19: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_51: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
    add_33: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_50, unsqueeze_51);  unsqueeze_50 = unsqueeze_51 = None
    index_2: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(constant_pad_nd_4, [None, None, unsqueeze_53, add_33]);  constant_pad_nd_4 = unsqueeze_53 = add_33 = None
    permute_31: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_2, [0, 1, 2, 4, 3, 5]);  index_2 = None
    clone_23: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_38: "f32[8, 1728, 196]" = torch.ops.aten.reshape.default(clone_23, [8, 1728, 196]);  clone_23 = None
    view_39: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.reshape.default(view_38, [8, 6, 32, 9, 196]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_32: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_39, [0, 1, 4, 3, 2]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_5: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_32, [8, 6, 196, 9, 32]);  permute_32 = None
    clone_26: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_44: "f32[9408, 9, 32]" = torch.ops.aten.reshape.default(clone_26, [9408, 9, 32]);  clone_26 = None
    bmm_2: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
    view_45: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.reshape.default(bmm_2, [8, 6, 196, 9, 32]);  bmm_2 = None
    permute_37: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_45, [0, 1, 4, 3, 2]);  view_45 = None
    clone_27: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_46: "f32[8, 1728, 196]" = torch.ops.aten.reshape.default(clone_27, [8, 1728, 196]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_47: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.reshape.default(view_46, [8, 192, 3, 3, 14, 14]);  view_46 = None
    permute_38: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_47, [0, 1, 2, 4, 3, 5]);  view_47 = None
    _unsafe_index_put_2: "f32[8, 192, 30, 30]" = torch.ops.prims._unsafe_index_put_.default(full_default_2, [None, None, unsqueeze_57, add_35], permute_38, True);  full_default_2 = unsqueeze_57 = add_35 = permute_38 = None
    constant_pad_nd_5: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_2, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_39: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_5, [0, 2, 3, 1]);  constant_pad_nd_5 = None
    clone_28: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    view_48: "f32[6272, 192]" = torch.ops.aten.reshape.default(clone_28, [6272, 192]);  clone_28 = None
    permute_40: "f32[192, 192]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    mm_5: "f32[6272, 192]" = torch.ops.aten.mm.default(view_48, permute_40);  view_48 = permute_40 = None
    view_49: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(mm_5, [8, 28, 28, 192]);  mm_5 = None
    add_36: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_49, arg45_1);  view_49 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_37: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_29, add_36);  add_29 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_30: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_37, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_30, [3], correction = 0, keepdim = True)
    getitem_10: "f32[8, 28, 28, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 28, 28, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_11: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_30, getitem_11);  clone_30 = getitem_11 = None
    add_38: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    mul_28: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = rsqrt_5 = None
    mul_29: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_28, arg46_1);  mul_28 = arg46_1 = None
    add_39: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_29, arg47_1);  mul_29 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[6272, 192]" = torch.ops.aten.reshape.default(add_39, [6272, 192]);  add_39 = None
    permute_41: "f32[192, 576]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[6272, 576]" = torch.ops.aten.mm.default(view_50, permute_41);  view_50 = permute_41 = None
    add_tensor_53: "f32[6272, 576]" = torch.ops.aten.add.Tensor(mm_default_53, arg49_1);  mm_default_53 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_51: "f32[8, 28, 28, 576]" = torch.ops.aten.reshape.default(add_tensor_53, [8, 28, 28, 576]);  add_tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_30: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_31: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476);  view_51 = None
    erf_2: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_40: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_32: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_30, add_40);  mul_30 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[6272, 576]" = torch.ops.aten.reshape.default(mul_32, [6272, 576]);  mul_32 = None
    permute_42: "f32[576, 192]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[6272, 192]" = torch.ops.aten.mm.default(view_52, permute_42);  view_52 = permute_42 = None
    add_tensor_52: "f32[6272, 192]" = torch.ops.aten.add.Tensor(mm_default_52, arg51_1);  mm_default_52 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(add_tensor_52, [8, 28, 28, 192]);  add_tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_41: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_37, view_53);  add_37 = view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_33: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_33, [3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 28, 28, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 28, 28, 1]" = var_mean_6[1];  var_mean_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    full_default_3: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_28: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_66: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_28, 0);  iota_28 = None
    iota_29: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_67: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
    add_46: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_66, unsqueeze_67);  unsqueeze_66 = unsqueeze_67 = None
    unsqueeze_68: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_46, -1);  add_46 = None
    unsqueeze_69: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    iota_30: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_70: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_30, 0);  iota_30 = None
    iota_31: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_71: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
    add_47: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_70, unsqueeze_71);  unsqueeze_70 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sub_12: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_33, getitem_13);  clone_33 = getitem_13 = None
    add_42: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    mul_33: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = rsqrt_6 = None
    mul_34: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None
    add_43: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_34, arg53_1);  mul_34 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_47: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_43, [0, 3, 1, 2])
    avg_pool2d_3: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_47, [2, 2], [2, 2], [0, 0], True);  permute_47 = None
    permute_48: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_3, [0, 2, 3, 1]);  avg_pool2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_58: "f32[1568, 192]" = torch.ops.aten.reshape.default(permute_48, [1568, 192]);  permute_48 = None
    permute_49: "f32[192, 486]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[1568, 486]" = torch.ops.aten.mm.default(view_58, permute_49);  view_58 = permute_49 = None
    add_tensor_51: "f32[1568, 486]" = torch.ops.aten.add.Tensor(mm_default_51, arg56_1);  mm_default_51 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_59: "f32[8, 14, 14, 486]" = torch.ops.aten.reshape.default(add_tensor_51, [8, 14, 14, 486]);  add_tensor_51 = None
    view_60: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.reshape.default(view_59, [8, 196, 6, 9, 9]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_50: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3, 4]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_35: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_50, 0.1767766952966369);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_35: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_35, memory_format = torch.contiguous_format);  mul_35 = None
    amax_3: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_35, [-1], True)
    sub_13: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_35, amax_3);  clone_35 = amax_3 = None
    exp_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_4: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_6: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(div_3, [8, 6, 196, 9, 9]);  div_3 = None
    view_61: "f32[9408, 9, 9]" = torch.ops.aten.reshape.default(expand_6, [9408, 9, 9]);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    view_54: "f32[6272, 192]" = torch.ops.aten.reshape.default(add_43, [6272, 192]);  add_43 = None
    permute_43: "f32[192, 192]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    mm_6: "f32[6272, 192]" = torch.ops.aten.mm.default(view_54, permute_43);  view_54 = permute_43 = None
    view_55: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(mm_6, [8, 28, 28, 192]);  mm_6 = None
    permute_44: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_55, [0, 3, 1, 2]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    constant_pad_nd_6: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_44, [1, 1, 1, 1], 0.0);  permute_44 = None
    iota_24: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_60: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_24, 0);  iota_24 = None
    iota_25: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_61: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
    add_44: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_60, unsqueeze_61);  unsqueeze_60 = unsqueeze_61 = None
    unsqueeze_64: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_44, -1);  add_44 = None
    unsqueeze_65: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    iota_26: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_62: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_26, 0);  iota_26 = None
    iota_27: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_63: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
    add_45: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_62, unsqueeze_63);  unsqueeze_62 = unsqueeze_63 = None
    index_3: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(constant_pad_nd_6, [None, None, unsqueeze_65, add_45]);  constant_pad_nd_6 = unsqueeze_65 = add_45 = None
    permute_45: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_3, [0, 1, 2, 4, 3, 5]);  index_3 = None
    clone_34: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_56: "f32[8, 1728, 196]" = torch.ops.aten.reshape.default(clone_34, [8, 1728, 196]);  clone_34 = None
    view_57: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.reshape.default(view_56, [8, 6, 32, 9, 196]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_46: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_57, [0, 1, 4, 3, 2]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_7: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_46, [8, 6, 196, 9, 32]);  permute_46 = None
    clone_37: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_62: "f32[9408, 9, 32]" = torch.ops.aten.reshape.default(clone_37, [9408, 9, 32]);  clone_37 = None
    bmm_3: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_61, view_62);  view_61 = view_62 = None
    view_63: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.reshape.default(bmm_3, [8, 6, 196, 9, 32]);  bmm_3 = None
    permute_51: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_63, [0, 1, 4, 3, 2]);  view_63 = None
    clone_38: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_64: "f32[8, 1728, 196]" = torch.ops.aten.reshape.default(clone_38, [8, 1728, 196]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_65: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.reshape.default(view_64, [8, 192, 3, 3, 14, 14]);  view_64 = None
    permute_52: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_65, [0, 1, 2, 4, 3, 5]);  view_65 = None
    _unsafe_index_put_3: "f32[8, 192, 30, 30]" = torch.ops.prims._unsafe_index_put_.default(full_default_3, [None, None, unsqueeze_69, add_47], permute_52, True);  full_default_3 = unsqueeze_69 = add_47 = permute_52 = None
    constant_pad_nd_7: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_3, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_53: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_7, [0, 2, 3, 1]);  constant_pad_nd_7 = None
    clone_39: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    view_66: "f32[6272, 192]" = torch.ops.aten.reshape.default(clone_39, [6272, 192]);  clone_39 = None
    permute_54: "f32[192, 192]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    mm_7: "f32[6272, 192]" = torch.ops.aten.mm.default(view_66, permute_54);  view_66 = permute_54 = None
    view_67: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(mm_7, [8, 28, 28, 192]);  mm_7 = None
    add_48: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_67, arg58_1);  view_67 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_49: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_41, add_48);  add_41 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_41: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_49, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_41, [3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 28, 28, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 28, 28, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_14: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_41, getitem_15);  clone_41 = getitem_15 = None
    add_50: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    mul_36: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = rsqrt_7 = None
    mul_37: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_36, arg59_1);  mul_36 = arg59_1 = None
    add_51: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_37, arg60_1);  mul_37 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_68: "f32[6272, 192]" = torch.ops.aten.reshape.default(add_51, [6272, 192]);  add_51 = None
    permute_55: "f32[192, 576]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[6272, 576]" = torch.ops.aten.mm.default(view_68, permute_55);  view_68 = permute_55 = None
    add_tensor_50: "f32[6272, 576]" = torch.ops.aten.add.Tensor(mm_default_50, arg62_1);  mm_default_50 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_69: "f32[8, 28, 28, 576]" = torch.ops.aten.reshape.default(add_tensor_50, [8, 28, 28, 576]);  add_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_38: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
    mul_39: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476);  view_69 = None
    erf_3: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_52: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_40: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_38, add_52);  mul_38 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_70: "f32[6272, 576]" = torch.ops.aten.reshape.default(mul_40, [6272, 576]);  mul_40 = None
    permute_56: "f32[576, 192]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[6272, 192]" = torch.ops.aten.mm.default(view_70, permute_56);  view_70 = permute_56 = None
    add_tensor_49: "f32[6272, 192]" = torch.ops.aten.add.Tensor(mm_default_49, arg64_1);  mm_default_49 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[8, 28, 28, 192]" = torch.ops.aten.reshape.default(add_tensor_49, [8, 28, 28, 192]);  add_tensor_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_53: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_49, view_71);  add_49 = view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:371, code: x = x.permute(0, 3, 1, 2)
    permute_57: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_53, [0, 3, 1, 2]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:372, code: x = self.proj(x)  # B, C, H, W
    convolution_4: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(permute_57, arg65_1, arg66_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_57 = arg65_1 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:373, code: x = x.permute(0, 2, 3, 1)
    permute_58: "f32[8, 14, 14, 384]" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:620, code: x = x + self.pos_embed
    add_54: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(permute_58, arg0_1);  permute_58 = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_45: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_54, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_45, [3], correction = 0, keepdim = True)
    getitem_16: "f32[8, 14, 14, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 14, 14, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_15: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_45, getitem_17);  clone_45 = getitem_17 = None
    add_55: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    mul_41: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_8);  sub_15 = rsqrt_8 = None
    mul_42: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_41, arg67_1);  mul_41 = arg67_1 = None
    add_56: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_42, arg68_1);  mul_42 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_72: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_56, [1568, 384]);  add_56 = None
    permute_59: "f32[384, 1152]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    mm_8: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_72, permute_59);  view_72 = permute_59 = None
    view_73: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_8, [8, 14, 14, 1152]);  mm_8 = None
    view_74: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_73, [8, 196, 3, 12, 32]);  view_73 = None
    permute_60: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_74, [2, 0, 3, 1, 4]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_60);  permute_60 = None
    getitem_18: "f32[8, 12, 196, 32]" = unbind[0]
    getitem_19: "f32[8, 12, 196, 32]" = unbind[1]
    getitem_20: "f32[8, 12, 196, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_8: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_18, [8, 12, 196, 32]);  getitem_18 = None
    clone_46: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_75: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_46, [96, 196, 32]);  clone_46 = None
    permute_61: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_19, [0, 1, 3, 2]);  getitem_19 = None
    expand_9: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_61, [8, 12, 32, 196]);  permute_61 = None
    clone_47: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_76: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_47, [96, 32, 196]);  clone_47 = None
    bmm_4: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
    view_77: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_4, [8, 12, 196, 196]);  bmm_4 = None
    mul_43: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_77, 0.1767766952966369);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_43, [-1], True)
    sub_16: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_43, amax_4);  mul_43 = amax_4 = None
    exp_4: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_5: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_10: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_4, [8, 12, 196, 196]);  div_4 = None
    view_78: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_10, [96, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_20, [8, 12, 196, 32]);  getitem_20 = None
    clone_49: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_79: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_49, [96, 196, 32]);  clone_49 = None
    bmm_5: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
    view_80: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_5, [8, 12, 196, 32]);  bmm_5 = None
    permute_62: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_50: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_81: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_50, [8, 14, 14, 384]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_82: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_81, [1568, 384]);  view_81 = None
    permute_63: "f32[384, 384]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[1568, 384]" = torch.ops.aten.mm.default(view_82, permute_63);  view_82 = permute_63 = None
    add_tensor_48: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_48, arg71_1);  mm_default_48 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_83: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_48, [8, 14, 14, 384]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_57: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_54, view_83);  add_54 = view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_52: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_57, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_52, [3], correction = 0, keepdim = True)
    getitem_21: "f32[8, 14, 14, 1]" = var_mean_9[0]
    getitem_22: "f32[8, 14, 14, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_17: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_52, getitem_22);  clone_52 = getitem_22 = None
    add_58: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_21, 1e-05);  getitem_21 = None
    rsqrt_9: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    mul_44: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_9);  sub_17 = rsqrt_9 = None
    mul_45: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_44, arg72_1);  mul_44 = arg72_1 = None
    add_59: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_45, arg73_1);  mul_45 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_84: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_59, [1568, 384]);  add_59 = None
    permute_64: "f32[384, 1152]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_84, permute_64);  view_84 = permute_64 = None
    add_tensor_47: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_47, arg75_1);  mm_default_47 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_85: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_47, [8, 14, 14, 1152]);  add_tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_47: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_4: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_60: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_48: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_46, add_60);  mul_46 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_86: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_48, [1568, 1152]);  mul_48 = None
    permute_65: "f32[1152, 384]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[1568, 384]" = torch.ops.aten.mm.default(view_86, permute_65);  view_86 = permute_65 = None
    add_tensor_46: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_46, arg77_1);  mm_default_46 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_87: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_46, [8, 14, 14, 384]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_61: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_57, view_87);  add_57 = view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_55: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_61, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_55, [3], correction = 0, keepdim = True)
    getitem_23: "f32[8, 14, 14, 1]" = var_mean_10[0]
    getitem_24: "f32[8, 14, 14, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_18: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_55, getitem_24);  clone_55 = getitem_24 = None
    add_62: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_23, 1e-05);  getitem_23 = None
    rsqrt_10: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    mul_49: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_10);  sub_18 = rsqrt_10 = None
    mul_50: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_49, arg78_1);  mul_49 = arg78_1 = None
    add_63: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_50, arg79_1);  mul_50 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_88: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_63, [1568, 384]);  add_63 = None
    permute_66: "f32[384, 1152]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    mm_9: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_88, permute_66);  view_88 = permute_66 = None
    view_89: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_9, [8, 14, 14, 1152]);  mm_9 = None
    view_90: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_89, [8, 196, 3, 12, 32]);  view_89 = None
    permute_67: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_90, [2, 0, 3, 1, 4]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
    getitem_25: "f32[8, 12, 196, 32]" = unbind_1[0]
    getitem_26: "f32[8, 12, 196, 32]" = unbind_1[1]
    getitem_27: "f32[8, 12, 196, 32]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_12: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_25, [8, 12, 196, 32]);  getitem_25 = None
    clone_56: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_91: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_56, [96, 196, 32]);  clone_56 = None
    permute_68: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_26, [0, 1, 3, 2]);  getitem_26 = None
    expand_13: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_68, [8, 12, 32, 196]);  permute_68 = None
    clone_57: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_92: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_57, [96, 32, 196]);  clone_57 = None
    bmm_6: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_91, view_92);  view_91 = view_92 = None
    view_93: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_6, [8, 12, 196, 196]);  bmm_6 = None
    mul_51: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_93, 0.1767766952966369);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_51, [-1], True)
    sub_19: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_51, amax_5);  mul_51 = amax_5 = None
    exp_5: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_6: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_14: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_5, [8, 12, 196, 196]);  div_5 = None
    view_94: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_14, [96, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_27, [8, 12, 196, 32]);  getitem_27 = None
    clone_59: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_95: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_59, [96, 196, 32]);  clone_59 = None
    bmm_7: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_94, view_95);  view_94 = view_95 = None
    view_96: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_7, [8, 12, 196, 32]);  bmm_7 = None
    permute_69: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    clone_60: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_97: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_60, [8, 14, 14, 384]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_98: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_97, [1568, 384]);  view_97 = None
    permute_70: "f32[384, 384]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[1568, 384]" = torch.ops.aten.mm.default(view_98, permute_70);  view_98 = permute_70 = None
    add_tensor_45: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_45, arg82_1);  mm_default_45 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_99: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_45, [8, 14, 14, 384]);  add_tensor_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_64: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_61, view_99);  add_61 = view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_62: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_64, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_62, [3], correction = 0, keepdim = True)
    getitem_28: "f32[8, 14, 14, 1]" = var_mean_11[0]
    getitem_29: "f32[8, 14, 14, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_20: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_62, getitem_29);  clone_62 = getitem_29 = None
    add_65: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_11: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    mul_52: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_11);  sub_20 = rsqrt_11 = None
    mul_53: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_52, arg83_1);  mul_52 = arg83_1 = None
    add_66: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_53, arg84_1);  mul_53 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_100: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_66, [1568, 384]);  add_66 = None
    permute_71: "f32[384, 1152]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_100, permute_71);  view_100 = permute_71 = None
    add_tensor_44: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_44, arg86_1);  mm_default_44 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_44, [8, 14, 14, 1152]);  add_tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_54: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, 0.5)
    mul_55: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476);  view_101 = None
    erf_5: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_67: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_56: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_54, add_67);  mul_54 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_102: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_56, [1568, 1152]);  mul_56 = None
    permute_72: "f32[1152, 384]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[1568, 384]" = torch.ops.aten.mm.default(view_102, permute_72);  view_102 = permute_72 = None
    add_tensor_43: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_43, arg88_1);  mm_default_43 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_43, [8, 14, 14, 384]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_68: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_64, view_103);  add_64 = view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_65: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_65, [3], correction = 0, keepdim = True)
    getitem_30: "f32[8, 14, 14, 1]" = var_mean_12[0]
    getitem_31: "f32[8, 14, 14, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_21: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_65, getitem_31);  clone_65 = getitem_31 = None
    add_69: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_12: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    mul_57: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_12);  sub_21 = rsqrt_12 = None
    mul_58: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_57, arg89_1);  mul_57 = arg89_1 = None
    add_70: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_58, arg90_1);  mul_58 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_104: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_70, [1568, 384]);  add_70 = None
    permute_73: "f32[384, 1152]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    mm_10: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_104, permute_73);  view_104 = permute_73 = None
    view_105: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_10, [8, 14, 14, 1152]);  mm_10 = None
    view_106: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_105, [8, 196, 3, 12, 32]);  view_105 = None
    permute_74: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_106, [2, 0, 3, 1, 4]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_32: "f32[8, 12, 196, 32]" = unbind_2[0]
    getitem_33: "f32[8, 12, 196, 32]" = unbind_2[1]
    getitem_34: "f32[8, 12, 196, 32]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_16: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_32, [8, 12, 196, 32]);  getitem_32 = None
    clone_66: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_107: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_66, [96, 196, 32]);  clone_66 = None
    permute_75: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_33, [0, 1, 3, 2]);  getitem_33 = None
    expand_17: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_75, [8, 12, 32, 196]);  permute_75 = None
    clone_67: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_108: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_67, [96, 32, 196]);  clone_67 = None
    bmm_8: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_107, view_108);  view_107 = view_108 = None
    view_109: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_8, [8, 12, 196, 196]);  bmm_8 = None
    mul_59: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_109, 0.1767766952966369);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_59, [-1], True)
    sub_22: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_59, amax_6);  mul_59 = amax_6 = None
    exp_6: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_7: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_18: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_6, [8, 12, 196, 196]);  div_6 = None
    view_110: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_18, [96, 196, 196]);  expand_18 = None
    expand_19: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_34, [8, 12, 196, 32]);  getitem_34 = None
    clone_69: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_111: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_69, [96, 196, 32]);  clone_69 = None
    bmm_9: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_110, view_111);  view_110 = view_111 = None
    view_112: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_9, [8, 12, 196, 32]);  bmm_9 = None
    permute_76: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    clone_70: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
    view_113: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_70, [8, 14, 14, 384]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_114: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_113, [1568, 384]);  view_113 = None
    permute_77: "f32[384, 384]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[1568, 384]" = torch.ops.aten.mm.default(view_114, permute_77);  view_114 = permute_77 = None
    add_tensor_42: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_42, arg93_1);  mm_default_42 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_115: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_42, [8, 14, 14, 384]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_71: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_68, view_115);  add_68 = view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_72: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_71, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_72, [3], correction = 0, keepdim = True)
    getitem_35: "f32[8, 14, 14, 1]" = var_mean_13[0]
    getitem_36: "f32[8, 14, 14, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_23: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_72, getitem_36);  clone_72 = getitem_36 = None
    add_72: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_35, 1e-05);  getitem_35 = None
    rsqrt_13: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    mul_60: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_13);  sub_23 = rsqrt_13 = None
    mul_61: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_60, arg94_1);  mul_60 = arg94_1 = None
    add_73: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_61, arg95_1);  mul_61 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_116: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_73, [1568, 384]);  add_73 = None
    permute_78: "f32[384, 1152]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_116, permute_78);  view_116 = permute_78 = None
    add_tensor_41: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_41, arg97_1);  mm_default_41 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_41, [8, 14, 14, 1152]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_62: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, 0.5)
    mul_63: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476);  view_117 = None
    erf_6: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_74: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_64: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_62, add_74);  mul_62 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_118: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_64, [1568, 1152]);  mul_64 = None
    permute_79: "f32[1152, 384]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[1568, 384]" = torch.ops.aten.mm.default(view_118, permute_79);  view_118 = permute_79 = None
    add_tensor_40: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_40, arg99_1);  mm_default_40 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 14, 14, 384]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_75: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_71, view_119);  add_71 = view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_75: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_75, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_75, [3], correction = 0, keepdim = True)
    getitem_37: "f32[8, 14, 14, 1]" = var_mean_14[0]
    getitem_38: "f32[8, 14, 14, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_24: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_75, getitem_38);  clone_75 = getitem_38 = None
    add_76: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
    rsqrt_14: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    mul_65: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_14);  sub_24 = rsqrt_14 = None
    mul_66: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_65, arg100_1);  mul_65 = arg100_1 = None
    add_77: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_66, arg101_1);  mul_66 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_120: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_77, [1568, 384]);  add_77 = None
    permute_80: "f32[384, 1152]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    mm_11: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_120, permute_80);  view_120 = permute_80 = None
    view_121: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_11, [8, 14, 14, 1152]);  mm_11 = None
    view_122: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_121, [8, 196, 3, 12, 32]);  view_121 = None
    permute_81: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_122, [2, 0, 3, 1, 4]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_81);  permute_81 = None
    getitem_39: "f32[8, 12, 196, 32]" = unbind_3[0]
    getitem_40: "f32[8, 12, 196, 32]" = unbind_3[1]
    getitem_41: "f32[8, 12, 196, 32]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_20: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_39, [8, 12, 196, 32]);  getitem_39 = None
    clone_76: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_123: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_76, [96, 196, 32]);  clone_76 = None
    permute_82: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_40, [0, 1, 3, 2]);  getitem_40 = None
    expand_21: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_82, [8, 12, 32, 196]);  permute_82 = None
    clone_77: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_124: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_77, [96, 32, 196]);  clone_77 = None
    bmm_10: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_123, view_124);  view_123 = view_124 = None
    view_125: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_10, [8, 12, 196, 196]);  bmm_10 = None
    mul_67: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_125, 0.1767766952966369);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_67, [-1], True)
    sub_25: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_67, amax_7);  mul_67 = amax_7 = None
    exp_7: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_8: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_22: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_7, [8, 12, 196, 196]);  div_7 = None
    view_126: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_22, [96, 196, 196]);  expand_22 = None
    expand_23: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_41, [8, 12, 196, 32]);  getitem_41 = None
    clone_79: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_127: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_79, [96, 196, 32]);  clone_79 = None
    bmm_11: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_126, view_127);  view_126 = view_127 = None
    view_128: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_11, [8, 12, 196, 32]);  bmm_11 = None
    permute_83: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    clone_80: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    view_129: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_80, [8, 14, 14, 384]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_130: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_129, [1568, 384]);  view_129 = None
    permute_84: "f32[384, 384]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[1568, 384]" = torch.ops.aten.mm.default(view_130, permute_84);  view_130 = permute_84 = None
    add_tensor_39: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_39, arg104_1);  mm_default_39 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_131: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 14, 14, 384]);  add_tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_78: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_75, view_131);  add_75 = view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_82: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_78, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_82, [3], correction = 0, keepdim = True)
    getitem_42: "f32[8, 14, 14, 1]" = var_mean_15[0]
    getitem_43: "f32[8, 14, 14, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_26: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_82, getitem_43);  clone_82 = getitem_43 = None
    add_79: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_15: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    mul_68: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_15);  sub_26 = rsqrt_15 = None
    mul_69: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_68, arg105_1);  mul_68 = arg105_1 = None
    add_80: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_69, arg106_1);  mul_69 = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_132: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_80, [1568, 384]);  add_80 = None
    permute_85: "f32[384, 1152]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_132, permute_85);  view_132 = permute_85 = None
    add_tensor_38: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_38, arg108_1);  mm_default_38 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_133: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 14, 14, 1152]);  add_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_70: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, 0.5)
    mul_71: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476);  view_133 = None
    erf_7: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
    add_81: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_72: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_70, add_81);  mul_70 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_134: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_72, [1568, 1152]);  mul_72 = None
    permute_86: "f32[1152, 384]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[1568, 384]" = torch.ops.aten.mm.default(view_134, permute_86);  view_134 = permute_86 = None
    add_tensor_37: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_37, arg110_1);  mm_default_37 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_135: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 14, 14, 384]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_82: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_78, view_135);  add_78 = view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_85: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_82, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_85, [3], correction = 0, keepdim = True)
    getitem_44: "f32[8, 14, 14, 1]" = var_mean_16[0]
    getitem_45: "f32[8, 14, 14, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_27: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_85, getitem_45);  clone_85 = getitem_45 = None
    add_83: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_16: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    mul_73: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_16);  sub_27 = rsqrt_16 = None
    mul_74: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_73, arg111_1);  mul_73 = arg111_1 = None
    add_84: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_74, arg112_1);  mul_74 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_136: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_84, [1568, 384]);  add_84 = None
    permute_87: "f32[384, 1152]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    mm_12: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_136, permute_87);  view_136 = permute_87 = None
    view_137: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_12, [8, 14, 14, 1152]);  mm_12 = None
    view_138: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_137, [8, 196, 3, 12, 32]);  view_137 = None
    permute_88: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_138, [2, 0, 3, 1, 4]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_88);  permute_88 = None
    getitem_46: "f32[8, 12, 196, 32]" = unbind_4[0]
    getitem_47: "f32[8, 12, 196, 32]" = unbind_4[1]
    getitem_48: "f32[8, 12, 196, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_24: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_46, [8, 12, 196, 32]);  getitem_46 = None
    clone_86: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_139: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_86, [96, 196, 32]);  clone_86 = None
    permute_89: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_47, [0, 1, 3, 2]);  getitem_47 = None
    expand_25: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_89, [8, 12, 32, 196]);  permute_89 = None
    clone_87: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_140: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_87, [96, 32, 196]);  clone_87 = None
    bmm_12: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_139, view_140);  view_139 = view_140 = None
    view_141: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_12, [8, 12, 196, 196]);  bmm_12 = None
    mul_75: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_141, 0.1767766952966369);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_75, [-1], True)
    sub_28: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_75, amax_8);  mul_75 = amax_8 = None
    exp_8: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_9: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_26: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_8, [8, 12, 196, 196]);  div_8 = None
    view_142: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_26, [96, 196, 196]);  expand_26 = None
    expand_27: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_48, [8, 12, 196, 32]);  getitem_48 = None
    clone_89: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_143: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_89, [96, 196, 32]);  clone_89 = None
    bmm_13: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_142, view_143);  view_142 = view_143 = None
    view_144: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_13, [8, 12, 196, 32]);  bmm_13 = None
    permute_90: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
    clone_90: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    view_145: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_90, [8, 14, 14, 384]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_146: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_145, [1568, 384]);  view_145 = None
    permute_91: "f32[384, 384]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[1568, 384]" = torch.ops.aten.mm.default(view_146, permute_91);  view_146 = permute_91 = None
    add_tensor_36: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_36, arg115_1);  mm_default_36 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_147: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_36, [8, 14, 14, 384]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_85: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_82, view_147);  add_82 = view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_92: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_85, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_92, [3], correction = 0, keepdim = True)
    getitem_49: "f32[8, 14, 14, 1]" = var_mean_17[0]
    getitem_50: "f32[8, 14, 14, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_29: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_92, getitem_50);  clone_92 = getitem_50 = None
    add_86: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-05);  getitem_49 = None
    rsqrt_17: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    mul_76: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_17);  sub_29 = rsqrt_17 = None
    mul_77: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_76, arg116_1);  mul_76 = arg116_1 = None
    add_87: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_77, arg117_1);  mul_77 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_148: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_87, [1568, 384]);  add_87 = None
    permute_92: "f32[384, 1152]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_148, permute_92);  view_148 = permute_92 = None
    add_tensor_35: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_35, arg119_1);  mm_default_35 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_149: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 14, 14, 1152]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_78: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
    mul_79: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
    erf_8: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_88: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_80: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_78, add_88);  mul_78 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_150: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_80, [1568, 1152]);  mul_80 = None
    permute_93: "f32[1152, 384]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[1568, 384]" = torch.ops.aten.mm.default(view_150, permute_93);  view_150 = permute_93 = None
    add_tensor_34: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_34, arg121_1);  mm_default_34 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_151: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 14, 14, 384]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_89: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_85, view_151);  add_85 = view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_95: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_89, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_95, [3], correction = 0, keepdim = True)
    getitem_51: "f32[8, 14, 14, 1]" = var_mean_18[0]
    getitem_52: "f32[8, 14, 14, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_30: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_95, getitem_52);  clone_95 = getitem_52 = None
    add_90: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_51, 1e-05);  getitem_51 = None
    rsqrt_18: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    mul_81: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_18);  sub_30 = rsqrt_18 = None
    mul_82: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_81, arg122_1);  mul_81 = arg122_1 = None
    add_91: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_82, arg123_1);  mul_82 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_152: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_91, [1568, 384]);  add_91 = None
    permute_94: "f32[384, 1152]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    mm_13: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_152, permute_94);  view_152 = permute_94 = None
    view_153: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_13, [8, 14, 14, 1152]);  mm_13 = None
    view_154: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_153, [8, 196, 3, 12, 32]);  view_153 = None
    permute_95: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_154, [2, 0, 3, 1, 4]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
    getitem_53: "f32[8, 12, 196, 32]" = unbind_5[0]
    getitem_54: "f32[8, 12, 196, 32]" = unbind_5[1]
    getitem_55: "f32[8, 12, 196, 32]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_28: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_53, [8, 12, 196, 32]);  getitem_53 = None
    clone_96: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_155: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_96, [96, 196, 32]);  clone_96 = None
    permute_96: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_54, [0, 1, 3, 2]);  getitem_54 = None
    expand_29: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_96, [8, 12, 32, 196]);  permute_96 = None
    clone_97: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_156: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_97, [96, 32, 196]);  clone_97 = None
    bmm_14: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_155, view_156);  view_155 = view_156 = None
    view_157: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_14, [8, 12, 196, 196]);  bmm_14 = None
    mul_83: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_157, 0.1767766952966369);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_83, [-1], True)
    sub_31: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_83, amax_9);  mul_83 = amax_9 = None
    exp_9: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_10: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_30: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_9, [8, 12, 196, 196]);  div_9 = None
    view_158: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_30, [96, 196, 196]);  expand_30 = None
    expand_31: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_55, [8, 12, 196, 32]);  getitem_55 = None
    clone_99: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_159: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_99, [96, 196, 32]);  clone_99 = None
    bmm_15: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_158, view_159);  view_158 = view_159 = None
    view_160: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_15, [8, 12, 196, 32]);  bmm_15 = None
    permute_97: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    clone_100: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    view_161: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_100, [8, 14, 14, 384]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_162: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_161, [1568, 384]);  view_161 = None
    permute_98: "f32[384, 384]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[1568, 384]" = torch.ops.aten.mm.default(view_162, permute_98);  view_162 = permute_98 = None
    add_tensor_33: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_33, arg126_1);  mm_default_33 = arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_163: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 14, 14, 384]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_92: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_89, view_163);  add_89 = view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_102: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_92, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_102, [3], correction = 0, keepdim = True)
    getitem_56: "f32[8, 14, 14, 1]" = var_mean_19[0]
    getitem_57: "f32[8, 14, 14, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_32: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_102, getitem_57);  clone_102 = getitem_57 = None
    add_93: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_19: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    mul_84: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_19);  sub_32 = rsqrt_19 = None
    mul_85: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_84, arg127_1);  mul_84 = arg127_1 = None
    add_94: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_85, arg128_1);  mul_85 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_164: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_94, [1568, 384]);  add_94 = None
    permute_99: "f32[384, 1152]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_164, permute_99);  view_164 = permute_99 = None
    add_tensor_32: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_32, arg130_1);  mm_default_32 = arg130_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_165: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 14, 14, 1152]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, 0.5)
    mul_87: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, 0.7071067811865476);  view_165 = None
    erf_9: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_95: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_88: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_86, add_95);  mul_86 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_166: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_88, [1568, 1152]);  mul_88 = None
    permute_100: "f32[1152, 384]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[1568, 384]" = torch.ops.aten.mm.default(view_166, permute_100);  view_166 = permute_100 = None
    add_tensor_31: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_31, arg132_1);  mm_default_31 = arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_167: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 14, 14, 384]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_96: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_92, view_167);  add_92 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_105: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_96, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_105, [3], correction = 0, keepdim = True)
    getitem_58: "f32[8, 14, 14, 1]" = var_mean_20[0]
    getitem_59: "f32[8, 14, 14, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_33: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_105, getitem_59);  clone_105 = getitem_59 = None
    add_97: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_20: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    mul_89: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_20);  sub_33 = rsqrt_20 = None
    mul_90: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_89, arg133_1);  mul_89 = arg133_1 = None
    add_98: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_90, arg134_1);  mul_90 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_168: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_98, [1568, 384]);  add_98 = None
    permute_101: "f32[384, 1152]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    mm_14: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_168, permute_101);  view_168 = permute_101 = None
    view_169: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_14, [8, 14, 14, 1152]);  mm_14 = None
    view_170: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_169, [8, 196, 3, 12, 32]);  view_169 = None
    permute_102: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_170, [2, 0, 3, 1, 4]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_102);  permute_102 = None
    getitem_60: "f32[8, 12, 196, 32]" = unbind_6[0]
    getitem_61: "f32[8, 12, 196, 32]" = unbind_6[1]
    getitem_62: "f32[8, 12, 196, 32]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_32: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_60, [8, 12, 196, 32]);  getitem_60 = None
    clone_106: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_171: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_106, [96, 196, 32]);  clone_106 = None
    permute_103: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_61, [0, 1, 3, 2]);  getitem_61 = None
    expand_33: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_103, [8, 12, 32, 196]);  permute_103 = None
    clone_107: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_172: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_107, [96, 32, 196]);  clone_107 = None
    bmm_16: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_171, view_172);  view_171 = view_172 = None
    view_173: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_16, [8, 12, 196, 196]);  bmm_16 = None
    mul_91: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_173, 0.1767766952966369);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_91, [-1], True)
    sub_34: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_91, amax_10);  mul_91 = amax_10 = None
    exp_10: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_11: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_34: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_10, [8, 12, 196, 196]);  div_10 = None
    view_174: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_34, [96, 196, 196]);  expand_34 = None
    expand_35: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_62, [8, 12, 196, 32]);  getitem_62 = None
    clone_109: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_175: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_109, [96, 196, 32]);  clone_109 = None
    bmm_17: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_174, view_175);  view_174 = view_175 = None
    view_176: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_17, [8, 12, 196, 32]);  bmm_17 = None
    permute_104: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    clone_110: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_177: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_110, [8, 14, 14, 384]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_178: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_177, [1568, 384]);  view_177 = None
    permute_105: "f32[384, 384]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[1568, 384]" = torch.ops.aten.mm.default(view_178, permute_105);  view_178 = permute_105 = None
    add_tensor_30: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_30, arg137_1);  mm_default_30 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_179: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 14, 14, 384]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_99: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_96, view_179);  add_96 = view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_112: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_99, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_112, [3], correction = 0, keepdim = True)
    getitem_63: "f32[8, 14, 14, 1]" = var_mean_21[0]
    getitem_64: "f32[8, 14, 14, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_35: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_112, getitem_64);  clone_112 = getitem_64 = None
    add_100: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
    rsqrt_21: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    mul_92: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_21);  sub_35 = rsqrt_21 = None
    mul_93: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_92, arg138_1);  mul_92 = arg138_1 = None
    add_101: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_93, arg139_1);  mul_93 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_180: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_101, [1568, 384]);  add_101 = None
    permute_106: "f32[384, 1152]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_180, permute_106);  view_180 = permute_106 = None
    add_tensor_29: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_29, arg141_1);  mm_default_29 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_181: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 14, 14, 1152]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_94: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
    mul_95: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476);  view_181 = None
    erf_10: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
    add_102: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_96: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_94, add_102);  mul_94 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_96, [1568, 1152]);  mul_96 = None
    permute_107: "f32[1152, 384]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[1568, 384]" = torch.ops.aten.mm.default(view_182, permute_107);  view_182 = permute_107 = None
    add_tensor_28: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_28, arg143_1);  mm_default_28 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_183: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 14, 14, 384]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_103: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_99, view_183);  add_99 = view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_115: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_103, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_115, [3], correction = 0, keepdim = True)
    getitem_65: "f32[8, 14, 14, 1]" = var_mean_22[0]
    getitem_66: "f32[8, 14, 14, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_36: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_115, getitem_66);  clone_115 = getitem_66 = None
    add_104: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
    rsqrt_22: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    mul_97: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_22);  sub_36 = rsqrt_22 = None
    mul_98: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_97, arg144_1);  mul_97 = arg144_1 = None
    add_105: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_98, arg145_1);  mul_98 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_184: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_105, [1568, 384]);  add_105 = None
    permute_108: "f32[384, 1152]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    mm_15: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_184, permute_108);  view_184 = permute_108 = None
    view_185: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_15, [8, 14, 14, 1152]);  mm_15 = None
    view_186: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_185, [8, 196, 3, 12, 32]);  view_185 = None
    permute_109: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_109);  permute_109 = None
    getitem_67: "f32[8, 12, 196, 32]" = unbind_7[0]
    getitem_68: "f32[8, 12, 196, 32]" = unbind_7[1]
    getitem_69: "f32[8, 12, 196, 32]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_36: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_67, [8, 12, 196, 32]);  getitem_67 = None
    clone_116: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_187: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_116, [96, 196, 32]);  clone_116 = None
    permute_110: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_68, [0, 1, 3, 2]);  getitem_68 = None
    expand_37: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_110, [8, 12, 32, 196]);  permute_110 = None
    clone_117: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_188: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_117, [96, 32, 196]);  clone_117 = None
    bmm_18: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
    view_189: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_18, [8, 12, 196, 196]);  bmm_18 = None
    mul_99: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_189, 0.1767766952966369);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_99, [-1], True)
    sub_37: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_99, amax_11);  mul_99 = amax_11 = None
    exp_11: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_12: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_38: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_11, [8, 12, 196, 196]);  div_11 = None
    view_190: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_38, [96, 196, 196]);  expand_38 = None
    expand_39: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_69, [8, 12, 196, 32]);  getitem_69 = None
    clone_119: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_191: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_119, [96, 196, 32]);  clone_119 = None
    bmm_19: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
    view_192: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_19, [8, 12, 196, 32]);  bmm_19 = None
    permute_111: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_120: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_193: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_120, [8, 14, 14, 384]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_194: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_193, [1568, 384]);  view_193 = None
    permute_112: "f32[384, 384]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[1568, 384]" = torch.ops.aten.mm.default(view_194, permute_112);  view_194 = permute_112 = None
    add_tensor_27: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_27, arg148_1);  mm_default_27 = arg148_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_195: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 14, 14, 384]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_106: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_103, view_195);  add_103 = view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_122: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_106, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_122, [3], correction = 0, keepdim = True)
    getitem_70: "f32[8, 14, 14, 1]" = var_mean_23[0]
    getitem_71: "f32[8, 14, 14, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_38: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_122, getitem_71);  clone_122 = getitem_71 = None
    add_107: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_23: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    mul_100: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_23);  sub_38 = rsqrt_23 = None
    mul_101: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_100, arg149_1);  mul_100 = arg149_1 = None
    add_108: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_101, arg150_1);  mul_101 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_196: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_108, [1568, 384]);  add_108 = None
    permute_113: "f32[384, 1152]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_196, permute_113);  view_196 = permute_113 = None
    add_tensor_26: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_26, arg152_1);  mm_default_26 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_197: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 14, 14, 1152]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_102: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_103: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
    erf_11: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_109: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_104: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_102, add_109);  mul_102 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_198: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_104, [1568, 1152]);  mul_104 = None
    permute_114: "f32[1152, 384]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[1568, 384]" = torch.ops.aten.mm.default(view_198, permute_114);  view_198 = permute_114 = None
    add_tensor_25: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_25, arg154_1);  mm_default_25 = arg154_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_199: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 14, 14, 384]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_110: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_106, view_199);  add_106 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_125: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_110, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_125, [3], correction = 0, keepdim = True)
    getitem_72: "f32[8, 14, 14, 1]" = var_mean_24[0]
    getitem_73: "f32[8, 14, 14, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_39: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_125, getitem_73);  clone_125 = getitem_73 = None
    add_111: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_24: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    mul_105: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_24);  sub_39 = rsqrt_24 = None
    mul_106: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_105, arg155_1);  mul_105 = arg155_1 = None
    add_112: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_106, arg156_1);  mul_106 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_200: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_112, [1568, 384]);  add_112 = None
    permute_115: "f32[384, 1152]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    mm_16: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_200, permute_115);  view_200 = permute_115 = None
    view_201: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_16, [8, 14, 14, 1152]);  mm_16 = None
    view_202: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_201, [8, 196, 3, 12, 32]);  view_201 = None
    permute_116: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_202, [2, 0, 3, 1, 4]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_116);  permute_116 = None
    getitem_74: "f32[8, 12, 196, 32]" = unbind_8[0]
    getitem_75: "f32[8, 12, 196, 32]" = unbind_8[1]
    getitem_76: "f32[8, 12, 196, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_40: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_74, [8, 12, 196, 32]);  getitem_74 = None
    clone_126: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_203: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_126, [96, 196, 32]);  clone_126 = None
    permute_117: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_75, [0, 1, 3, 2]);  getitem_75 = None
    expand_41: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_117, [8, 12, 32, 196]);  permute_117 = None
    clone_127: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_204: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_127, [96, 32, 196]);  clone_127 = None
    bmm_20: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_203, view_204);  view_203 = view_204 = None
    view_205: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_20, [8, 12, 196, 196]);  bmm_20 = None
    mul_107: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_205, 0.1767766952966369);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_107, [-1], True)
    sub_40: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_107, amax_12);  mul_107 = amax_12 = None
    exp_12: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_13: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_42: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_12, [8, 12, 196, 196]);  div_12 = None
    view_206: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_42, [96, 196, 196]);  expand_42 = None
    expand_43: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_76, [8, 12, 196, 32]);  getitem_76 = None
    clone_129: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_207: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_129, [96, 196, 32]);  clone_129 = None
    bmm_21: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_206, view_207);  view_206 = view_207 = None
    view_208: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_21, [8, 12, 196, 32]);  bmm_21 = None
    permute_118: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_130: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_209: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_130, [8, 14, 14, 384]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_210: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_209, [1568, 384]);  view_209 = None
    permute_119: "f32[384, 384]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[1568, 384]" = torch.ops.aten.mm.default(view_210, permute_119);  view_210 = permute_119 = None
    add_tensor_24: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_24, arg159_1);  mm_default_24 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_211: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 14, 14, 384]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_113: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_110, view_211);  add_110 = view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_132: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_113, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_132, [3], correction = 0, keepdim = True)
    getitem_77: "f32[8, 14, 14, 1]" = var_mean_25[0]
    getitem_78: "f32[8, 14, 14, 1]" = var_mean_25[1];  var_mean_25 = None
    sub_41: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_132, getitem_78);  clone_132 = getitem_78 = None
    add_114: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
    rsqrt_25: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    mul_108: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_25);  sub_41 = rsqrt_25 = None
    mul_109: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_108, arg160_1);  mul_108 = arg160_1 = None
    add_115: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_109, arg161_1);  mul_109 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_212: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_115, [1568, 384]);  add_115 = None
    permute_120: "f32[384, 1152]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_212, permute_120);  view_212 = permute_120 = None
    add_tensor_23: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_23, arg163_1);  mm_default_23 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_213: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 14, 14, 1152]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_110: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
    mul_111: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
    erf_12: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_116: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_112: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_110, add_116);  mul_110 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_214: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_112, [1568, 1152]);  mul_112 = None
    permute_121: "f32[1152, 384]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[1568, 384]" = torch.ops.aten.mm.default(view_214, permute_121);  view_214 = permute_121 = None
    add_tensor_22: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_22, arg165_1);  mm_default_22 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_215: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 14, 14, 384]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_117: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_113, view_215);  add_113 = view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_135: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_117, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_135, [3], correction = 0, keepdim = True)
    getitem_79: "f32[8, 14, 14, 1]" = var_mean_26[0]
    getitem_80: "f32[8, 14, 14, 1]" = var_mean_26[1];  var_mean_26 = None
    sub_42: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_135, getitem_80);  clone_135 = getitem_80 = None
    add_118: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_79, 1e-05);  getitem_79 = None
    rsqrt_26: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    mul_113: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_26);  sub_42 = rsqrt_26 = None
    mul_114: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_113, arg166_1);  mul_113 = arg166_1 = None
    add_119: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_114, arg167_1);  mul_114 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_216: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_119, [1568, 384]);  add_119 = None
    permute_122: "f32[384, 1152]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    mm_17: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_216, permute_122);  view_216 = permute_122 = None
    view_217: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_17, [8, 14, 14, 1152]);  mm_17 = None
    view_218: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_217, [8, 196, 3, 12, 32]);  view_217 = None
    permute_123: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_218, [2, 0, 3, 1, 4]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_123);  permute_123 = None
    getitem_81: "f32[8, 12, 196, 32]" = unbind_9[0]
    getitem_82: "f32[8, 12, 196, 32]" = unbind_9[1]
    getitem_83: "f32[8, 12, 196, 32]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_44: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_81, [8, 12, 196, 32]);  getitem_81 = None
    clone_136: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_219: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_136, [96, 196, 32]);  clone_136 = None
    permute_124: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_82, [0, 1, 3, 2]);  getitem_82 = None
    expand_45: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_124, [8, 12, 32, 196]);  permute_124 = None
    clone_137: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_220: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_137, [96, 32, 196]);  clone_137 = None
    bmm_22: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None
    view_221: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_22, [8, 12, 196, 196]);  bmm_22 = None
    mul_115: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_221, 0.1767766952966369);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_115, [-1], True)
    sub_43: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_115, amax_13);  mul_115 = amax_13 = None
    exp_13: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_14: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_46: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_13, [8, 12, 196, 196]);  div_13 = None
    view_222: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_46, [96, 196, 196]);  expand_46 = None
    expand_47: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_83, [8, 12, 196, 32]);  getitem_83 = None
    clone_139: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_223: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_139, [96, 196, 32]);  clone_139 = None
    bmm_23: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_222, view_223);  view_222 = view_223 = None
    view_224: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_23, [8, 12, 196, 32]);  bmm_23 = None
    permute_125: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    clone_140: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    view_225: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_140, [8, 14, 14, 384]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_226: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_225, [1568, 384]);  view_225 = None
    permute_126: "f32[384, 384]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[1568, 384]" = torch.ops.aten.mm.default(view_226, permute_126);  view_226 = permute_126 = None
    add_tensor_21: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_21, arg170_1);  mm_default_21 = arg170_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_227: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 14, 14, 384]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_120: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_117, view_227);  add_117 = view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_142: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_120, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_142, [3], correction = 0, keepdim = True)
    getitem_84: "f32[8, 14, 14, 1]" = var_mean_27[0]
    getitem_85: "f32[8, 14, 14, 1]" = var_mean_27[1];  var_mean_27 = None
    sub_44: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_142, getitem_85);  clone_142 = getitem_85 = None
    add_121: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_27: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    mul_116: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_27);  sub_44 = rsqrt_27 = None
    mul_117: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_116, arg171_1);  mul_116 = arg171_1 = None
    add_122: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_117, arg172_1);  mul_117 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_228: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_122, [1568, 384]);  add_122 = None
    permute_127: "f32[384, 1152]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_228, permute_127);  view_228 = permute_127 = None
    add_tensor_20: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_20, arg174_1);  mm_default_20 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_229: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 14, 14, 1152]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_118: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, 0.5)
    mul_119: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, 0.7071067811865476);  view_229 = None
    erf_13: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
    add_123: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_120: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_118, add_123);  mul_118 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_230: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_120, [1568, 1152]);  mul_120 = None
    permute_128: "f32[1152, 384]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[1568, 384]" = torch.ops.aten.mm.default(view_230, permute_128);  view_230 = permute_128 = None
    add_tensor_19: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_19, arg176_1);  mm_default_19 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_231: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 14, 14, 384]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_124: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_120, view_231);  add_120 = view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_145: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_124, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_145, [3], correction = 0, keepdim = True)
    getitem_86: "f32[8, 14, 14, 1]" = var_mean_28[0]
    getitem_87: "f32[8, 14, 14, 1]" = var_mean_28[1];  var_mean_28 = None
    sub_45: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_145, getitem_87);  clone_145 = getitem_87 = None
    add_125: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_28: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    mul_121: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_28);  sub_45 = rsqrt_28 = None
    mul_122: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_121, arg177_1);  mul_121 = arg177_1 = None
    add_126: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_122, arg178_1);  mul_122 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_232: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_126, [1568, 384]);  add_126 = None
    permute_129: "f32[384, 1152]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    mm_18: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_232, permute_129);  view_232 = permute_129 = None
    view_233: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_18, [8, 14, 14, 1152]);  mm_18 = None
    view_234: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_233, [8, 196, 3, 12, 32]);  view_233 = None
    permute_130: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_234, [2, 0, 3, 1, 4]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_130);  permute_130 = None
    getitem_88: "f32[8, 12, 196, 32]" = unbind_10[0]
    getitem_89: "f32[8, 12, 196, 32]" = unbind_10[1]
    getitem_90: "f32[8, 12, 196, 32]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_48: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_88, [8, 12, 196, 32]);  getitem_88 = None
    clone_146: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_235: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_146, [96, 196, 32]);  clone_146 = None
    permute_131: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_89, [0, 1, 3, 2]);  getitem_89 = None
    expand_49: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_131, [8, 12, 32, 196]);  permute_131 = None
    clone_147: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_236: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_147, [96, 32, 196]);  clone_147 = None
    bmm_24: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_235, view_236);  view_235 = view_236 = None
    view_237: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_24, [8, 12, 196, 196]);  bmm_24 = None
    mul_123: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_237, 0.1767766952966369);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_14: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_123, [-1], True)
    sub_46: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_123, amax_14);  mul_123 = amax_14 = None
    exp_14: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_15: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_50: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_14, [8, 12, 196, 196]);  div_14 = None
    view_238: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_50, [96, 196, 196]);  expand_50 = None
    expand_51: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_90, [8, 12, 196, 32]);  getitem_90 = None
    clone_149: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_239: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_149, [96, 196, 32]);  clone_149 = None
    bmm_25: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_238, view_239);  view_238 = view_239 = None
    view_240: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_25, [8, 12, 196, 32]);  bmm_25 = None
    permute_132: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    clone_150: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_241: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_150, [8, 14, 14, 384]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_242: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_241, [1568, 384]);  view_241 = None
    permute_133: "f32[384, 384]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[1568, 384]" = torch.ops.aten.mm.default(view_242, permute_133);  view_242 = permute_133 = None
    add_tensor_18: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_18, arg181_1);  mm_default_18 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_243: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 14, 14, 384]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_127: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_124, view_243);  add_124 = view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_152: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_127, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_152, [3], correction = 0, keepdim = True)
    getitem_91: "f32[8, 14, 14, 1]" = var_mean_29[0]
    getitem_92: "f32[8, 14, 14, 1]" = var_mean_29[1];  var_mean_29 = None
    sub_47: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_152, getitem_92);  clone_152 = getitem_92 = None
    add_128: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
    rsqrt_29: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    mul_124: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_29);  sub_47 = rsqrt_29 = None
    mul_125: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_124, arg182_1);  mul_124 = arg182_1 = None
    add_129: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_125, arg183_1);  mul_125 = arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_244: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_129, [1568, 384]);  add_129 = None
    permute_134: "f32[384, 1152]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_244, permute_134);  view_244 = permute_134 = None
    add_tensor_17: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_17, arg185_1);  mm_default_17 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_245: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 14, 14, 1152]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_126: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, 0.5)
    mul_127: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, 0.7071067811865476);  view_245 = None
    erf_14: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_127);  mul_127 = None
    add_130: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_128: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_126, add_130);  mul_126 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_246: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_128, [1568, 1152]);  mul_128 = None
    permute_135: "f32[1152, 384]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[1568, 384]" = torch.ops.aten.mm.default(view_246, permute_135);  view_246 = permute_135 = None
    add_tensor_16: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_16, arg187_1);  mm_default_16 = arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_247: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 14, 14, 384]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_131: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_127, view_247);  add_127 = view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_155: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_131, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_155, [3], correction = 0, keepdim = True)
    getitem_93: "f32[8, 14, 14, 1]" = var_mean_30[0]
    getitem_94: "f32[8, 14, 14, 1]" = var_mean_30[1];  var_mean_30 = None
    sub_48: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_155, getitem_94);  clone_155 = getitem_94 = None
    add_132: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_93, 1e-05);  getitem_93 = None
    rsqrt_30: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    mul_129: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_30);  sub_48 = rsqrt_30 = None
    mul_130: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_129, arg188_1);  mul_129 = arg188_1 = None
    add_133: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_130, arg189_1);  mul_130 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_248: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_133, [1568, 384]);  add_133 = None
    permute_136: "f32[384, 1152]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    mm_19: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_248, permute_136);  view_248 = permute_136 = None
    view_249: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_19, [8, 14, 14, 1152]);  mm_19 = None
    view_250: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_249, [8, 196, 3, 12, 32]);  view_249 = None
    permute_137: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_250, [2, 0, 3, 1, 4]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
    getitem_95: "f32[8, 12, 196, 32]" = unbind_11[0]
    getitem_96: "f32[8, 12, 196, 32]" = unbind_11[1]
    getitem_97: "f32[8, 12, 196, 32]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_52: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_95, [8, 12, 196, 32]);  getitem_95 = None
    clone_156: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_251: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_156, [96, 196, 32]);  clone_156 = None
    permute_138: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_96, [0, 1, 3, 2]);  getitem_96 = None
    expand_53: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_138, [8, 12, 32, 196]);  permute_138 = None
    clone_157: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_252: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_157, [96, 32, 196]);  clone_157 = None
    bmm_26: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
    view_253: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_26, [8, 12, 196, 196]);  bmm_26 = None
    mul_131: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_253, 0.1767766952966369);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_15: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_131, [-1], True)
    sub_49: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_131, amax_15);  mul_131 = amax_15 = None
    exp_15: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_16: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_54: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_15, [8, 12, 196, 196]);  div_15 = None
    view_254: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_54, [96, 196, 196]);  expand_54 = None
    expand_55: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_97, [8, 12, 196, 32]);  getitem_97 = None
    clone_159: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_255: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_159, [96, 196, 32]);  clone_159 = None
    bmm_27: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
    view_256: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_27, [8, 12, 196, 32]);  bmm_27 = None
    permute_139: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_160: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_257: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_160, [8, 14, 14, 384]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_258: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_257, [1568, 384]);  view_257 = None
    permute_140: "f32[384, 384]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[1568, 384]" = torch.ops.aten.mm.default(view_258, permute_140);  view_258 = permute_140 = None
    add_tensor_15: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_15, arg192_1);  mm_default_15 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_259: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 14, 14, 384]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_134: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_131, view_259);  add_131 = view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_162: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_134, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_162, [3], correction = 0, keepdim = True)
    getitem_98: "f32[8, 14, 14, 1]" = var_mean_31[0]
    getitem_99: "f32[8, 14, 14, 1]" = var_mean_31[1];  var_mean_31 = None
    sub_50: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_162, getitem_99);  clone_162 = getitem_99 = None
    add_135: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_31: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    mul_132: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_31);  sub_50 = rsqrt_31 = None
    mul_133: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_132, arg193_1);  mul_132 = arg193_1 = None
    add_136: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_133, arg194_1);  mul_133 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_260: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_136, [1568, 384]);  add_136 = None
    permute_141: "f32[384, 1152]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_260, permute_141);  view_260 = permute_141 = None
    add_tensor_14: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_14, arg196_1);  mm_default_14 = arg196_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_261: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 14, 14, 1152]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_134: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_135: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_15: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_137: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_136: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_134, add_137);  mul_134 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_262: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_136, [1568, 1152]);  mul_136 = None
    permute_142: "f32[1152, 384]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[1568, 384]" = torch.ops.aten.mm.default(view_262, permute_142);  view_262 = permute_142 = None
    add_tensor_13: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_13, arg198_1);  mm_default_13 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_263: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 14, 14, 384]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_138: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_134, view_263);  add_134 = view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_165: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_138, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_165, [3], correction = 0, keepdim = True)
    getitem_100: "f32[8, 14, 14, 1]" = var_mean_32[0]
    getitem_101: "f32[8, 14, 14, 1]" = var_mean_32[1];  var_mean_32 = None
    sub_51: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_165, getitem_101);  clone_165 = getitem_101 = None
    add_139: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_32: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    mul_137: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_32);  sub_51 = rsqrt_32 = None
    mul_138: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_137, arg199_1);  mul_137 = arg199_1 = None
    add_140: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_138, arg200_1);  mul_138 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_264: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_140, [1568, 384]);  add_140 = None
    permute_143: "f32[384, 1152]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    mm_20: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_264, permute_143);  view_264 = permute_143 = None
    view_265: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_20, [8, 14, 14, 1152]);  mm_20 = None
    view_266: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_265, [8, 196, 3, 12, 32]);  view_265 = None
    permute_144: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_266, [2, 0, 3, 1, 4]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_12 = torch.ops.aten.unbind.int(permute_144);  permute_144 = None
    getitem_102: "f32[8, 12, 196, 32]" = unbind_12[0]
    getitem_103: "f32[8, 12, 196, 32]" = unbind_12[1]
    getitem_104: "f32[8, 12, 196, 32]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_56: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_102, [8, 12, 196, 32]);  getitem_102 = None
    clone_166: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_267: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_166, [96, 196, 32]);  clone_166 = None
    permute_145: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_103, [0, 1, 3, 2]);  getitem_103 = None
    expand_57: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_145, [8, 12, 32, 196]);  permute_145 = None
    clone_167: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_268: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_167, [96, 32, 196]);  clone_167 = None
    bmm_28: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_267, view_268);  view_267 = view_268 = None
    view_269: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_28, [8, 12, 196, 196]);  bmm_28 = None
    mul_139: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_269, 0.1767766952966369);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_16: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_139, [-1], True)
    sub_52: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_139, amax_16);  mul_139 = amax_16 = None
    exp_16: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_17: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_58: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_16, [8, 12, 196, 196]);  div_16 = None
    view_270: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_58, [96, 196, 196]);  expand_58 = None
    expand_59: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_104, [8, 12, 196, 32]);  getitem_104 = None
    clone_169: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_271: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_169, [96, 196, 32]);  clone_169 = None
    bmm_29: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_270, view_271);  view_270 = view_271 = None
    view_272: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_29, [8, 12, 196, 32]);  bmm_29 = None
    permute_146: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    clone_170: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    view_273: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_170, [8, 14, 14, 384]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_274: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_273, [1568, 384]);  view_273 = None
    permute_147: "f32[384, 384]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[1568, 384]" = torch.ops.aten.mm.default(view_274, permute_147);  view_274 = permute_147 = None
    add_tensor_12: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_12, arg203_1);  mm_default_12 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_275: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 14, 14, 384]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_141: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_138, view_275);  add_138 = view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_172: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_141, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_172, [3], correction = 0, keepdim = True)
    getitem_105: "f32[8, 14, 14, 1]" = var_mean_33[0]
    getitem_106: "f32[8, 14, 14, 1]" = var_mean_33[1];  var_mean_33 = None
    sub_53: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_172, getitem_106);  clone_172 = getitem_106 = None
    add_142: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_105, 1e-05);  getitem_105 = None
    rsqrt_33: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    mul_140: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_33);  sub_53 = rsqrt_33 = None
    mul_141: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_140, arg204_1);  mul_140 = arg204_1 = None
    add_143: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_141, arg205_1);  mul_141 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_276: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_143, [1568, 384]);  add_143 = None
    permute_148: "f32[384, 1152]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_276, permute_148);  view_276 = permute_148 = None
    add_tensor_11: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_11, arg207_1);  mm_default_11 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_277: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 14, 14, 1152]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_142: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, 0.5)
    mul_143: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, 0.7071067811865476);  view_277 = None
    erf_16: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
    add_144: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_144: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_142, add_144);  mul_142 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_278: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_144, [1568, 1152]);  mul_144 = None
    permute_149: "f32[1152, 384]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[1568, 384]" = torch.ops.aten.mm.default(view_278, permute_149);  view_278 = permute_149 = None
    add_tensor_10: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_10, arg209_1);  mm_default_10 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_279: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 14, 14, 384]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_145: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_141, view_279);  add_141 = view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_175: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_145, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_175, [3], correction = 0, keepdim = True)
    getitem_107: "f32[8, 14, 14, 1]" = var_mean_34[0]
    getitem_108: "f32[8, 14, 14, 1]" = var_mean_34[1];  var_mean_34 = None
    sub_54: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_175, getitem_108);  clone_175 = getitem_108 = None
    add_146: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_107, 1e-05);  getitem_107 = None
    rsqrt_34: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    mul_145: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_34);  sub_54 = rsqrt_34 = None
    mul_146: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_145, arg210_1);  mul_145 = arg210_1 = None
    add_147: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_146, arg211_1);  mul_146 = arg211_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_280: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_147, [1568, 384]);  add_147 = None
    permute_150: "f32[384, 1152]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
    mm_21: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_280, permute_150);  view_280 = permute_150 = None
    view_281: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(mm_21, [8, 14, 14, 1152]);  mm_21 = None
    view_282: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.reshape.default(view_281, [8, 196, 3, 12, 32]);  view_281 = None
    permute_151: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_282, [2, 0, 3, 1, 4]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_13 = torch.ops.aten.unbind.int(permute_151);  permute_151 = None
    getitem_109: "f32[8, 12, 196, 32]" = unbind_13[0]
    getitem_110: "f32[8, 12, 196, 32]" = unbind_13[1]
    getitem_111: "f32[8, 12, 196, 32]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_60: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_109, [8, 12, 196, 32]);  getitem_109 = None
    clone_176: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_283: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_176, [96, 196, 32]);  clone_176 = None
    permute_152: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_110, [0, 1, 3, 2]);  getitem_110 = None
    expand_61: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_152, [8, 12, 32, 196]);  permute_152 = None
    clone_177: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_284: "f32[96, 32, 196]" = torch.ops.aten.reshape.default(clone_177, [96, 32, 196]);  clone_177 = None
    bmm_30: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_283, view_284);  view_283 = view_284 = None
    view_285: "f32[8, 12, 196, 196]" = torch.ops.aten.reshape.default(bmm_30, [8, 12, 196, 196]);  bmm_30 = None
    mul_147: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_285, 0.1767766952966369);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_17: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_147, [-1], True)
    sub_55: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_147, amax_17);  mul_147 = amax_17 = None
    exp_17: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_18: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_62: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(div_17, [8, 12, 196, 196]);  div_17 = None
    view_286: "f32[96, 196, 196]" = torch.ops.aten.reshape.default(expand_62, [96, 196, 196]);  expand_62 = None
    expand_63: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_111, [8, 12, 196, 32]);  getitem_111 = None
    clone_179: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_287: "f32[96, 196, 32]" = torch.ops.aten.reshape.default(clone_179, [96, 196, 32]);  clone_179 = None
    bmm_31: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_286, view_287);  view_286 = view_287 = None
    view_288: "f32[8, 12, 196, 32]" = torch.ops.aten.reshape.default(bmm_31, [8, 12, 196, 32]);  bmm_31 = None
    permute_153: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
    clone_180: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    view_289: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_180, [8, 14, 14, 384]);  clone_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_290: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_289, [1568, 384]);  view_289 = None
    permute_154: "f32[384, 384]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[1568, 384]" = torch.ops.aten.mm.default(view_290, permute_154);  view_290 = permute_154 = None
    add_tensor_9: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_9, arg214_1);  mm_default_9 = arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_291: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 14, 14, 384]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_148: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_145, view_291);  add_145 = view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_182: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_148, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_182, [3], correction = 0, keepdim = True)
    getitem_112: "f32[8, 14, 14, 1]" = var_mean_35[0]
    getitem_113: "f32[8, 14, 14, 1]" = var_mean_35[1];  var_mean_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:633, code: cls_tokens = self.cls_token.expand(B, -1, -1)
    expand_64: "f32[8, 1, 384]" = torch.ops.aten.expand.default(arg1_1, [8, -1, -1]);  arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sub_56: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_182, getitem_113);  clone_182 = getitem_113 = None
    add_149: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_35: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    mul_148: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_35);  sub_56 = rsqrt_35 = None
    mul_149: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_148, arg215_1);  mul_148 = arg215_1 = None
    add_150: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_149, arg216_1);  mul_149 = arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_292: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_150, [1568, 384]);  add_150 = None
    permute_155: "f32[384, 1152]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_292, permute_155);  view_292 = permute_155 = None
    add_tensor_8: "f32[1568, 1152]" = torch.ops.aten.add.Tensor(mm_default_8, arg218_1);  mm_default_8 = arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_293: "f32[8, 14, 14, 1152]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 14, 14, 1152]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_150: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, 0.5)
    mul_151: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, 0.7071067811865476);  view_293 = None
    erf_17: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
    add_151: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_152: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_150, add_151);  mul_150 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_294: "f32[1568, 1152]" = torch.ops.aten.reshape.default(mul_152, [1568, 1152]);  mul_152 = None
    permute_156: "f32[1152, 384]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[1568, 384]" = torch.ops.aten.mm.default(view_294, permute_156);  view_294 = permute_156 = None
    add_tensor_7: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_7, arg220_1);  mm_default_7 = arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_295: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 14, 14, 384]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_152: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_148, view_295);  add_148 = view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:628, code: x = x.reshape(B, -1, C)
    view_296: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_152, [8, 196, 384]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:634, code: x = torch.cat([cls_tokens, x], dim=1)
    cat: "f32[8, 197, 384]" = torch.ops.aten.cat.default([expand_64, view_296], 1);  expand_64 = view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    var_mean_36 = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 197, 1]" = var_mean_36[0]
    getitem_115: "f32[8, 197, 1]" = var_mean_36[1];  var_mean_36 = None
    sub_57: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat, getitem_115);  getitem_115 = None
    add_153: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_36: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    mul_153: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_36);  sub_57 = rsqrt_36 = None
    mul_154: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_153, arg221_1);  mul_153 = arg221_1 = None
    add_154: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_154, arg222_1);  mul_154 = arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_297: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_154, [1576, 384])
    permute_157: "f32[384, 768]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_297, permute_157);  view_297 = permute_157 = None
    view_298: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_22, [8, 197, 768]);  mm_22 = None
    view_299: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.reshape.default(view_298, [8, 197, 2, 12, 32]);  view_298 = None
    permute_158: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.permute.default(view_299, [2, 0, 3, 1, 4]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    unbind_14 = torch.ops.aten.unbind.int(permute_158);  permute_158 = None
    getitem_116: "f32[8, 12, 197, 32]" = unbind_14[0]
    getitem_117: "f32[8, 12, 197, 32]" = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    slice_10: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(cat, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    slice_12: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_154, 1, 0, 1);  add_154 = None
    view_300: "f32[8, 384]" = torch.ops.aten.reshape.default(slice_12, [8, 384]);  slice_12 = None
    permute_159: "f32[384, 384]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    mm_23: "f32[8, 384]" = torch.ops.aten.mm.default(view_300, permute_159);  view_300 = permute_159 = None
    view_301: "f32[8, 1, 384]" = torch.ops.aten.reshape.default(mm_23, [8, 1, 384]);  mm_23 = None
    view_302: "f32[8, 12, 1, 32]" = torch.ops.aten.reshape.default(view_301, [8, 12, 1, 32]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    mul_155: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_302, 0.1767766952966369);  view_302 = None
    expand_65: "f32[8, 12, 1, 32]" = torch.ops.aten.expand.default(mul_155, [8, 12, 1, 32]);  mul_155 = None
    view_303: "f32[96, 1, 32]" = torch.ops.aten.reshape.default(expand_65, [96, 1, 32]);  expand_65 = None
    permute_160: "f32[8, 12, 32, 197]" = torch.ops.aten.permute.default(getitem_116, [0, 1, 3, 2]);  getitem_116 = None
    expand_66: "f32[8, 12, 32, 197]" = torch.ops.aten.expand.default(permute_160, [8, 12, 32, 197]);  permute_160 = None
    clone_185: "f32[8, 12, 32, 197]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_304: "f32[96, 32, 197]" = torch.ops.aten.reshape.default(clone_185, [96, 32, 197]);  clone_185 = None
    bmm_32: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_303, view_304);  view_303 = view_304 = None
    view_305: "f32[8, 12, 1, 197]" = torch.ops.aten.reshape.default(bmm_32, [8, 12, 1, 197]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    amax_18: "f32[8, 12, 1, 1]" = torch.ops.aten.amax.default(view_305, [-1], True)
    sub_58: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(view_305, amax_18);  view_305 = amax_18 = None
    exp_18: "f32[8, 12, 1, 197]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_19: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[8, 12, 1, 197]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    expand_67: "f32[8, 12, 1, 197]" = torch.ops.aten.expand.default(div_18, [8, 12, 1, 197]);  div_18 = None
    view_306: "f32[96, 1, 197]" = torch.ops.aten.reshape.default(expand_67, [96, 1, 197]);  expand_67 = None
    expand_68: "f32[8, 12, 197, 32]" = torch.ops.aten.expand.default(getitem_117, [8, 12, 197, 32]);  getitem_117 = None
    clone_187: "f32[8, 12, 197, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_307: "f32[96, 197, 32]" = torch.ops.aten.reshape.default(clone_187, [96, 197, 32]);  clone_187 = None
    bmm_33: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_306, view_307);  view_306 = view_307 = None
    view_308: "f32[8, 12, 1, 32]" = torch.ops.aten.reshape.default(bmm_33, [8, 12, 1, 32]);  bmm_33 = None
    permute_161: "f32[8, 1, 12, 32]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
    view_309: "f32[8, 1, 384]" = torch.ops.aten.reshape.default(permute_161, [8, 1, 384]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_310: "f32[8, 384]" = torch.ops.aten.reshape.default(view_309, [8, 384]);  view_309 = None
    permute_162: "f32[384, 384]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[8, 384]" = torch.ops.aten.mm.default(view_310, permute_162);  view_310 = permute_162 = None
    add_tensor_6: "f32[8, 384]" = torch.ops.aten.add.Tensor(mm_default_6, arg226_1);  mm_default_6 = arg226_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_311: "f32[8, 1, 384]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 1, 384]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_155: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_10, view_311);  slice_10 = view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    var_mean_37 = torch.ops.aten.var_mean.correction(add_155, [2], correction = 0, keepdim = True)
    getitem_118: "f32[8, 1, 1]" = var_mean_37[0]
    getitem_119: "f32[8, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    sub_59: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_155, getitem_119);  getitem_119 = None
    add_156: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_37: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    mul_156: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_37);  sub_59 = rsqrt_37 = None
    mul_157: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_156, arg227_1);  mul_156 = arg227_1 = None
    add_157: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(mul_157, arg228_1);  mul_157 = arg228_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_312: "f32[8, 384]" = torch.ops.aten.reshape.default(add_157, [8, 384]);  add_157 = None
    permute_163: "f32[384, 1152]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[8, 1152]" = torch.ops.aten.mm.default(view_312, permute_163);  view_312 = permute_163 = None
    add_tensor_5: "f32[8, 1152]" = torch.ops.aten.add.Tensor(mm_default_5, arg230_1);  mm_default_5 = arg230_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_313: "f32[8, 1, 1152]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 1, 1152]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_158: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, 0.5)
    mul_159: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, 0.7071067811865476);  view_313 = None
    erf_18: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
    add_158: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_160: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_158, add_158);  mul_158 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_314: "f32[8, 1152]" = torch.ops.aten.reshape.default(mul_160, [8, 1152]);  mul_160 = None
    permute_164: "f32[1152, 384]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[8, 384]" = torch.ops.aten.mm.default(view_314, permute_164);  view_314 = permute_164 = None
    add_tensor_4: "f32[8, 384]" = torch.ops.aten.add.Tensor(mm_default_4, arg232_1);  mm_default_4 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_315: "f32[8, 1, 384]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 1, 384]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_159: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(add_155, view_315);  add_155 = view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_15: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(cat, 1, 1, 9223372036854775807);  cat = None
    cat_1: "f32[8, 197, 384]" = torch.ops.aten.cat.default([add_159, slice_15], 1);  add_159 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    var_mean_38 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 197, 1]" = var_mean_38[0]
    getitem_121: "f32[8, 197, 1]" = var_mean_38[1];  var_mean_38 = None
    sub_60: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_1, getitem_121);  getitem_121 = None
    add_160: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_38: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    mul_161: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_38);  sub_60 = rsqrt_38 = None
    mul_162: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_161, arg233_1);  mul_161 = arg233_1 = None
    add_161: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_162, arg234_1);  mul_162 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_316: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_161, [1576, 384])
    permute_165: "f32[384, 768]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
    mm_24: "f32[1576, 768]" = torch.ops.aten.mm.default(view_316, permute_165);  view_316 = permute_165 = None
    view_317: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_24, [8, 197, 768]);  mm_24 = None
    view_318: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.reshape.default(view_317, [8, 197, 2, 12, 32]);  view_317 = None
    permute_166: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.permute.default(view_318, [2, 0, 3, 1, 4]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    unbind_15 = torch.ops.aten.unbind.int(permute_166);  permute_166 = None
    getitem_122: "f32[8, 12, 197, 32]" = unbind_15[0]
    getitem_123: "f32[8, 12, 197, 32]" = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    slice_17: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(cat_1, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    slice_19: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_161, 1, 0, 1);  add_161 = None
    view_319: "f32[8, 384]" = torch.ops.aten.reshape.default(slice_19, [8, 384]);  slice_19 = None
    permute_167: "f32[384, 384]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    mm_25: "f32[8, 384]" = torch.ops.aten.mm.default(view_319, permute_167);  view_319 = permute_167 = None
    view_320: "f32[8, 1, 384]" = torch.ops.aten.reshape.default(mm_25, [8, 1, 384]);  mm_25 = None
    view_321: "f32[8, 12, 1, 32]" = torch.ops.aten.reshape.default(view_320, [8, 12, 1, 32]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    mul_163: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_321, 0.1767766952966369);  view_321 = None
    expand_69: "f32[8, 12, 1, 32]" = torch.ops.aten.expand.default(mul_163, [8, 12, 1, 32]);  mul_163 = None
    view_322: "f32[96, 1, 32]" = torch.ops.aten.reshape.default(expand_69, [96, 1, 32]);  expand_69 = None
    permute_168: "f32[8, 12, 32, 197]" = torch.ops.aten.permute.default(getitem_122, [0, 1, 3, 2]);  getitem_122 = None
    expand_70: "f32[8, 12, 32, 197]" = torch.ops.aten.expand.default(permute_168, [8, 12, 32, 197]);  permute_168 = None
    clone_191: "f32[8, 12, 32, 197]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_323: "f32[96, 32, 197]" = torch.ops.aten.reshape.default(clone_191, [96, 32, 197]);  clone_191 = None
    bmm_34: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_322, view_323);  view_322 = view_323 = None
    view_324: "f32[8, 12, 1, 197]" = torch.ops.aten.reshape.default(bmm_34, [8, 12, 1, 197]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    amax_19: "f32[8, 12, 1, 1]" = torch.ops.aten.amax.default(view_324, [-1], True)
    sub_61: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(view_324, amax_19);  view_324 = amax_19 = None
    exp_19: "f32[8, 12, 1, 197]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_20: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[8, 12, 1, 197]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    expand_71: "f32[8, 12, 1, 197]" = torch.ops.aten.expand.default(div_19, [8, 12, 1, 197]);  div_19 = None
    view_325: "f32[96, 1, 197]" = torch.ops.aten.reshape.default(expand_71, [96, 1, 197]);  expand_71 = None
    expand_72: "f32[8, 12, 197, 32]" = torch.ops.aten.expand.default(getitem_123, [8, 12, 197, 32]);  getitem_123 = None
    clone_193: "f32[8, 12, 197, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_326: "f32[96, 197, 32]" = torch.ops.aten.reshape.default(clone_193, [96, 197, 32]);  clone_193 = None
    bmm_35: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_325, view_326);  view_325 = view_326 = None
    view_327: "f32[8, 12, 1, 32]" = torch.ops.aten.reshape.default(bmm_35, [8, 12, 1, 32]);  bmm_35 = None
    permute_169: "f32[8, 1, 12, 32]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    view_328: "f32[8, 1, 384]" = torch.ops.aten.reshape.default(permute_169, [8, 1, 384]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_329: "f32[8, 384]" = torch.ops.aten.reshape.default(view_328, [8, 384]);  view_328 = None
    permute_170: "f32[384, 384]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[8, 384]" = torch.ops.aten.mm.default(view_329, permute_170);  view_329 = permute_170 = None
    add_tensor_3: "f32[8, 384]" = torch.ops.aten.add.Tensor(mm_default_3, arg238_1);  mm_default_3 = arg238_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_330: "f32[8, 1, 384]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 1, 384]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_162: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_17, view_330);  slice_17 = view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    var_mean_39 = torch.ops.aten.var_mean.correction(add_162, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 1, 1]" = var_mean_39[0]
    getitem_125: "f32[8, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    sub_62: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_162, getitem_125);  getitem_125 = None
    add_163: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_39: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    mul_164: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_39);  sub_62 = rsqrt_39 = None
    mul_165: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_164, arg239_1);  mul_164 = arg239_1 = None
    add_164: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(mul_165, arg240_1);  mul_165 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_331: "f32[8, 384]" = torch.ops.aten.reshape.default(add_164, [8, 384]);  add_164 = None
    permute_171: "f32[384, 1152]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[8, 1152]" = torch.ops.aten.mm.default(view_331, permute_171);  view_331 = permute_171 = None
    add_tensor_2: "f32[8, 1152]" = torch.ops.aten.add.Tensor(mm_default_2, arg242_1);  mm_default_2 = arg242_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_332: "f32[8, 1, 1152]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 1, 1152]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_166: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.5)
    mul_167: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476);  view_332 = None
    erf_19: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_165: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_168: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_166, add_165);  mul_166 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_333: "f32[8, 1152]" = torch.ops.aten.reshape.default(mul_168, [8, 1152]);  mul_168 = None
    permute_172: "f32[1152, 384]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[8, 384]" = torch.ops.aten.mm.default(view_333, permute_172);  view_333 = permute_172 = None
    add_tensor_1: "f32[8, 384]" = torch.ops.aten.add.Tensor(mm_default_1, arg244_1);  mm_default_1 = arg244_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_334: "f32[8, 1, 384]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 1, 384]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_166: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(add_162, view_334);  add_162 = view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_22: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(cat_1, 1, 1, 9223372036854775807);  cat_1 = None
    cat_2: "f32[8, 197, 384]" = torch.ops.aten.cat.default([add_166, slice_22], 1);  add_166 = slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:703, code: x = self.norm(x)
    var_mean_40 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 197, 1]" = var_mean_40[0]
    getitem_127: "f32[8, 197, 1]" = var_mean_40[1];  var_mean_40 = None
    sub_63: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_2, getitem_127);  cat_2 = getitem_127 = None
    add_167: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
    rsqrt_40: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    mul_169: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_40);  sub_63 = rsqrt_40 = None
    mul_170: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_169, arg245_1);  mul_169 = arg245_1 = None
    add_168: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_170, arg246_1);  mul_170 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:719, code: aux = self.aux_head(x[:, 1:])
    slice_25: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_168, 1, 1, 9223372036854775807)
    clone_198: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_25, memory_format = torch.contiguous_format);  slice_25 = None
    view_335: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_198, [1568, 384]);  clone_198 = None
    permute_174: "f32[384, 1000]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
    mm_26: "f32[1568, 1000]" = torch.ops.aten.mm.default(view_335, permute_174);  view_335 = permute_174 = None
    view_336: "f32[8, 196, 1000]" = torch.ops.aten.reshape.default(mm_26, [8, 196, 1000]);  mm_26 = None
    add_169: "f32[8, 196, 1000]" = torch.ops.aten.add.Tensor(view_336, arg250_1);  view_336 = arg250_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:720, code: out = out + 0.5 * aux.max(1)[0]
    max_1 = torch.ops.aten.max.dim(add_169, 1);  add_169 = None
    getitem_128: "f32[8, 1000]" = max_1[0];  max_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:710, code: out = x[:, 0]
    select: "f32[8, 384]" = torch.ops.aten.select.int(add_168, 1, 0);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:716, code: out = self.head(out)
    permute_173: "f32[384, 1000]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[8, 1000]" = torch.ops.aten.mm.default(select, permute_173);  select = permute_173 = None
    add_tensor: "f32[8, 1000]" = torch.ops.aten.add.Tensor(mm_default, arg248_1);  mm_default = arg248_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:720, code: out = out + 0.5 * aux.max(1)[0]
    mul_171: "f32[8, 1000]" = torch.ops.aten.mul.Tensor(getitem_128, 0.5);  getitem_128 = None
    add_170: "f32[8, 1000]" = torch.ops.aten.add.Tensor(add_tensor, mul_171);  add_tensor = mul_171 = None
    return (add_170,)
    