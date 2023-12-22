from __future__ import annotations



def forward(self, arg0_1: "f32[16]", arg1_1: "f32[16]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[32]", arg7_1: "f32[32]", arg8_1: "f32[128]", arg9_1: "f32[128]", arg10_1: "f32[128]", arg11_1: "f32[128]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[256]", arg15_1: "f32[256]", arg16_1: "f32[256]", arg17_1: "f32[256]", arg18_1: "f32[64]", arg19_1: "f32[64]", arg20_1: "f32[256]", arg21_1: "f32[256]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[256]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[256]", arg30_1: "f32[96]", arg31_1: "f32[96]", arg32_1: "f32[96]", arg33_1: "f32[96]", arg34_1: "f32[96]", arg35_1: "f32[96]", arg36_1: "f32[96]", arg37_1: "f32[96]", arg38_1: "f32[384]", arg39_1: "f32[384]", arg40_1: "f32[384]", arg41_1: "f32[384]", arg42_1: "f32[128]", arg43_1: "f32[128]", arg44_1: "f32[128]", arg45_1: "f32[128]", arg46_1: "f32[128]", arg47_1: "f32[128]", arg48_1: "f32[128]", arg49_1: "f32[128]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[512]", arg53_1: "f32[512]", arg54_1: "f32[160]", arg55_1: "f32[160]", arg56_1: "f32[160]", arg57_1: "f32[160]", arg58_1: "f32[160]", arg59_1: "f32[160]", arg60_1: "f32[160]", arg61_1: "f32[160]", arg62_1: "f32[640]", arg63_1: "f32[640]", arg64_1: "f32[16, 3, 3, 3]", arg65_1: "f32[64, 16, 1, 1]", arg66_1: "f32[64, 1, 3, 3]", arg67_1: "f32[32, 64, 1, 1]", arg68_1: "f32[128, 32, 1, 1]", arg69_1: "f32[128, 1, 3, 3]", arg70_1: "f32[64, 128, 1, 1]", arg71_1: "f32[256, 64, 1, 1]", arg72_1: "f32[256, 1, 3, 3]", arg73_1: "f32[64, 256, 1, 1]", arg74_1: "f32[256, 64, 1, 1]", arg75_1: "f32[256, 1, 3, 3]", arg76_1: "f32[64, 256, 1, 1]", arg77_1: "f32[256, 64, 1, 1]", arg78_1: "f32[256, 1, 3, 3]", arg79_1: "f32[96, 256, 1, 1]", arg80_1: "f32[96, 96, 3, 3]", arg81_1: "f32[144, 96, 1, 1]", arg82_1: "f32[144]", arg83_1: "f32[144]", arg84_1: "f32[432, 144]", arg85_1: "f32[432]", arg86_1: "f32[144, 144]", arg87_1: "f32[144]", arg88_1: "f32[144]", arg89_1: "f32[144]", arg90_1: "f32[288, 144]", arg91_1: "f32[288]", arg92_1: "f32[144, 288]", arg93_1: "f32[144]", arg94_1: "f32[144]", arg95_1: "f32[144]", arg96_1: "f32[432, 144]", arg97_1: "f32[432]", arg98_1: "f32[144, 144]", arg99_1: "f32[144]", arg100_1: "f32[144]", arg101_1: "f32[144]", arg102_1: "f32[288, 144]", arg103_1: "f32[288]", arg104_1: "f32[144, 288]", arg105_1: "f32[144]", arg106_1: "f32[144]", arg107_1: "f32[144]", arg108_1: "f32[96, 144, 1, 1]", arg109_1: "f32[96, 192, 3, 3]", arg110_1: "f32[384, 96, 1, 1]", arg111_1: "f32[384, 1, 3, 3]", arg112_1: "f32[128, 384, 1, 1]", arg113_1: "f32[128, 128, 3, 3]", arg114_1: "f32[192, 128, 1, 1]", arg115_1: "f32[192]", arg116_1: "f32[192]", arg117_1: "f32[576, 192]", arg118_1: "f32[576]", arg119_1: "f32[192, 192]", arg120_1: "f32[192]", arg121_1: "f32[192]", arg122_1: "f32[192]", arg123_1: "f32[384, 192]", arg124_1: "f32[384]", arg125_1: "f32[192, 384]", arg126_1: "f32[192]", arg127_1: "f32[192]", arg128_1: "f32[192]", arg129_1: "f32[576, 192]", arg130_1: "f32[576]", arg131_1: "f32[192, 192]", arg132_1: "f32[192]", arg133_1: "f32[192]", arg134_1: "f32[192]", arg135_1: "f32[384, 192]", arg136_1: "f32[384]", arg137_1: "f32[192, 384]", arg138_1: "f32[192]", arg139_1: "f32[192]", arg140_1: "f32[192]", arg141_1: "f32[576, 192]", arg142_1: "f32[576]", arg143_1: "f32[192, 192]", arg144_1: "f32[192]", arg145_1: "f32[192]", arg146_1: "f32[192]", arg147_1: "f32[384, 192]", arg148_1: "f32[384]", arg149_1: "f32[192, 384]", arg150_1: "f32[192]", arg151_1: "f32[192]", arg152_1: "f32[192]", arg153_1: "f32[576, 192]", arg154_1: "f32[576]", arg155_1: "f32[192, 192]", arg156_1: "f32[192]", arg157_1: "f32[192]", arg158_1: "f32[192]", arg159_1: "f32[384, 192]", arg160_1: "f32[384]", arg161_1: "f32[192, 384]", arg162_1: "f32[192]", arg163_1: "f32[192]", arg164_1: "f32[192]", arg165_1: "f32[128, 192, 1, 1]", arg166_1: "f32[128, 256, 3, 3]", arg167_1: "f32[512, 128, 1, 1]", arg168_1: "f32[512, 1, 3, 3]", arg169_1: "f32[160, 512, 1, 1]", arg170_1: "f32[160, 160, 3, 3]", arg171_1: "f32[240, 160, 1, 1]", arg172_1: "f32[240]", arg173_1: "f32[240]", arg174_1: "f32[720, 240]", arg175_1: "f32[720]", arg176_1: "f32[240, 240]", arg177_1: "f32[240]", arg178_1: "f32[240]", arg179_1: "f32[240]", arg180_1: "f32[480, 240]", arg181_1: "f32[480]", arg182_1: "f32[240, 480]", arg183_1: "f32[240]", arg184_1: "f32[240]", arg185_1: "f32[240]", arg186_1: "f32[720, 240]", arg187_1: "f32[720]", arg188_1: "f32[240, 240]", arg189_1: "f32[240]", arg190_1: "f32[240]", arg191_1: "f32[240]", arg192_1: "f32[480, 240]", arg193_1: "f32[480]", arg194_1: "f32[240, 480]", arg195_1: "f32[240]", arg196_1: "f32[240]", arg197_1: "f32[240]", arg198_1: "f32[720, 240]", arg199_1: "f32[720]", arg200_1: "f32[240, 240]", arg201_1: "f32[240]", arg202_1: "f32[240]", arg203_1: "f32[240]", arg204_1: "f32[480, 240]", arg205_1: "f32[480]", arg206_1: "f32[240, 480]", arg207_1: "f32[240]", arg208_1: "f32[240]", arg209_1: "f32[240]", arg210_1: "f32[160, 240, 1, 1]", arg211_1: "f32[160, 320, 3, 3]", arg212_1: "f32[640, 160, 1, 1]", arg213_1: "f32[1000, 640]", arg214_1: "f32[1000]", arg215_1: "f32[16]", arg216_1: "f32[16]", arg217_1: "f32[64]", arg218_1: "f32[64]", arg219_1: "f32[64]", arg220_1: "f32[64]", arg221_1: "f32[32]", arg222_1: "f32[32]", arg223_1: "f32[128]", arg224_1: "f32[128]", arg225_1: "f32[128]", arg226_1: "f32[128]", arg227_1: "f32[64]", arg228_1: "f32[64]", arg229_1: "f32[256]", arg230_1: "f32[256]", arg231_1: "f32[256]", arg232_1: "f32[256]", arg233_1: "f32[64]", arg234_1: "f32[64]", arg235_1: "f32[256]", arg236_1: "f32[256]", arg237_1: "f32[256]", arg238_1: "f32[256]", arg239_1: "f32[64]", arg240_1: "f32[64]", arg241_1: "f32[256]", arg242_1: "f32[256]", arg243_1: "f32[256]", arg244_1: "f32[256]", arg245_1: "f32[96]", arg246_1: "f32[96]", arg247_1: "f32[96]", arg248_1: "f32[96]", arg249_1: "f32[96]", arg250_1: "f32[96]", arg251_1: "f32[96]", arg252_1: "f32[96]", arg253_1: "f32[384]", arg254_1: "f32[384]", arg255_1: "f32[384]", arg256_1: "f32[384]", arg257_1: "f32[128]", arg258_1: "f32[128]", arg259_1: "f32[128]", arg260_1: "f32[128]", arg261_1: "f32[128]", arg262_1: "f32[128]", arg263_1: "f32[128]", arg264_1: "f32[128]", arg265_1: "f32[512]", arg266_1: "f32[512]", arg267_1: "f32[512]", arg268_1: "f32[512]", arg269_1: "f32[160]", arg270_1: "f32[160]", arg271_1: "f32[160]", arg272_1: "f32[160]", arg273_1: "f32[160]", arg274_1: "f32[160]", arg275_1: "f32[160]", arg276_1: "f32[160]", arg277_1: "f32[640]", arg278_1: "f32[640]", arg279_1: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 16, 128, 128]" = torch.ops.aten.convolution.default(arg279_1, arg64_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg279_1 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    sub: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    add: "f32[16]" = torch.ops.aten.add.Tensor(arg216_1, 1e-05);  arg216_1 = None
    sqrt: "f32[16]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    mul_1: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid: "f32[8, 16, 128, 128]" = torch.ops.aten.sigmoid.default(add_1)
    mul_3: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(add_1, sigmoid);  add_1 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_3, arg65_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_3 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    sub_1: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    mul_5: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_1: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_3)
    mul_7: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_3, sigmoid_1);  add_3 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(mul_7, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  mul_7 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    sub_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    add_4: "f32[64]" = torch.ops.aten.add.Tensor(arg220_1, 1e-05);  arg220_1 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_8, -1);  mul_8 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    mul_9: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_10: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_21);  mul_9 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_23);  mul_10 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_2: "f32[8, 64, 128, 128]" = torch.ops.aten.sigmoid.default(add_5)
    mul_11: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_5, sigmoid_2);  add_5 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_11, arg67_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_11 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_24: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
    unsqueeze_25: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    sub_3: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    add_6: "f32[32]" = torch.ops.aten.add.Tensor(arg222_1, 1e-05);  arg222_1 = None
    sqrt_3: "f32[32]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_26: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_27: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    mul_13: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_29: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_14: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_29);  mul_13 = unsqueeze_29 = None
    unsqueeze_30: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_31: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_31);  mul_14 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(add_7, arg68_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_7 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_32: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
    unsqueeze_33: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    sub_4: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    add_8: "f32[128]" = torch.ops.aten.add.Tensor(arg224_1, 1e-05);  arg224_1 = None
    sqrt_4: "f32[128]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_34: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_35: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    mul_16: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_37: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_17: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_37);  mul_16 = unsqueeze_37 = None
    unsqueeze_38: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_39: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_39);  mul_17 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_3: "f32[8, 128, 128, 128]" = torch.ops.aten.sigmoid.default(add_9)
    mul_18: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_3);  add_9 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_18, arg69_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 128);  mul_18 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    sub_5: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    add_10: "f32[128]" = torch.ops.aten.add.Tensor(arg226_1, 1e-05);  arg226_1 = None
    sqrt_5: "f32[128]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_19: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_19, -1);  mul_19 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    mul_20: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_21: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_45);  mul_20 = unsqueeze_45 = None
    unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_47);  mul_21 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_4: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_11)
    mul_22: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_11, sigmoid_4);  add_11 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_22, arg70_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_22 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    sub_6: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
    sqrt_6: "f32[64]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_23: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    mul_24: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
    unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_25: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_53);  mul_24 = unsqueeze_53 = None
    unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_55);  mul_25 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_13, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
    unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    sub_7: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
    add_14: "f32[256]" = torch.ops.aten.add.Tensor(arg230_1, 1e-05);  arg230_1 = None
    sqrt_7: "f32[256]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_26: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_26, -1);  mul_26 = None
    unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    mul_27: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_61: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_28: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_61);  mul_27 = unsqueeze_61 = None
    unsqueeze_62: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_63: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_63);  mul_28 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_5: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_15)
    mul_29: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_15, sigmoid_5);  add_15 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_29, arg72_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256);  mul_29 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_64: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
    unsqueeze_65: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    sub_8: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
    add_16: "f32[256]" = torch.ops.aten.add.Tensor(arg232_1, 1e-05);  arg232_1 = None
    sqrt_8: "f32[256]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_66: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_67: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    mul_31: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_69: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_32: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_69);  mul_31 = unsqueeze_69 = None
    unsqueeze_70: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_71: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_71);  mul_32 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_17)
    mul_33: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_17, sigmoid_6);  add_17 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_33, arg73_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_33 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
    unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    sub_9: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
    add_18: "f32[64]" = torch.ops.aten.add.Tensor(arg234_1, 1e-05);  arg234_1 = None
    sqrt_9: "f32[64]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_9: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_34: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_34, -1);  mul_34 = None
    unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    mul_35: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_77: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_36: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_77);  mul_35 = unsqueeze_77 = None
    unsqueeze_78: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_79: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_19: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_36, unsqueeze_79);  mul_36 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_20: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_19, add_13);  add_19 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_20, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
    unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    sub_10: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
    add_21: "f32[256]" = torch.ops.aten.add.Tensor(arg236_1, 1e-05);  arg236_1 = None
    sqrt_10: "f32[256]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_10: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_37: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_37, -1);  mul_37 = None
    unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    mul_38: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_39: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_38, unsqueeze_85);  mul_38 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_22: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_87);  mul_39 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_7: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_22)
    mul_40: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_22, sigmoid_7);  add_22 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_40, arg75_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256);  mul_40 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
    unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    sub_11: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
    add_23: "f32[256]" = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
    sqrt_11: "f32[256]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_11: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_41: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_41, -1);  mul_41 = None
    unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    mul_42: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_93: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_43: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_93);  mul_42 = unsqueeze_93 = None
    unsqueeze_94: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_95: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_24: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_95);  mul_43 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_8: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_24)
    mul_44: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_8);  add_24 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_44, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_44 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_96: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
    unsqueeze_97: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    sub_12: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
    add_25: "f32[64]" = torch.ops.aten.add.Tensor(arg240_1, 1e-05);  arg240_1 = None
    sqrt_12: "f32[64]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
    reciprocal_12: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_45: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_98: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_99: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    mul_46: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
    unsqueeze_101: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_47: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_101);  mul_46 = unsqueeze_101 = None
    unsqueeze_102: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_103: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_26: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_103);  mul_47 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_27: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_26, add_20);  add_26 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(add_27, arg77_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_27 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_104: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
    unsqueeze_105: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    sub_13: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(arg242_1, 1e-05);  arg242_1 = None
    sqrt_13: "f32[256]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_13: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_48: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_106: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_107: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    mul_49: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_109: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_50: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_109);  mul_49 = unsqueeze_109 = None
    unsqueeze_110: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
    unsqueeze_111: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_29: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_111);  mul_50 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_9: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_29)
    mul_51: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_29, sigmoid_9);  add_29 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_51, arg78_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 256);  mul_51 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
    unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    sub_14: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
    add_30: "f32[256]" = torch.ops.aten.add.Tensor(arg244_1, 1e-05);  arg244_1 = None
    sqrt_14: "f32[256]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_14: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_52, -1);  mul_52 = None
    unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    mul_53: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_117: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_54: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_53, unsqueeze_117);  mul_53 = unsqueeze_117 = None
    unsqueeze_118: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_119: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_31: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_119);  mul_54 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_10: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_31)
    mul_55: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_31, sigmoid_10);  add_31 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(mul_55, arg79_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_55 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_120: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
    unsqueeze_121: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    sub_15: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
    add_32: "f32[96]" = torch.ops.aten.add.Tensor(arg246_1, 1e-05);  arg246_1 = None
    sqrt_15: "f32[96]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_15: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_56: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_122: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_56, -1);  mul_56 = None
    unsqueeze_123: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    mul_57: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_125: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_58: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_125);  mul_57 = unsqueeze_125 = None
    unsqueeze_126: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_127: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_33: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_127);  mul_58 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(add_33, arg80_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_128: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
    unsqueeze_129: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    sub_16: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
    add_34: "f32[96]" = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
    sqrt_16: "f32[96]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_16: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_59: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_130: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_59, -1);  mul_59 = None
    unsqueeze_131: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    mul_60: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_133: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_61: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_60, unsqueeze_133);  mul_60 = unsqueeze_133 = None
    unsqueeze_134: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
    unsqueeze_135: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_35: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_61, unsqueeze_135);  mul_61 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_11: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_35)
    mul_62: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_35, sigmoid_11);  add_35 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_17: "f32[8, 144, 32, 32]" = torch.ops.aten.convolution.default(mul_62, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_62 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    view: "f32[18432, 2, 16, 2]" = torch.ops.aten.reshape.default(convolution_17, [18432, 2, 16, 2]);  convolution_17 = None
    permute: "f32[18432, 16, 2, 2]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    clone: "f32[18432, 16, 2, 2]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    view_1: "f32[8, 144, 256, 4]" = torch.ops.aten.reshape.default(clone, [8, 144, 256, 4]);  clone = None
    permute_1: "f32[8, 4, 256, 144]" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
    clone_1: "f32[8, 4, 256, 144]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_2: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(clone_1, [32, 256, 144]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean = torch.ops.aten.var_mean.correction(view_2, [2], correction = 0, keepdim = True)
    getitem: "f32[32, 256, 1]" = var_mean[0]
    getitem_1: "f32[32, 256, 1]" = var_mean[1];  var_mean = None
    sub_17: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(view_2, getitem_1);  getitem_1 = None
    add_36: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    mul_63: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt);  sub_17 = rsqrt = None
    mul_64: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_63, arg82_1);  mul_63 = arg82_1 = None
    add_37: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_64, arg83_1);  mul_64 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_3: "f32[8192, 144]" = torch.ops.aten.reshape.default(add_37, [8192, 144]);  add_37 = None
    permute_2: "f32[144, 432]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm: "f32[8192, 432]" = torch.ops.aten.addmm.default(arg85_1, view_3, permute_2);  arg85_1 = view_3 = permute_2 = None
    view_4: "f32[32, 256, 432]" = torch.ops.aten.reshape.default(addmm, [32, 256, 432]);  addmm = None
    view_5: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.reshape.default(view_4, [32, 256, 3, 4, 36]);  view_4 = None
    permute_3: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_2: "f32[32, 4, 256, 36]" = unbind[0]
    getitem_3: "f32[32, 4, 256, 36]" = unbind[1]
    getitem_4: "f32[32, 4, 256, 36]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_2, getitem_3, getitem_4);  getitem_2 = getitem_3 = getitem_4 = None
    getitem_5: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_4: "f32[32, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_6: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(permute_4, [32, 256, 144]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_7: "f32[8192, 144]" = torch.ops.aten.reshape.default(view_6, [8192, 144]);  view_6 = None
    permute_5: "f32[144, 144]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_1: "f32[8192, 144]" = torch.ops.aten.addmm.default(arg87_1, view_7, permute_5);  arg87_1 = view_7 = permute_5 = None
    view_8: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(addmm_1, [32, 256, 144]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_38: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(view_2, view_8);  view_2 = view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_14: "f32[32, 256, 1]" = var_mean_1[0]
    getitem_15: "f32[32, 256, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_18: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_38, getitem_15);  getitem_15 = None
    add_39: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_1: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    mul_65: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_1);  sub_18 = rsqrt_1 = None
    mul_66: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_65, arg88_1);  mul_65 = arg88_1 = None
    add_40: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_66, arg89_1);  mul_66 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_9: "f32[8192, 144]" = torch.ops.aten.reshape.default(add_40, [8192, 144]);  add_40 = None
    permute_6: "f32[144, 288]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_2: "f32[8192, 288]" = torch.ops.aten.addmm.default(arg91_1, view_9, permute_6);  arg91_1 = view_9 = permute_6 = None
    view_10: "f32[32, 256, 288]" = torch.ops.aten.reshape.default(addmm_2, [32, 256, 288]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_12: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_10)
    mul_67: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_10, sigmoid_12);  view_10 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[8192, 288]" = torch.ops.aten.reshape.default(mul_67, [8192, 288]);  mul_67 = None
    permute_7: "f32[288, 144]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    addmm_3: "f32[8192, 144]" = torch.ops.aten.addmm.default(arg93_1, view_11, permute_7);  arg93_1 = view_11 = permute_7 = None
    view_12: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(addmm_3, [32, 256, 144]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_41: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_38, view_12);  add_38 = view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_16: "f32[32, 256, 1]" = var_mean_2[0]
    getitem_17: "f32[32, 256, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_19: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_41, getitem_17);  getitem_17 = None
    add_42: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_2: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    mul_68: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_2);  sub_19 = rsqrt_2 = None
    mul_69: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_68, arg94_1);  mul_68 = arg94_1 = None
    add_43: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_69, arg95_1);  mul_69 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_13: "f32[8192, 144]" = torch.ops.aten.reshape.default(add_43, [8192, 144]);  add_43 = None
    permute_8: "f32[144, 432]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_4: "f32[8192, 432]" = torch.ops.aten.addmm.default(arg97_1, view_13, permute_8);  arg97_1 = view_13 = permute_8 = None
    view_14: "f32[32, 256, 432]" = torch.ops.aten.reshape.default(addmm_4, [32, 256, 432]);  addmm_4 = None
    view_15: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.reshape.default(view_14, [32, 256, 3, 4, 36]);  view_14 = None
    permute_9: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_18: "f32[32, 4, 256, 36]" = unbind_1[0]
    getitem_19: "f32[32, 4, 256, 36]" = unbind_1[1]
    getitem_20: "f32[32, 4, 256, 36]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_18, getitem_19, getitem_20);  getitem_18 = getitem_19 = getitem_20 = None
    getitem_21: "f32[32, 4, 256, 36]" = _scaled_dot_product_flash_attention_1[0];  _scaled_dot_product_flash_attention_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_10: "f32[32, 256, 4, 36]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
    view_16: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(permute_10, [32, 256, 144]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_17: "f32[8192, 144]" = torch.ops.aten.reshape.default(view_16, [8192, 144]);  view_16 = None
    permute_11: "f32[144, 144]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_5: "f32[8192, 144]" = torch.ops.aten.addmm.default(arg99_1, view_17, permute_11);  arg99_1 = view_17 = permute_11 = None
    view_18: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(addmm_5, [32, 256, 144]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_44: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_41, view_18);  add_41 = view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
    getitem_30: "f32[32, 256, 1]" = var_mean_3[0]
    getitem_31: "f32[32, 256, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_20: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_44, getitem_31);  getitem_31 = None
    add_45: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_3: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    mul_70: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_3);  sub_20 = rsqrt_3 = None
    mul_71: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_70, arg100_1);  mul_70 = arg100_1 = None
    add_46: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_71, arg101_1);  mul_71 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_19: "f32[8192, 144]" = torch.ops.aten.reshape.default(add_46, [8192, 144]);  add_46 = None
    permute_12: "f32[144, 288]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_6: "f32[8192, 288]" = torch.ops.aten.addmm.default(arg103_1, view_19, permute_12);  arg103_1 = view_19 = permute_12 = None
    view_20: "f32[32, 256, 288]" = torch.ops.aten.reshape.default(addmm_6, [32, 256, 288]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_13: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_20)
    mul_72: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_20, sigmoid_13);  view_20 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_21: "f32[8192, 288]" = torch.ops.aten.reshape.default(mul_72, [8192, 288]);  mul_72 = None
    permute_13: "f32[288, 144]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_7: "f32[8192, 144]" = torch.ops.aten.addmm.default(arg105_1, view_21, permute_13);  arg105_1 = view_21 = permute_13 = None
    view_22: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(addmm_7, [32, 256, 144]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_47: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_44, view_22);  add_44 = view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_32: "f32[32, 256, 1]" = var_mean_4[0]
    getitem_33: "f32[32, 256, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_21: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(add_47, getitem_33);  add_47 = getitem_33 = None
    add_48: "f32[32, 256, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_4: "f32[32, 256, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    mul_73: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_4);  sub_21 = rsqrt_4 = None
    mul_74: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_73, arg106_1);  mul_73 = arg106_1 = None
    add_49: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_74, arg107_1);  mul_74 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    view_23: "f32[8, 4, 256, 144]" = torch.ops.aten.reshape.default(add_49, [8, 4, 256, -1]);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    permute_14: "f32[8, 144, 256, 4]" = torch.ops.aten.permute.default(view_23, [0, 3, 2, 1]);  view_23 = None
    clone_8: "f32[8, 144, 256, 4]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_24: "f32[18432, 16, 2, 2]" = torch.ops.aten.reshape.default(clone_8, [18432, 16, 2, 2]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    permute_15: "f32[18432, 2, 16, 2]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    clone_9: "f32[18432, 2, 16, 2]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_25: "f32[8, 144, 32, 32]" = torch.ops.aten.reshape.default(clone_9, [8, 144, 32, 32]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(view_25, arg108_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_25 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_136: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
    unsqueeze_137: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    sub_22: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_137);  convolution_18 = unsqueeze_137 = None
    add_50: "f32[96]" = torch.ops.aten.add.Tensor(arg250_1, 1e-05);  arg250_1 = None
    sqrt_17: "f32[96]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_17: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_75: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_138: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_139: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    mul_76: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_139);  sub_22 = unsqueeze_139 = None
    unsqueeze_140: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_141: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_77: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_141);  mul_76 = unsqueeze_141 = None
    unsqueeze_142: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_143: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_51: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_143);  mul_77 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_14: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_51)
    mul_78: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_51, sigmoid_14);  add_51 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat: "f32[8, 192, 32, 32]" = torch.ops.aten.cat.default([add_33, mul_78], 1);  add_33 = mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 96, 32, 32]" = torch.ops.aten.convolution.default(cat, arg109_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_144: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
    unsqueeze_145: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    sub_23: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_145);  convolution_19 = unsqueeze_145 = None
    add_52: "f32[96]" = torch.ops.aten.add.Tensor(arg252_1, 1e-05);  arg252_1 = None
    sqrt_18: "f32[96]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_18: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_79: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_146: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_79, -1);  mul_79 = None
    unsqueeze_147: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    mul_80: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_147);  sub_23 = unsqueeze_147 = None
    unsqueeze_148: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_149: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_81: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_149);  mul_80 = unsqueeze_149 = None
    unsqueeze_150: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_151: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_53: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_81, unsqueeze_151);  mul_81 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_15: "f32[8, 96, 32, 32]" = torch.ops.aten.sigmoid.default(add_53)
    mul_82: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_53, sigmoid_15);  add_53 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 384, 32, 32]" = torch.ops.aten.convolution.default(mul_82, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_82 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
    unsqueeze_153: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    sub_24: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_153);  convolution_20 = unsqueeze_153 = None
    add_54: "f32[384]" = torch.ops.aten.add.Tensor(arg254_1, 1e-05);  arg254_1 = None
    sqrt_19: "f32[384]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_19: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_83: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_154: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_83, -1);  mul_83 = None
    unsqueeze_155: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    mul_84: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_155);  sub_24 = unsqueeze_155 = None
    unsqueeze_156: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_157: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_85: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_157);  mul_84 = unsqueeze_157 = None
    unsqueeze_158: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_159: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_55: "f32[8, 384, 32, 32]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_159);  mul_85 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_16: "f32[8, 384, 32, 32]" = torch.ops.aten.sigmoid.default(add_55)
    mul_86: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_16);  add_55 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 384, 16, 16]" = torch.ops.aten.convolution.default(mul_86, arg111_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 384);  mul_86 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_160: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
    unsqueeze_161: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    sub_25: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_161);  convolution_21 = unsqueeze_161 = None
    add_56: "f32[384]" = torch.ops.aten.add.Tensor(arg256_1, 1e-05);  arg256_1 = None
    sqrt_20: "f32[384]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_20: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_87: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_162: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_163: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    mul_88: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_163);  sub_25 = unsqueeze_163 = None
    unsqueeze_164: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_165: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_89: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_165);  mul_88 = unsqueeze_165 = None
    unsqueeze_166: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_167: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_57: "f32[8, 384, 16, 16]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_167);  mul_89 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_17: "f32[8, 384, 16, 16]" = torch.ops.aten.sigmoid.default(add_57)
    mul_90: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(add_57, sigmoid_17);  add_57 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(mul_90, arg112_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_90 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_168: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
    unsqueeze_169: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    sub_26: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_169);  convolution_22 = unsqueeze_169 = None
    add_58: "f32[128]" = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
    sqrt_21: "f32[128]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_21: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_91: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_170: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_91, -1);  mul_91 = None
    unsqueeze_171: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    mul_92: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_171);  sub_26 = unsqueeze_171 = None
    unsqueeze_172: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_173: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_93: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_173);  mul_92 = unsqueeze_173 = None
    unsqueeze_174: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_175: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_59: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_175);  mul_93 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(add_59, arg113_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
    unsqueeze_177: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    sub_27: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_177);  convolution_23 = unsqueeze_177 = None
    add_60: "f32[128]" = torch.ops.aten.add.Tensor(arg260_1, 1e-05);  arg260_1 = None
    sqrt_22: "f32[128]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_22: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_94: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_178: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_94, -1);  mul_94 = None
    unsqueeze_179: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    mul_95: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_179);  sub_27 = unsqueeze_179 = None
    unsqueeze_180: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_181: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_96: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_95, unsqueeze_181);  mul_95 = unsqueeze_181 = None
    unsqueeze_182: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
    unsqueeze_183: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_61: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_96, unsqueeze_183);  mul_96 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_18: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_61)
    mul_97: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_18);  add_61 = sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_24: "f32[8, 192, 16, 16]" = torch.ops.aten.convolution.default(mul_97, arg114_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_97 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    view_26: "f32[12288, 2, 8, 2]" = torch.ops.aten.reshape.default(convolution_24, [12288, 2, 8, 2]);  convolution_24 = None
    permute_16: "f32[12288, 8, 2, 2]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    clone_10: "f32[12288, 8, 2, 2]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_27: "f32[8, 192, 64, 4]" = torch.ops.aten.reshape.default(clone_10, [8, 192, 64, 4]);  clone_10 = None
    permute_17: "f32[8, 4, 64, 192]" = torch.ops.aten.permute.default(view_27, [0, 3, 2, 1]);  view_27 = None
    clone_11: "f32[8, 4, 64, 192]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_28: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(clone_11, [32, 64, 192]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(view_28, [2], correction = 0, keepdim = True)
    getitem_34: "f32[32, 64, 1]" = var_mean_5[0]
    getitem_35: "f32[32, 64, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_28: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(view_28, getitem_35);  getitem_35 = None
    add_62: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_5: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    mul_98: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_5);  sub_28 = rsqrt_5 = None
    mul_99: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_98, arg115_1);  mul_98 = arg115_1 = None
    add_63: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_99, arg116_1);  mul_99 = arg116_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_29: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_63, [2048, 192]);  add_63 = None
    permute_18: "f32[192, 576]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_8: "f32[2048, 576]" = torch.ops.aten.addmm.default(arg118_1, view_29, permute_18);  arg118_1 = view_29 = permute_18 = None
    view_30: "f32[32, 64, 576]" = torch.ops.aten.reshape.default(addmm_8, [32, 64, 576]);  addmm_8 = None
    view_31: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.reshape.default(view_30, [32, 64, 3, 4, 48]);  view_30 = None
    permute_19: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_19);  permute_19 = None
    getitem_36: "f32[32, 4, 64, 48]" = unbind_2[0]
    getitem_37: "f32[32, 4, 64, 48]" = unbind_2[1]
    getitem_38: "f32[32, 4, 64, 48]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_36, getitem_37, getitem_38);  getitem_36 = getitem_37 = getitem_38 = None
    getitem_39: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_2[0];  _scaled_dot_product_flash_attention_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_20: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3]);  getitem_39 = None
    view_32: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(permute_20, [32, 64, 192]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_33: "f32[2048, 192]" = torch.ops.aten.reshape.default(view_32, [2048, 192]);  view_32 = None
    permute_21: "f32[192, 192]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_9: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg120_1, view_33, permute_21);  arg120_1 = view_33 = permute_21 = None
    view_34: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(addmm_9, [32, 64, 192]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_64: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(view_28, view_34);  view_28 = view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_48: "f32[32, 64, 1]" = var_mean_6[0]
    getitem_49: "f32[32, 64, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_29: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_64, getitem_49);  getitem_49 = None
    add_65: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_6: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    mul_100: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_6);  sub_29 = rsqrt_6 = None
    mul_101: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_100, arg121_1);  mul_100 = arg121_1 = None
    add_66: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_101, arg122_1);  mul_101 = arg122_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_35: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_66, [2048, 192]);  add_66 = None
    permute_22: "f32[192, 384]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    addmm_10: "f32[2048, 384]" = torch.ops.aten.addmm.default(arg124_1, view_35, permute_22);  arg124_1 = view_35 = permute_22 = None
    view_36: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(addmm_10, [32, 64, 384]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_19: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_36)
    mul_102: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_36, sigmoid_19);  view_36 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_37: "f32[2048, 384]" = torch.ops.aten.reshape.default(mul_102, [2048, 384]);  mul_102 = None
    permute_23: "f32[384, 192]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    addmm_11: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg126_1, view_37, permute_23);  arg126_1 = view_37 = permute_23 = None
    view_38: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(addmm_11, [32, 64, 192]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_67: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_64, view_38);  add_64 = view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_50: "f32[32, 64, 1]" = var_mean_7[0]
    getitem_51: "f32[32, 64, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_30: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_67, getitem_51);  getitem_51 = None
    add_68: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_7: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    mul_103: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_7);  sub_30 = rsqrt_7 = None
    mul_104: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_103, arg127_1);  mul_103 = arg127_1 = None
    add_69: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_104, arg128_1);  mul_104 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_39: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_69, [2048, 192]);  add_69 = None
    permute_24: "f32[192, 576]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    addmm_12: "f32[2048, 576]" = torch.ops.aten.addmm.default(arg130_1, view_39, permute_24);  arg130_1 = view_39 = permute_24 = None
    view_40: "f32[32, 64, 576]" = torch.ops.aten.reshape.default(addmm_12, [32, 64, 576]);  addmm_12 = None
    view_41: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.reshape.default(view_40, [32, 64, 3, 4, 48]);  view_40 = None
    permute_25: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_41, [2, 0, 3, 1, 4]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
    getitem_52: "f32[32, 4, 64, 48]" = unbind_3[0]
    getitem_53: "f32[32, 4, 64, 48]" = unbind_3[1]
    getitem_54: "f32[32, 4, 64, 48]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_52, getitem_53, getitem_54);  getitem_52 = getitem_53 = getitem_54 = None
    getitem_55: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_3[0];  _scaled_dot_product_flash_attention_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_26: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
    view_42: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(permute_26, [32, 64, 192]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_43: "f32[2048, 192]" = torch.ops.aten.reshape.default(view_42, [2048, 192]);  view_42 = None
    permute_27: "f32[192, 192]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_13: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg132_1, view_43, permute_27);  arg132_1 = view_43 = permute_27 = None
    view_44: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(addmm_13, [32, 64, 192]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_70: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_67, view_44);  add_67 = view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_8 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_64: "f32[32, 64, 1]" = var_mean_8[0]
    getitem_65: "f32[32, 64, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_31: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_70, getitem_65);  getitem_65 = None
    add_71: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_8: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    mul_105: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_8);  sub_31 = rsqrt_8 = None
    mul_106: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_105, arg133_1);  mul_105 = arg133_1 = None
    add_72: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_106, arg134_1);  mul_106 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_72, [2048, 192]);  add_72 = None
    permute_28: "f32[192, 384]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_14: "f32[2048, 384]" = torch.ops.aten.addmm.default(arg136_1, view_45, permute_28);  arg136_1 = view_45 = permute_28 = None
    view_46: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(addmm_14, [32, 64, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_20: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_46)
    mul_107: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_46, sigmoid_20);  view_46 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[2048, 384]" = torch.ops.aten.reshape.default(mul_107, [2048, 384]);  mul_107 = None
    permute_29: "f32[384, 192]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_15: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg138_1, view_47, permute_29);  arg138_1 = view_47 = permute_29 = None
    view_48: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(addmm_15, [32, 64, 192]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_73: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_70, view_48);  add_70 = view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_9 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_66: "f32[32, 64, 1]" = var_mean_9[0]
    getitem_67: "f32[32, 64, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_32: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_73, getitem_67);  getitem_67 = None
    add_74: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_9: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    mul_108: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_9);  sub_32 = rsqrt_9 = None
    mul_109: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_108, arg139_1);  mul_108 = arg139_1 = None
    add_75: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_109, arg140_1);  mul_109 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_49: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_75, [2048, 192]);  add_75 = None
    permute_30: "f32[192, 576]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    addmm_16: "f32[2048, 576]" = torch.ops.aten.addmm.default(arg142_1, view_49, permute_30);  arg142_1 = view_49 = permute_30 = None
    view_50: "f32[32, 64, 576]" = torch.ops.aten.reshape.default(addmm_16, [32, 64, 576]);  addmm_16 = None
    view_51: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.reshape.default(view_50, [32, 64, 3, 4, 48]);  view_50 = None
    permute_31: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
    getitem_68: "f32[32, 4, 64, 48]" = unbind_4[0]
    getitem_69: "f32[32, 4, 64, 48]" = unbind_4[1]
    getitem_70: "f32[32, 4, 64, 48]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_68, getitem_69, getitem_70);  getitem_68 = getitem_69 = getitem_70 = None
    getitem_71: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_4[0];  _scaled_dot_product_flash_attention_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_32: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
    view_52: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(permute_32, [32, 64, 192]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_53: "f32[2048, 192]" = torch.ops.aten.reshape.default(view_52, [2048, 192]);  view_52 = None
    permute_33: "f32[192, 192]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_17: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg144_1, view_53, permute_33);  arg144_1 = view_53 = permute_33 = None
    view_54: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(addmm_17, [32, 64, 192]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_76: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_73, view_54);  add_73 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_10 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
    getitem_80: "f32[32, 64, 1]" = var_mean_10[0]
    getitem_81: "f32[32, 64, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_33: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_76, getitem_81);  getitem_81 = None
    add_77: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_10: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    mul_110: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_10);  sub_33 = rsqrt_10 = None
    mul_111: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_110, arg145_1);  mul_110 = arg145_1 = None
    add_78: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_111, arg146_1);  mul_111 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_55: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_78, [2048, 192]);  add_78 = None
    permute_34: "f32[192, 384]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_18: "f32[2048, 384]" = torch.ops.aten.addmm.default(arg148_1, view_55, permute_34);  arg148_1 = view_55 = permute_34 = None
    view_56: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(addmm_18, [32, 64, 384]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_21: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_56)
    mul_112: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_56, sigmoid_21);  view_56 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_57: "f32[2048, 384]" = torch.ops.aten.reshape.default(mul_112, [2048, 384]);  mul_112 = None
    permute_35: "f32[384, 192]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_19: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg150_1, view_57, permute_35);  arg150_1 = view_57 = permute_35 = None
    view_58: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(addmm_19, [32, 64, 192]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_79: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_76, view_58);  add_76 = view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_11 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_82: "f32[32, 64, 1]" = var_mean_11[0]
    getitem_83: "f32[32, 64, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_34: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_79, getitem_83);  getitem_83 = None
    add_80: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_11: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    mul_113: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_11);  sub_34 = rsqrt_11 = None
    mul_114: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_113, arg151_1);  mul_113 = arg151_1 = None
    add_81: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_114, arg152_1);  mul_114 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_59: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_81, [2048, 192]);  add_81 = None
    permute_36: "f32[192, 576]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_20: "f32[2048, 576]" = torch.ops.aten.addmm.default(arg154_1, view_59, permute_36);  arg154_1 = view_59 = permute_36 = None
    view_60: "f32[32, 64, 576]" = torch.ops.aten.reshape.default(addmm_20, [32, 64, 576]);  addmm_20 = None
    view_61: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.reshape.default(view_60, [32, 64, 3, 4, 48]);  view_60 = None
    permute_37: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.permute.default(view_61, [2, 0, 3, 1, 4]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
    getitem_84: "f32[32, 4, 64, 48]" = unbind_5[0]
    getitem_85: "f32[32, 4, 64, 48]" = unbind_5[1]
    getitem_86: "f32[32, 4, 64, 48]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_84, getitem_85, getitem_86);  getitem_84 = getitem_85 = getitem_86 = None
    getitem_87: "f32[32, 4, 64, 48]" = _scaled_dot_product_flash_attention_5[0];  _scaled_dot_product_flash_attention_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_38: "f32[32, 64, 4, 48]" = torch.ops.aten.permute.default(getitem_87, [0, 2, 1, 3]);  getitem_87 = None
    view_62: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(permute_38, [32, 64, 192]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_63: "f32[2048, 192]" = torch.ops.aten.reshape.default(view_62, [2048, 192]);  view_62 = None
    permute_39: "f32[192, 192]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    addmm_21: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg156_1, view_63, permute_39);  arg156_1 = view_63 = permute_39 = None
    view_64: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(addmm_21, [32, 64, 192]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_82: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_79, view_64);  add_79 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_12 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
    getitem_96: "f32[32, 64, 1]" = var_mean_12[0]
    getitem_97: "f32[32, 64, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_35: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_82, getitem_97);  getitem_97 = None
    add_83: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_12: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    mul_115: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_12);  sub_35 = rsqrt_12 = None
    mul_116: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_115, arg157_1);  mul_115 = arg157_1 = None
    add_84: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_116, arg158_1);  mul_116 = arg158_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_65: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_84, [2048, 192]);  add_84 = None
    permute_40: "f32[192, 384]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_22: "f32[2048, 384]" = torch.ops.aten.addmm.default(arg160_1, view_65, permute_40);  arg160_1 = view_65 = permute_40 = None
    view_66: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(addmm_22, [32, 64, 384]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_22: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_66)
    mul_117: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_66, sigmoid_22);  view_66 = sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_67: "f32[2048, 384]" = torch.ops.aten.reshape.default(mul_117, [2048, 384]);  mul_117 = None
    permute_41: "f32[384, 192]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    addmm_23: "f32[2048, 192]" = torch.ops.aten.addmm.default(arg162_1, view_67, permute_41);  arg162_1 = view_67 = permute_41 = None
    view_68: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(addmm_23, [32, 64, 192]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_85: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_82, view_68);  add_82 = view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_98: "f32[32, 64, 1]" = var_mean_13[0]
    getitem_99: "f32[32, 64, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_36: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(add_85, getitem_99);  add_85 = getitem_99 = None
    add_86: "f32[32, 64, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_13: "f32[32, 64, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    mul_118: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_13);  sub_36 = rsqrt_13 = None
    mul_119: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_118, arg163_1);  mul_118 = arg163_1 = None
    add_87: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_119, arg164_1);  mul_119 = arg164_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    view_69: "f32[8, 4, 64, 192]" = torch.ops.aten.reshape.default(add_87, [8, 4, 64, -1]);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    permute_42: "f32[8, 192, 64, 4]" = torch.ops.aten.permute.default(view_69, [0, 3, 2, 1]);  view_69 = None
    clone_24: "f32[8, 192, 64, 4]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    view_70: "f32[12288, 8, 2, 2]" = torch.ops.aten.reshape.default(clone_24, [12288, 8, 2, 2]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    permute_43: "f32[12288, 2, 8, 2]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    clone_25: "f32[12288, 2, 8, 2]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_71: "f32[8, 192, 16, 16]" = torch.ops.aten.reshape.default(clone_25, [8, 192, 16, 16]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(view_71, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_71 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_184: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
    unsqueeze_185: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    sub_37: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_185);  convolution_25 = unsqueeze_185 = None
    add_88: "f32[128]" = torch.ops.aten.add.Tensor(arg262_1, 1e-05);  arg262_1 = None
    sqrt_23: "f32[128]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
    reciprocal_23: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_120: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_186: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_187: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    mul_121: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_187);  sub_37 = unsqueeze_187 = None
    unsqueeze_188: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_189: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_122: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_189);  mul_121 = unsqueeze_189 = None
    unsqueeze_190: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_191: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_89: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_191);  mul_122 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_23: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_89)
    mul_123: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_89, sigmoid_23);  add_89 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat_1: "f32[8, 256, 16, 16]" = torch.ops.aten.cat.default([add_59, mul_123], 1);  add_59 = mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 128, 16, 16]" = torch.ops.aten.convolution.default(cat_1, arg166_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_1 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_192: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
    unsqueeze_193: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    sub_38: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_193);  convolution_26 = unsqueeze_193 = None
    add_90: "f32[128]" = torch.ops.aten.add.Tensor(arg264_1, 1e-05);  arg264_1 = None
    sqrt_24: "f32[128]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_24: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_124: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_194: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_124, -1);  mul_124 = None
    unsqueeze_195: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    mul_125: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_195);  sub_38 = unsqueeze_195 = None
    unsqueeze_196: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
    unsqueeze_197: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_126: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_125, unsqueeze_197);  mul_125 = unsqueeze_197 = None
    unsqueeze_198: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_199: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_91: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_199);  mul_126 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_24: "f32[8, 128, 16, 16]" = torch.ops.aten.sigmoid.default(add_91)
    mul_127: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_91, sigmoid_24);  add_91 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_127, arg167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_127 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
    unsqueeze_201: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    sub_39: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_201);  convolution_27 = unsqueeze_201 = None
    add_92: "f32[512]" = torch.ops.aten.add.Tensor(arg266_1, 1e-05);  arg266_1 = None
    sqrt_25: "f32[512]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_25: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_128: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_202: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_128, -1);  mul_128 = None
    unsqueeze_203: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    mul_129: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_203);  sub_39 = unsqueeze_203 = None
    unsqueeze_204: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_205: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_130: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_205);  mul_129 = unsqueeze_205 = None
    unsqueeze_206: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
    unsqueeze_207: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_93: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_130, unsqueeze_207);  mul_130 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_25: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_93)
    mul_131: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_93, sigmoid_25);  add_93 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_131, arg168_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 512);  mul_131 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_208: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
    unsqueeze_209: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    sub_40: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_209);  convolution_28 = unsqueeze_209 = None
    add_94: "f32[512]" = torch.ops.aten.add.Tensor(arg268_1, 1e-05);  arg268_1 = None
    sqrt_26: "f32[512]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_26: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_132: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_210: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_211: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    mul_133: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_211);  sub_40 = unsqueeze_211 = None
    unsqueeze_212: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
    unsqueeze_213: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_134: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_213);  mul_133 = unsqueeze_213 = None
    unsqueeze_214: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_215: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_95: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_215);  mul_134 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_26: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_95)
    mul_135: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_95, sigmoid_26);  add_95 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(mul_135, arg169_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_135 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_216: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
    unsqueeze_217: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    sub_41: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_217);  convolution_29 = unsqueeze_217 = None
    add_96: "f32[160]" = torch.ops.aten.add.Tensor(arg270_1, 1e-05);  arg270_1 = None
    sqrt_27: "f32[160]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_27: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_136: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_218: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_136, -1);  mul_136 = None
    unsqueeze_219: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    mul_137: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_219);  sub_41 = unsqueeze_219 = None
    unsqueeze_220: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
    unsqueeze_221: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_138: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_137, unsqueeze_221);  mul_137 = unsqueeze_221 = None
    unsqueeze_222: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_223: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_97: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_138, unsqueeze_223);  mul_138 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(add_97, arg170_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg170_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
    unsqueeze_225: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    sub_42: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_225);  convolution_30 = unsqueeze_225 = None
    add_98: "f32[160]" = torch.ops.aten.add.Tensor(arg272_1, 1e-05);  arg272_1 = None
    sqrt_28: "f32[160]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
    reciprocal_28: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_139: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_226: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_139, -1);  mul_139 = None
    unsqueeze_227: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    mul_140: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_227);  sub_42 = unsqueeze_227 = None
    unsqueeze_228: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_229: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_141: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_229);  mul_140 = unsqueeze_229 = None
    unsqueeze_230: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
    unsqueeze_231: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_99: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_141, unsqueeze_231);  mul_141 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_27: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_99)
    mul_142: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_99, sigmoid_27);  add_99 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_31: "f32[8, 240, 8, 8]" = torch.ops.aten.convolution.default(mul_142, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_142 = arg171_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    view_72: "f32[7680, 2, 4, 2]" = torch.ops.aten.reshape.default(convolution_31, [7680, 2, 4, 2]);  convolution_31 = None
    permute_44: "f32[7680, 4, 2, 2]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    clone_26: "f32[7680, 4, 2, 2]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    view_73: "f32[8, 240, 16, 4]" = torch.ops.aten.reshape.default(clone_26, [8, 240, 16, 4]);  clone_26 = None
    permute_45: "f32[8, 4, 16, 240]" = torch.ops.aten.permute.default(view_73, [0, 3, 2, 1]);  view_73 = None
    clone_27: "f32[8, 4, 16, 240]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_74: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(clone_27, [32, 16, 240]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(view_74, [2], correction = 0, keepdim = True)
    getitem_100: "f32[32, 16, 1]" = var_mean_14[0]
    getitem_101: "f32[32, 16, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_43: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(view_74, getitem_101);  getitem_101 = None
    add_100: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_14: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    mul_143: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_14);  sub_43 = rsqrt_14 = None
    mul_144: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_143, arg172_1);  mul_143 = arg172_1 = None
    add_101: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_144, arg173_1);  mul_144 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_75: "f32[512, 240]" = torch.ops.aten.reshape.default(add_101, [512, 240]);  add_101 = None
    permute_46: "f32[240, 720]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_24: "f32[512, 720]" = torch.ops.aten.addmm.default(arg175_1, view_75, permute_46);  arg175_1 = view_75 = permute_46 = None
    view_76: "f32[32, 16, 720]" = torch.ops.aten.reshape.default(addmm_24, [32, 16, 720]);  addmm_24 = None
    view_77: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.reshape.default(view_76, [32, 16, 3, 4, 60]);  view_76 = None
    permute_47: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_77, [2, 0, 3, 1, 4]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
    getitem_102: "f32[32, 4, 16, 60]" = unbind_6[0]
    getitem_103: "f32[32, 4, 16, 60]" = unbind_6[1]
    getitem_104: "f32[32, 4, 16, 60]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_102, getitem_103, getitem_104);  getitem_102 = getitem_103 = getitem_104 = None
    getitem_105: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_6[0];  _scaled_dot_product_flash_attention_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_48: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_105, [0, 2, 1, 3]);  getitem_105 = None
    view_78: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(permute_48, [32, 16, 240]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_79: "f32[512, 240]" = torch.ops.aten.reshape.default(view_78, [512, 240]);  view_78 = None
    permute_49: "f32[240, 240]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_25: "f32[512, 240]" = torch.ops.aten.addmm.default(arg177_1, view_79, permute_49);  arg177_1 = view_79 = permute_49 = None
    view_80: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(addmm_25, [32, 16, 240]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_102: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(view_74, view_80);  view_74 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
    getitem_114: "f32[32, 16, 1]" = var_mean_15[0]
    getitem_115: "f32[32, 16, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_44: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_102, getitem_115);  getitem_115 = None
    add_103: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_15: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    mul_145: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_15);  sub_44 = rsqrt_15 = None
    mul_146: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_145, arg178_1);  mul_145 = arg178_1 = None
    add_104: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_146, arg179_1);  mul_146 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_81: "f32[512, 240]" = torch.ops.aten.reshape.default(add_104, [512, 240]);  add_104 = None
    permute_50: "f32[240, 480]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_26: "f32[512, 480]" = torch.ops.aten.addmm.default(arg181_1, view_81, permute_50);  arg181_1 = view_81 = permute_50 = None
    view_82: "f32[32, 16, 480]" = torch.ops.aten.reshape.default(addmm_26, [32, 16, 480]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_28: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_82)
    mul_147: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_82, sigmoid_28);  view_82 = sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[512, 480]" = torch.ops.aten.reshape.default(mul_147, [512, 480]);  mul_147 = None
    permute_51: "f32[480, 240]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_27: "f32[512, 240]" = torch.ops.aten.addmm.default(arg183_1, view_83, permute_51);  arg183_1 = view_83 = permute_51 = None
    view_84: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(addmm_27, [32, 16, 240]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_105: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_102, view_84);  add_102 = view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_116: "f32[32, 16, 1]" = var_mean_16[0]
    getitem_117: "f32[32, 16, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_45: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_105, getitem_117);  getitem_117 = None
    add_106: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_16: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    mul_148: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_16);  sub_45 = rsqrt_16 = None
    mul_149: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_148, arg184_1);  mul_148 = arg184_1 = None
    add_107: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_149, arg185_1);  mul_149 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_85: "f32[512, 240]" = torch.ops.aten.reshape.default(add_107, [512, 240]);  add_107 = None
    permute_52: "f32[240, 720]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_28: "f32[512, 720]" = torch.ops.aten.addmm.default(arg187_1, view_85, permute_52);  arg187_1 = view_85 = permute_52 = None
    view_86: "f32[32, 16, 720]" = torch.ops.aten.reshape.default(addmm_28, [32, 16, 720]);  addmm_28 = None
    view_87: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.reshape.default(view_86, [32, 16, 3, 4, 60]);  view_86 = None
    permute_53: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
    getitem_118: "f32[32, 4, 16, 60]" = unbind_7[0]
    getitem_119: "f32[32, 4, 16, 60]" = unbind_7[1]
    getitem_120: "f32[32, 4, 16, 60]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_118, getitem_119, getitem_120);  getitem_118 = getitem_119 = getitem_120 = None
    getitem_121: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_7[0];  _scaled_dot_product_flash_attention_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_54: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3]);  getitem_121 = None
    view_88: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(permute_54, [32, 16, 240]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_89: "f32[512, 240]" = torch.ops.aten.reshape.default(view_88, [512, 240]);  view_88 = None
    permute_55: "f32[240, 240]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    addmm_29: "f32[512, 240]" = torch.ops.aten.addmm.default(arg189_1, view_89, permute_55);  arg189_1 = view_89 = permute_55 = None
    view_90: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(addmm_29, [32, 16, 240]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_108: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_105, view_90);  add_105 = view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
    getitem_130: "f32[32, 16, 1]" = var_mean_17[0]
    getitem_131: "f32[32, 16, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_46: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_108, getitem_131);  getitem_131 = None
    add_109: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_17: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    mul_150: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_17);  sub_46 = rsqrt_17 = None
    mul_151: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_150, arg190_1);  mul_150 = arg190_1 = None
    add_110: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_151, arg191_1);  mul_151 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_91: "f32[512, 240]" = torch.ops.aten.reshape.default(add_110, [512, 240]);  add_110 = None
    permute_56: "f32[240, 480]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_30: "f32[512, 480]" = torch.ops.aten.addmm.default(arg193_1, view_91, permute_56);  arg193_1 = view_91 = permute_56 = None
    view_92: "f32[32, 16, 480]" = torch.ops.aten.reshape.default(addmm_30, [32, 16, 480]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_29: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_92)
    mul_152: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_92, sigmoid_29);  view_92 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_93: "f32[512, 480]" = torch.ops.aten.reshape.default(mul_152, [512, 480]);  mul_152 = None
    permute_57: "f32[480, 240]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    addmm_31: "f32[512, 240]" = torch.ops.aten.addmm.default(arg195_1, view_93, permute_57);  arg195_1 = view_93 = permute_57 = None
    view_94: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(addmm_31, [32, 16, 240]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_111: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_108, view_94);  add_108 = view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_111, [2], correction = 0, keepdim = True)
    getitem_132: "f32[32, 16, 1]" = var_mean_18[0]
    getitem_133: "f32[32, 16, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_47: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_111, getitem_133);  getitem_133 = None
    add_112: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
    rsqrt_18: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    mul_153: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_18);  sub_47 = rsqrt_18 = None
    mul_154: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_153, arg196_1);  mul_153 = arg196_1 = None
    add_113: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_154, arg197_1);  mul_154 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_95: "f32[512, 240]" = torch.ops.aten.reshape.default(add_113, [512, 240]);  add_113 = None
    permute_58: "f32[240, 720]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    addmm_32: "f32[512, 720]" = torch.ops.aten.addmm.default(arg199_1, view_95, permute_58);  arg199_1 = view_95 = permute_58 = None
    view_96: "f32[32, 16, 720]" = torch.ops.aten.reshape.default(addmm_32, [32, 16, 720]);  addmm_32 = None
    view_97: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.reshape.default(view_96, [32, 16, 3, 4, 60]);  view_96 = None
    permute_59: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.permute.default(view_97, [2, 0, 3, 1, 4]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_59);  permute_59 = None
    getitem_134: "f32[32, 4, 16, 60]" = unbind_8[0]
    getitem_135: "f32[32, 4, 16, 60]" = unbind_8[1]
    getitem_136: "f32[32, 4, 16, 60]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_134, getitem_135, getitem_136);  getitem_134 = getitem_135 = getitem_136 = None
    getitem_137: "f32[32, 4, 16, 60]" = _scaled_dot_product_flash_attention_8[0];  _scaled_dot_product_flash_attention_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_60: "f32[32, 16, 4, 60]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None
    view_98: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(permute_60, [32, 16, 240]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_99: "f32[512, 240]" = torch.ops.aten.reshape.default(view_98, [512, 240]);  view_98 = None
    permute_61: "f32[240, 240]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
    addmm_33: "f32[512, 240]" = torch.ops.aten.addmm.default(arg201_1, view_99, permute_61);  arg201_1 = view_99 = permute_61 = None
    view_100: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(addmm_33, [32, 16, 240]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_114: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_111, view_100);  add_111 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_146: "f32[32, 16, 1]" = var_mean_19[0]
    getitem_147: "f32[32, 16, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_48: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_114, getitem_147);  getitem_147 = None
    add_115: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
    rsqrt_19: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    mul_155: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_19);  sub_48 = rsqrt_19 = None
    mul_156: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_155, arg202_1);  mul_155 = arg202_1 = None
    add_116: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_156, arg203_1);  mul_156 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[512, 240]" = torch.ops.aten.reshape.default(add_116, [512, 240]);  add_116 = None
    permute_62: "f32[240, 480]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    addmm_34: "f32[512, 480]" = torch.ops.aten.addmm.default(arg205_1, view_101, permute_62);  arg205_1 = view_101 = permute_62 = None
    view_102: "f32[32, 16, 480]" = torch.ops.aten.reshape.default(addmm_34, [32, 16, 480]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_30: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_102)
    mul_157: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_102, sigmoid_30);  view_102 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[512, 480]" = torch.ops.aten.reshape.default(mul_157, [512, 480]);  mul_157 = None
    permute_63: "f32[480, 240]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    addmm_35: "f32[512, 240]" = torch.ops.aten.addmm.default(arg207_1, view_103, permute_63);  arg207_1 = view_103 = permute_63 = None
    view_104: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(addmm_35, [32, 16, 240]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_117: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_114, view_104);  add_114 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_148: "f32[32, 16, 1]" = var_mean_20[0]
    getitem_149: "f32[32, 16, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_49: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(add_117, getitem_149);  add_117 = getitem_149 = None
    add_118: "f32[32, 16, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
    rsqrt_20: "f32[32, 16, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    mul_158: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_20);  sub_49 = rsqrt_20 = None
    mul_159: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_158, arg208_1);  mul_158 = arg208_1 = None
    add_119: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_159, arg209_1);  mul_159 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    view_105: "f32[8, 4, 16, 240]" = torch.ops.aten.reshape.default(add_119, [8, 4, 16, -1]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    permute_64: "f32[8, 240, 16, 4]" = torch.ops.aten.permute.default(view_105, [0, 3, 2, 1]);  view_105 = None
    clone_37: "f32[8, 240, 16, 4]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    view_106: "f32[7680, 4, 2, 2]" = torch.ops.aten.reshape.default(clone_37, [7680, 4, 2, 2]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    permute_65: "f32[7680, 2, 4, 2]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    clone_38: "f32[7680, 2, 4, 2]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
    view_107: "f32[8, 240, 8, 8]" = torch.ops.aten.reshape.default(clone_38, [8, 240, 8, 8]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(view_107, arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_107 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg273_1, -1);  arg273_1 = None
    unsqueeze_233: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    sub_50: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_233);  convolution_32 = unsqueeze_233 = None
    add_120: "f32[160]" = torch.ops.aten.add.Tensor(arg274_1, 1e-05);  arg274_1 = None
    sqrt_29: "f32[160]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_29: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_160: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_234: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
    unsqueeze_235: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    mul_161: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_235);  sub_50 = unsqueeze_235 = None
    unsqueeze_236: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_237: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_162: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_237);  mul_161 = unsqueeze_237 = None
    unsqueeze_238: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_239: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_121: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_239);  mul_162 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_31: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_121)
    mul_163: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_121, sigmoid_31);  add_121 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    cat_2: "f32[8, 320, 8, 8]" = torch.ops.aten.cat.default([add_97, mul_163], 1);  add_97 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 160, 8, 8]" = torch.ops.aten.convolution.default(cat_2, arg211_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_2 = arg211_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_240: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
    unsqueeze_241: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    sub_51: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_241);  convolution_33 = unsqueeze_241 = None
    add_122: "f32[160]" = torch.ops.aten.add.Tensor(arg276_1, 1e-05);  arg276_1 = None
    sqrt_30: "f32[160]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
    reciprocal_30: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_164: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_242: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_164, -1);  mul_164 = None
    unsqueeze_243: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    mul_165: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_243);  sub_51 = unsqueeze_243 = None
    unsqueeze_244: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
    unsqueeze_245: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_166: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_165, unsqueeze_245);  mul_165 = unsqueeze_245 = None
    unsqueeze_246: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_247: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_123: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_247);  mul_166 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_32: "f32[8, 160, 8, 8]" = torch.ops.aten.sigmoid.default(add_123)
    mul_167: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_123, sigmoid_32);  add_123 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(mul_167, arg212_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_167 = arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
    unsqueeze_249: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    sub_52: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_249);  convolution_34 = unsqueeze_249 = None
    add_124: "f32[640]" = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
    sqrt_31: "f32[640]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_31: "f32[640]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_168: "f32[640]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_250: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
    unsqueeze_251: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    mul_169: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_251);  sub_52 = unsqueeze_251 = None
    unsqueeze_252: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_253: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_170: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_253);  mul_169 = unsqueeze_253 = None
    unsqueeze_254: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
    unsqueeze_255: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_125: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_255);  mul_170 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[8, 640, 8, 8]" = torch.ops.aten.sigmoid.default(add_125)
    mul_171: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(add_125, sigmoid_33);  add_125 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 640, 1, 1]" = torch.ops.aten.mean.dim(mul_171, [-1, -2], True);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_108: "f32[8, 640]" = torch.ops.aten.reshape.default(mean, [8, 640]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute_66: "f32[640, 1000]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
    addmm_36: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg214_1, view_108, permute_66);  arg214_1 = view_108 = permute_66 = None
    return (addmm_36,)
    