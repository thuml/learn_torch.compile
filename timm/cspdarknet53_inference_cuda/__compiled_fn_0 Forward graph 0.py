from __future__ import annotations



def forward(self, arg0_1: "f32[32]", arg1_1: "f32[32]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[32]", arg7_1: "f32[32]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[128]", arg17_1: "f32[128]", arg18_1: "f32[64]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "f32[64]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[64]", arg27_1: "f32[64]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[256]", arg31_1: "f32[256]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[128]", arg35_1: "f32[128]", arg36_1: "f32[128]", arg37_1: "f32[128]", arg38_1: "f32[128]", arg39_1: "f32[128]", arg40_1: "f32[128]", arg41_1: "f32[128]", arg42_1: "f32[128]", arg43_1: "f32[128]", arg44_1: "f32[128]", arg45_1: "f32[128]", arg46_1: "f32[128]", arg47_1: "f32[128]", arg48_1: "f32[128]", arg49_1: "f32[128]", arg50_1: "f32[128]", arg51_1: "f32[128]", arg52_1: "f32[128]", arg53_1: "f32[128]", arg54_1: "f32[128]", arg55_1: "f32[128]", arg56_1: "f32[128]", arg57_1: "f32[128]", arg58_1: "f32[128]", arg59_1: "f32[128]", arg60_1: "f32[128]", arg61_1: "f32[128]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128]", arg65_1: "f32[128]", arg66_1: "f32[128]", arg67_1: "f32[128]", arg68_1: "f32[256]", arg69_1: "f32[256]", arg70_1: "f32[512]", arg71_1: "f32[512]", arg72_1: "f32[512]", arg73_1: "f32[512]", arg74_1: "f32[256]", arg75_1: "f32[256]", arg76_1: "f32[256]", arg77_1: "f32[256]", arg78_1: "f32[256]", arg79_1: "f32[256]", arg80_1: "f32[256]", arg81_1: "f32[256]", arg82_1: "f32[256]", arg83_1: "f32[256]", arg84_1: "f32[256]", arg85_1: "f32[256]", arg86_1: "f32[256]", arg87_1: "f32[256]", arg88_1: "f32[256]", arg89_1: "f32[256]", arg90_1: "f32[256]", arg91_1: "f32[256]", arg92_1: "f32[256]", arg93_1: "f32[256]", arg94_1: "f32[256]", arg95_1: "f32[256]", arg96_1: "f32[256]", arg97_1: "f32[256]", arg98_1: "f32[256]", arg99_1: "f32[256]", arg100_1: "f32[256]", arg101_1: "f32[256]", arg102_1: "f32[256]", arg103_1: "f32[256]", arg104_1: "f32[256]", arg105_1: "f32[256]", arg106_1: "f32[256]", arg107_1: "f32[256]", arg108_1: "f32[512]", arg109_1: "f32[512]", arg110_1: "f32[1024]", arg111_1: "f32[1024]", arg112_1: "f32[1024]", arg113_1: "f32[1024]", arg114_1: "f32[512]", arg115_1: "f32[512]", arg116_1: "f32[512]", arg117_1: "f32[512]", arg118_1: "f32[512]", arg119_1: "f32[512]", arg120_1: "f32[512]", arg121_1: "f32[512]", arg122_1: "f32[512]", arg123_1: "f32[512]", arg124_1: "f32[512]", arg125_1: "f32[512]", arg126_1: "f32[512]", arg127_1: "f32[512]", arg128_1: "f32[512]", arg129_1: "f32[512]", arg130_1: "f32[512]", arg131_1: "f32[512]", arg132_1: "f32[1024]", arg133_1: "f32[1024]", arg134_1: "f32[32, 3, 3, 3]", arg135_1: "f32[64, 32, 3, 3]", arg136_1: "f32[128, 64, 1, 1]", arg137_1: "f32[32, 64, 1, 1]", arg138_1: "f32[64, 32, 3, 3]", arg139_1: "f32[64, 64, 1, 1]", arg140_1: "f32[64, 128, 1, 1]", arg141_1: "f32[128, 64, 3, 3]", arg142_1: "f32[128, 128, 1, 1]", arg143_1: "f32[64, 64, 1, 1]", arg144_1: "f32[64, 64, 3, 3]", arg145_1: "f32[64, 64, 1, 1]", arg146_1: "f32[64, 64, 3, 3]", arg147_1: "f32[64, 64, 1, 1]", arg148_1: "f32[128, 128, 1, 1]", arg149_1: "f32[256, 128, 3, 3]", arg150_1: "f32[256, 256, 1, 1]", arg151_1: "f32[128, 128, 1, 1]", arg152_1: "f32[128, 128, 3, 3]", arg153_1: "f32[128, 128, 1, 1]", arg154_1: "f32[128, 128, 3, 3]", arg155_1: "f32[128, 128, 1, 1]", arg156_1: "f32[128, 128, 3, 3]", arg157_1: "f32[128, 128, 1, 1]", arg158_1: "f32[128, 128, 3, 3]", arg159_1: "f32[128, 128, 1, 1]", arg160_1: "f32[128, 128, 3, 3]", arg161_1: "f32[128, 128, 1, 1]", arg162_1: "f32[128, 128, 3, 3]", arg163_1: "f32[128, 128, 1, 1]", arg164_1: "f32[128, 128, 3, 3]", arg165_1: "f32[128, 128, 1, 1]", arg166_1: "f32[128, 128, 3, 3]", arg167_1: "f32[128, 128, 1, 1]", arg168_1: "f32[256, 256, 1, 1]", arg169_1: "f32[512, 256, 3, 3]", arg170_1: "f32[512, 512, 1, 1]", arg171_1: "f32[256, 256, 1, 1]", arg172_1: "f32[256, 256, 3, 3]", arg173_1: "f32[256, 256, 1, 1]", arg174_1: "f32[256, 256, 3, 3]", arg175_1: "f32[256, 256, 1, 1]", arg176_1: "f32[256, 256, 3, 3]", arg177_1: "f32[256, 256, 1, 1]", arg178_1: "f32[256, 256, 3, 3]", arg179_1: "f32[256, 256, 1, 1]", arg180_1: "f32[256, 256, 3, 3]", arg181_1: "f32[256, 256, 1, 1]", arg182_1: "f32[256, 256, 3, 3]", arg183_1: "f32[256, 256, 1, 1]", arg184_1: "f32[256, 256, 3, 3]", arg185_1: "f32[256, 256, 1, 1]", arg186_1: "f32[256, 256, 3, 3]", arg187_1: "f32[256, 256, 1, 1]", arg188_1: "f32[512, 512, 1, 1]", arg189_1: "f32[1024, 512, 3, 3]", arg190_1: "f32[1024, 1024, 1, 1]", arg191_1: "f32[512, 512, 1, 1]", arg192_1: "f32[512, 512, 3, 3]", arg193_1: "f32[512, 512, 1, 1]", arg194_1: "f32[512, 512, 3, 3]", arg195_1: "f32[512, 512, 1, 1]", arg196_1: "f32[512, 512, 3, 3]", arg197_1: "f32[512, 512, 1, 1]", arg198_1: "f32[512, 512, 3, 3]", arg199_1: "f32[512, 512, 1, 1]", arg200_1: "f32[1024, 1024, 1, 1]", arg201_1: "f32[1000, 1024]", arg202_1: "f32[1000]", arg203_1: "f32[32]", arg204_1: "f32[32]", arg205_1: "f32[64]", arg206_1: "f32[64]", arg207_1: "f32[128]", arg208_1: "f32[128]", arg209_1: "f32[32]", arg210_1: "f32[32]", arg211_1: "f32[64]", arg212_1: "f32[64]", arg213_1: "f32[64]", arg214_1: "f32[64]", arg215_1: "f32[64]", arg216_1: "f32[64]", arg217_1: "f32[128]", arg218_1: "f32[128]", arg219_1: "f32[128]", arg220_1: "f32[128]", arg221_1: "f32[64]", arg222_1: "f32[64]", arg223_1: "f32[64]", arg224_1: "f32[64]", arg225_1: "f32[64]", arg226_1: "f32[64]", arg227_1: "f32[64]", arg228_1: "f32[64]", arg229_1: "f32[64]", arg230_1: "f32[64]", arg231_1: "f32[128]", arg232_1: "f32[128]", arg233_1: "f32[256]", arg234_1: "f32[256]", arg235_1: "f32[256]", arg236_1: "f32[256]", arg237_1: "f32[128]", arg238_1: "f32[128]", arg239_1: "f32[128]", arg240_1: "f32[128]", arg241_1: "f32[128]", arg242_1: "f32[128]", arg243_1: "f32[128]", arg244_1: "f32[128]", arg245_1: "f32[128]", arg246_1: "f32[128]", arg247_1: "f32[128]", arg248_1: "f32[128]", arg249_1: "f32[128]", arg250_1: "f32[128]", arg251_1: "f32[128]", arg252_1: "f32[128]", arg253_1: "f32[128]", arg254_1: "f32[128]", arg255_1: "f32[128]", arg256_1: "f32[128]", arg257_1: "f32[128]", arg258_1: "f32[128]", arg259_1: "f32[128]", arg260_1: "f32[128]", arg261_1: "f32[128]", arg262_1: "f32[128]", arg263_1: "f32[128]", arg264_1: "f32[128]", arg265_1: "f32[128]", arg266_1: "f32[128]", arg267_1: "f32[128]", arg268_1: "f32[128]", arg269_1: "f32[128]", arg270_1: "f32[128]", arg271_1: "f32[256]", arg272_1: "f32[256]", arg273_1: "f32[512]", arg274_1: "f32[512]", arg275_1: "f32[512]", arg276_1: "f32[512]", arg277_1: "f32[256]", arg278_1: "f32[256]", arg279_1: "f32[256]", arg280_1: "f32[256]", arg281_1: "f32[256]", arg282_1: "f32[256]", arg283_1: "f32[256]", arg284_1: "f32[256]", arg285_1: "f32[256]", arg286_1: "f32[256]", arg287_1: "f32[256]", arg288_1: "f32[256]", arg289_1: "f32[256]", arg290_1: "f32[256]", arg291_1: "f32[256]", arg292_1: "f32[256]", arg293_1: "f32[256]", arg294_1: "f32[256]", arg295_1: "f32[256]", arg296_1: "f32[256]", arg297_1: "f32[256]", arg298_1: "f32[256]", arg299_1: "f32[256]", arg300_1: "f32[256]", arg301_1: "f32[256]", arg302_1: "f32[256]", arg303_1: "f32[256]", arg304_1: "f32[256]", arg305_1: "f32[256]", arg306_1: "f32[256]", arg307_1: "f32[256]", arg308_1: "f32[256]", arg309_1: "f32[256]", arg310_1: "f32[256]", arg311_1: "f32[512]", arg312_1: "f32[512]", arg313_1: "f32[1024]", arg314_1: "f32[1024]", arg315_1: "f32[1024]", arg316_1: "f32[1024]", arg317_1: "f32[512]", arg318_1: "f32[512]", arg319_1: "f32[512]", arg320_1: "f32[512]", arg321_1: "f32[512]", arg322_1: "f32[512]", arg323_1: "f32[512]", arg324_1: "f32[512]", arg325_1: "f32[512]", arg326_1: "f32[512]", arg327_1: "f32[512]", arg328_1: "f32[512]", arg329_1: "f32[512]", arg330_1: "f32[512]", arg331_1: "f32[512]", arg332_1: "f32[512]", arg333_1: "f32[512]", arg334_1: "f32[512]", arg335_1: "f32[1024]", arg336_1: "f32[1024]", arg337_1: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 32, 256, 256]" = torch.ops.aten.convolution.default(arg337_1, arg134_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg337_1 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type: "f32[32]" = torch.ops.prims.convert_element_type.default(arg203_1, torch.float32);  arg203_1 = None
    convert_element_type_1: "f32[32]" = torch.ops.prims.convert_element_type.default(arg204_1, torch.float32);  arg204_1 = None
    add: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[32]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
    mul_1: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[8, 32, 256, 256]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt: "b8[8, 32, 256, 256]" = torch.ops.aten.gt.Scalar(add_1, 0)
    mul_3: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(add_1, 0.01)
    where: "f32[8, 32, 256, 256]" = torch.ops.aten.where.self(gt, add_1, mul_3);  gt = add_1 = mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_1: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(where, arg135_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_2: "f32[64]" = torch.ops.prims.convert_element_type.default(arg205_1, torch.float32);  arg205_1 = None
    convert_element_type_3: "f32[64]" = torch.ops.prims.convert_element_type.default(arg206_1, torch.float32);  arg206_1 = None
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_4, -1);  mul_4 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
    mul_5: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_6: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_13);  mul_5 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_15);  mul_6 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_1: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_3, 0)
    mul_7: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_3, 0.01)
    where_1: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_1, add_3, mul_7);  gt_1 = add_3 = mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(where_1, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_1 = arg136_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_4: "f32[128]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.float32);  arg207_1 = None
    convert_element_type_5: "f32[128]" = torch.ops.prims.convert_element_type.default(arg208_1, torch.float32);  arg208_1 = None
    add_4: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[128]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_8: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_8, -1);  mul_8 = None
    unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
    mul_9: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
    unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_10: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_9, unsqueeze_21);  mul_9 = unsqueeze_21 = None
    unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
    unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_23);  mul_10 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_2: "b8[8, 128, 128, 128]" = torch.ops.aten.gt.Scalar(add_5, 0)
    mul_11: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(add_5, 0.01)
    where_2: "f32[8, 128, 128, 128]" = torch.ops.aten.where.self(gt_2, add_5, mul_11);  gt_2 = add_5 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(where_2, [64, 64], 1)
    getitem_3: "f32[8, 64, 128, 128]" = split_with_sizes_1[1];  split_with_sizes_1 = None
    convolution_3: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(getitem_3, arg137_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_6: "f32[32]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.float32);  arg209_1 = None
    convert_element_type_7: "f32[32]" = torch.ops.prims.convert_element_type.default(arg210_1, torch.float32);  arg210_1 = None
    add_6: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[32]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[32]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_27: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
    mul_13: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
    unsqueeze_29: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_14: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_29);  mul_13 = unsqueeze_29 = None
    unsqueeze_30: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
    unsqueeze_31: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_31);  mul_14 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_3: "b8[8, 32, 128, 128]" = torch.ops.aten.gt.Scalar(add_7, 0)
    mul_15: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_7, 0.01)
    where_3: "f32[8, 32, 128, 128]" = torch.ops.aten.where.self(gt_3, add_7, mul_15);  gt_3 = add_7 = mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(where_3, arg138_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_3 = arg138_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_8: "f32[64]" = torch.ops.prims.convert_element_type.default(arg211_1, torch.float32);  arg211_1 = None
    convert_element_type_9: "f32[64]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.float32);  arg212_1 = None
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[64]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_16, -1);  mul_16 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
    mul_17: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_18: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_37);  mul_17 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_18, unsqueeze_39);  mul_18 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_4: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_9, 0)
    mul_19: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, 0.01)
    where_4: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_4, add_9, mul_19);  gt_4 = add_9 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_10: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(where_4, getitem_3);  where_4 = getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(add_10, arg139_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_10 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_10: "f32[64]" = torch.ops.prims.convert_element_type.default(arg213_1, torch.float32);  arg213_1 = None
    convert_element_type_11: "f32[64]" = torch.ops.prims.convert_element_type.default(arg214_1, torch.float32);  arg214_1 = None
    add_11: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[64]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
    reciprocal_5: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_20, -1);  mul_20 = None
    unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
    mul_21: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
    unsqueeze_45: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_22: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_45);  mul_21 = unsqueeze_45 = None
    unsqueeze_46: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
    unsqueeze_47: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_12: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_47);  mul_22 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_5: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_12, 0)
    mul_23: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_12, 0.01)
    where_5: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_5, add_12, mul_23);  gt_5 = add_12 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(where_2, [64, 64], 1);  where_2 = None
    getitem_4: "f32[8, 64, 128, 128]" = split_with_sizes_2[0];  split_with_sizes_2 = None
    cat: "f32[8, 128, 128, 128]" = torch.ops.aten.cat.default([getitem_4, where_5], 1);  getitem_4 = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(cat, arg140_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_12: "f32[64]" = torch.ops.prims.convert_element_type.default(arg215_1, torch.float32);  arg215_1 = None
    convert_element_type_13: "f32[64]" = torch.ops.prims.convert_element_type.default(arg216_1, torch.float32);  arg216_1 = None
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[64]" = torch.ops.aten.sqrt.default(add_13);  add_13 = None
    reciprocal_6: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_24: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
    mul_25: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
    unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_26: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_53);  mul_25 = unsqueeze_53 = None
    unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
    unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_14: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_55);  mul_26 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_6: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(add_14, 0)
    mul_27: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, 0.01)
    where_6: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_6, add_14, mul_27);  gt_6 = add_14 = mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_7: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(where_6, arg141_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where_6 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_14: "f32[128]" = torch.ops.prims.convert_element_type.default(arg217_1, torch.float32);  arg217_1 = None
    convert_element_type_15: "f32[128]" = torch.ops.prims.convert_element_type.default(arg218_1, torch.float32);  arg218_1 = None
    add_15: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[128]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    reciprocal_7: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_28: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_28, -1);  mul_28 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
    mul_29: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_30: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_61);  mul_29 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_16: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_63);  mul_30 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_7: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_16, 0)
    mul_31: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_16, 0.01)
    where_7: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_7, add_16, mul_31);  gt_7 = add_16 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(where_7, arg142_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_7 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_16: "f32[128]" = torch.ops.prims.convert_element_type.default(arg219_1, torch.float32);  arg219_1 = None
    convert_element_type_17: "f32[128]" = torch.ops.prims.convert_element_type.default(arg220_1, torch.float32);  arg220_1 = None
    add_17: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[128]" = torch.ops.aten.sqrt.default(add_17);  add_17 = None
    reciprocal_8: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_32: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_32, -1);  mul_32 = None
    unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
    mul_33: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_34: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_33, unsqueeze_69);  mul_33 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_18: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_71);  mul_34 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_8: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_18, 0)
    mul_35: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_18, 0.01)
    where_8: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_8, add_18, mul_35);  gt_8 = add_18 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(where_8, [64, 64], 1)
    getitem_9: "f32[8, 64, 64, 64]" = split_with_sizes_4[1];  split_with_sizes_4 = None
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_9, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_18: "f32[64]" = torch.ops.prims.convert_element_type.default(arg221_1, torch.float32);  arg221_1 = None
    convert_element_type_19: "f32[64]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.float32);  arg222_1 = None
    add_19: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[64]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_9: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
    mul_37: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
    unsqueeze_77: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_38: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_77);  mul_37 = unsqueeze_77 = None
    unsqueeze_78: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
    unsqueeze_79: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_20: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_79);  mul_38 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_9: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_20, 0)
    mul_39: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_20, 0.01)
    where_9: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_9, add_20, mul_39);  gt_9 = add_20 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(where_9, arg144_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_9 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_20: "f32[64]" = torch.ops.prims.convert_element_type.default(arg223_1, torch.float32);  arg223_1 = None
    convert_element_type_21: "f32[64]" = torch.ops.prims.convert_element_type.default(arg224_1, torch.float32);  arg224_1 = None
    add_21: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[64]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_10: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_40, -1);  mul_40 = None
    unsqueeze_83: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
    mul_41: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
    unsqueeze_85: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_42: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_85);  mul_41 = unsqueeze_85 = None
    unsqueeze_86: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
    unsqueeze_87: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_22: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_87);  mul_42 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_10: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_22, 0)
    mul_43: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_22, 0.01)
    where_10: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_10, add_22, mul_43);  gt_10 = add_22 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_23: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(where_10, getitem_9);  where_10 = getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(add_23, arg145_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_22: "f32[64]" = torch.ops.prims.convert_element_type.default(arg225_1, torch.float32);  arg225_1 = None
    convert_element_type_23: "f32[64]" = torch.ops.prims.convert_element_type.default(arg226_1, torch.float32);  arg226_1 = None
    add_24: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[64]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_11: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_44: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_44, -1);  mul_44 = None
    unsqueeze_91: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
    mul_45: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
    unsqueeze_93: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_46: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_45, unsqueeze_93);  mul_45 = unsqueeze_93 = None
    unsqueeze_94: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
    unsqueeze_95: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_25: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_95);  mul_46 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_11: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_25, 0)
    mul_47: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_25, 0.01)
    where_11: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_11, add_25, mul_47);  gt_11 = add_25 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(where_11, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_11 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_24: "f32[64]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.float32);  arg227_1 = None
    convert_element_type_25: "f32[64]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.float32);  arg228_1 = None
    add_26: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[64]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_12: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_48: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_99: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
    mul_49: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
    unsqueeze_101: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_50: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_101);  mul_49 = unsqueeze_101 = None
    unsqueeze_102: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
    unsqueeze_103: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_27: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_103);  mul_50 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_12: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_27, 0)
    mul_51: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_27, 0.01)
    where_12: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_12, add_27, mul_51);  gt_12 = add_27 = mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_28: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(where_12, add_23);  where_12 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(add_28, arg147_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_28 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_26: "f32[64]" = torch.ops.prims.convert_element_type.default(arg229_1, torch.float32);  arg229_1 = None
    convert_element_type_27: "f32[64]" = torch.ops.prims.convert_element_type.default(arg230_1, torch.float32);  arg230_1 = None
    add_29: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[64]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
    reciprocal_13: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_52, -1);  mul_52 = None
    unsqueeze_107: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
    mul_53: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
    unsqueeze_109: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_54: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_53, unsqueeze_109);  mul_53 = unsqueeze_109 = None
    unsqueeze_110: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
    unsqueeze_111: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_30: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_111);  mul_54 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_13: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(add_30, 0)
    mul_55: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_30, 0.01)
    where_13: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_13, add_30, mul_55);  gt_13 = add_30 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(where_8, [64, 64], 1);  where_8 = None
    getitem_10: "f32[8, 64, 64, 64]" = split_with_sizes_5[0];  split_with_sizes_5 = None
    cat_1: "f32[8, 128, 64, 64]" = torch.ops.aten.cat.default([getitem_10, where_13], 1);  getitem_10 = where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(cat_1, arg148_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_1 = arg148_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_28: "f32[128]" = torch.ops.prims.convert_element_type.default(arg231_1, torch.float32);  arg231_1 = None
    convert_element_type_29: "f32[128]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.float32);  arg232_1 = None
    add_31: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[128]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
    reciprocal_14: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_56: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_56, -1);  mul_56 = None
    unsqueeze_115: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
    mul_57: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
    unsqueeze_117: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_58: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_117);  mul_57 = unsqueeze_117 = None
    unsqueeze_118: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
    unsqueeze_119: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_32: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_119);  mul_58 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_14: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(add_32, 0)
    mul_59: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_32, 0.01)
    where_14: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_14, add_32, mul_59);  gt_14 = add_32 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_15: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(where_14, arg149_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where_14 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_30: "f32[256]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.float32);  arg233_1 = None
    convert_element_type_31: "f32[256]" = torch.ops.prims.convert_element_type.default(arg234_1, torch.float32);  arg234_1 = None
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[256]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_15: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_60: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_123: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
    mul_61: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
    unsqueeze_125: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_62: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_125);  mul_61 = unsqueeze_125 = None
    unsqueeze_126: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
    unsqueeze_127: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_34: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_127);  mul_62 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_15: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_34, 0)
    mul_63: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_34, 0.01)
    where_15: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_15, add_34, mul_63);  gt_15 = add_34 = mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(where_15, arg150_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_15 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_32: "f32[256]" = torch.ops.prims.convert_element_type.default(arg235_1, torch.float32);  arg235_1 = None
    convert_element_type_33: "f32[256]" = torch.ops.prims.convert_element_type.default(arg236_1, torch.float32);  arg236_1 = None
    add_35: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[256]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
    reciprocal_16: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_64: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_64, -1);  mul_64 = None
    unsqueeze_131: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
    mul_65: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
    unsqueeze_133: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_66: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_133);  mul_65 = unsqueeze_133 = None
    unsqueeze_134: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
    unsqueeze_135: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_36: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_66, unsqueeze_135);  mul_66 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_16: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_36, 0)
    mul_67: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_36, 0.01)
    where_16: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_16, add_36, mul_67);  gt_16 = add_36 = mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(where_16, [128, 128], 1)
    getitem_15: "f32[8, 128, 32, 32]" = split_with_sizes_7[1];  split_with_sizes_7 = None
    convolution_17: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(getitem_15, arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg151_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_34: "f32[128]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.float32);  arg237_1 = None
    convert_element_type_35: "f32[128]" = torch.ops.prims.convert_element_type.default(arg238_1, torch.float32);  arg238_1 = None
    add_37: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[128]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
    reciprocal_17: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_68: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_68, -1);  mul_68 = None
    unsqueeze_139: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
    mul_69: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
    unsqueeze_141: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_70: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_69, unsqueeze_141);  mul_69 = unsqueeze_141 = None
    unsqueeze_142: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
    unsqueeze_143: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_38: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_143);  mul_70 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_17: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_38, 0)
    mul_71: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_38, 0.01)
    where_17: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_17, add_38, mul_71);  gt_17 = add_38 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_17, arg152_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_17 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_36: "f32[128]" = torch.ops.prims.convert_element_type.default(arg239_1, torch.float32);  arg239_1 = None
    convert_element_type_37: "f32[128]" = torch.ops.prims.convert_element_type.default(arg240_1, torch.float32);  arg240_1 = None
    add_39: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[128]" = torch.ops.aten.sqrt.default(add_39);  add_39 = None
    reciprocal_18: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_72: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_147: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
    mul_73: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
    unsqueeze_149: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_74: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_149);  mul_73 = unsqueeze_149 = None
    unsqueeze_150: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
    unsqueeze_151: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_40: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_151);  mul_74 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_18: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_40, 0)
    mul_75: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_40, 0.01)
    where_18: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_18, add_40, mul_75);  gt_18 = add_40 = mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_41: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_18, getitem_15);  where_18 = getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_41, arg153_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_38: "f32[128]" = torch.ops.prims.convert_element_type.default(arg241_1, torch.float32);  arg241_1 = None
    convert_element_type_39: "f32[128]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.float32);  arg242_1 = None
    add_42: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[128]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_19: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_76: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_76, -1);  mul_76 = None
    unsqueeze_155: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
    mul_77: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
    unsqueeze_157: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_78: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_157);  mul_77 = unsqueeze_157 = None
    unsqueeze_158: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
    unsqueeze_159: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_43: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_159);  mul_78 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_19: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_43, 0)
    mul_79: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_43, 0.01)
    where_19: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_19, add_43, mul_79);  gt_19 = add_43 = mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_19, arg154_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_19 = arg154_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_40: "f32[128]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.float32);  arg243_1 = None
    convert_element_type_41: "f32[128]" = torch.ops.prims.convert_element_type.default(arg244_1, torch.float32);  arg244_1 = None
    add_44: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[128]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_20: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_80: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_80, -1);  mul_80 = None
    unsqueeze_163: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  convolution_20 = unsqueeze_161 = None
    mul_81: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
    unsqueeze_165: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_82: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_165);  mul_81 = unsqueeze_165 = None
    unsqueeze_166: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
    unsqueeze_167: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_45: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_82, unsqueeze_167);  mul_82 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_20: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_45, 0)
    mul_83: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_45, 0.01)
    where_20: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_20, add_45, mul_83);  gt_20 = add_45 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_46: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_20, add_41);  where_20 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_46, arg155_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_42: "f32[128]" = torch.ops.prims.convert_element_type.default(arg245_1, torch.float32);  arg245_1 = None
    convert_element_type_43: "f32[128]" = torch.ops.prims.convert_element_type.default(arg246_1, torch.float32);  arg246_1 = None
    add_47: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[128]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_21: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_84: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_171: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  convolution_21 = unsqueeze_169 = None
    mul_85: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
    unsqueeze_173: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_86: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_173);  mul_85 = unsqueeze_173 = None
    unsqueeze_174: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
    unsqueeze_175: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_48: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_175);  mul_86 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_21: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_48, 0)
    mul_87: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_48, 0.01)
    where_21: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_21, add_48, mul_87);  gt_21 = add_48 = mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_21, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_21 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_44: "f32[128]" = torch.ops.prims.convert_element_type.default(arg247_1, torch.float32);  arg247_1 = None
    convert_element_type_45: "f32[128]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.float32);  arg248_1 = None
    add_49: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[128]" = torch.ops.aten.sqrt.default(add_49);  add_49 = None
    reciprocal_22: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_88, -1);  mul_88 = None
    unsqueeze_179: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  convolution_22 = unsqueeze_177 = None
    mul_89: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
    unsqueeze_181: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_90: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_181);  mul_89 = unsqueeze_181 = None
    unsqueeze_182: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
    unsqueeze_183: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_50: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_183);  mul_90 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_22: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_50, 0)
    mul_91: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_50, 0.01)
    where_22: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_22, add_50, mul_91);  gt_22 = add_50 = mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_51: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_22, add_46);  where_22 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_51, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_46: "f32[128]" = torch.ops.prims.convert_element_type.default(arg249_1, torch.float32);  arg249_1 = None
    convert_element_type_47: "f32[128]" = torch.ops.prims.convert_element_type.default(arg250_1, torch.float32);  arg250_1 = None
    add_52: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[128]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_23: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_92: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_92, -1);  mul_92 = None
    unsqueeze_187: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  convolution_23 = unsqueeze_185 = None
    mul_93: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
    unsqueeze_189: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_94: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_189);  mul_93 = unsqueeze_189 = None
    unsqueeze_190: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
    unsqueeze_191: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_53: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_191);  mul_94 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_23: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_53, 0)
    mul_95: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_53, 0.01)
    where_23: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_23, add_53, mul_95);  gt_23 = add_53 = mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_23, arg158_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_23 = arg158_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_48: "f32[128]" = torch.ops.prims.convert_element_type.default(arg251_1, torch.float32);  arg251_1 = None
    convert_element_type_49: "f32[128]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.float32);  arg252_1 = None
    add_54: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[128]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_24: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_96: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_195: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  convolution_24 = unsqueeze_193 = None
    mul_97: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
    unsqueeze_197: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_98: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_197);  mul_97 = unsqueeze_197 = None
    unsqueeze_198: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
    unsqueeze_199: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_55: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_199);  mul_98 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_24: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_55, 0)
    mul_99: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_55, 0.01)
    where_24: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_24, add_55, mul_99);  gt_24 = add_55 = mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_56: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_24, add_51);  where_24 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_56, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_50: "f32[128]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.float32);  arg253_1 = None
    convert_element_type_51: "f32[128]" = torch.ops.prims.convert_element_type.default(arg254_1, torch.float32);  arg254_1 = None
    add_57: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[128]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
    reciprocal_25: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_100: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_100, -1);  mul_100 = None
    unsqueeze_203: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  convolution_25 = unsqueeze_201 = None
    mul_101: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
    unsqueeze_205: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_102: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_205);  mul_101 = unsqueeze_205 = None
    unsqueeze_206: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
    unsqueeze_207: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_58: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_102, unsqueeze_207);  mul_102 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_25: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_58, 0)
    mul_103: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_58, 0.01)
    where_25: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_25, add_58, mul_103);  gt_25 = add_58 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_25, arg160_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_25 = arg160_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_52: "f32[128]" = torch.ops.prims.convert_element_type.default(arg255_1, torch.float32);  arg255_1 = None
    convert_element_type_53: "f32[128]" = torch.ops.prims.convert_element_type.default(arg256_1, torch.float32);  arg256_1 = None
    add_59: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[128]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
    reciprocal_26: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_104: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_104, -1);  mul_104 = None
    unsqueeze_211: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  convolution_26 = unsqueeze_209 = None
    mul_105: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
    unsqueeze_213: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_106: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_213);  mul_105 = unsqueeze_213 = None
    unsqueeze_214: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
    unsqueeze_215: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_60: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_215);  mul_106 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_26: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_60, 0)
    mul_107: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_60, 0.01)
    where_26: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_26, add_60, mul_107);  gt_26 = add_60 = mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_61: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_26, add_56);  where_26 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_61, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_54: "f32[128]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.float32);  arg257_1 = None
    convert_element_type_55: "f32[128]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.float32);  arg258_1 = None
    add_62: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[128]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_27: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_108: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_219: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  convolution_27 = unsqueeze_217 = None
    mul_109: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
    unsqueeze_221: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_110: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_221);  mul_109 = unsqueeze_221 = None
    unsqueeze_222: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
    unsqueeze_223: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_63: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_223);  mul_110 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_27: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_63, 0)
    mul_111: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_63, 0.01)
    where_27: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_27, add_63, mul_111);  gt_27 = add_63 = mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_27, arg162_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_27 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_56: "f32[128]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.float32);  arg259_1 = None
    convert_element_type_57: "f32[128]" = torch.ops.prims.convert_element_type.default(arg260_1, torch.float32);  arg260_1 = None
    add_64: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[128]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_28: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_112: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_112, -1);  mul_112 = None
    unsqueeze_227: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  convolution_28 = unsqueeze_225 = None
    mul_113: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
    unsqueeze_229: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_114: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_229);  mul_113 = unsqueeze_229 = None
    unsqueeze_230: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
    unsqueeze_231: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_65: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_231);  mul_114 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_28: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_65, 0)
    mul_115: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_65, 0.01)
    where_28: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_28, add_65, mul_115);  gt_28 = add_65 = mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_66: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_28, add_61);  where_28 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_66, arg163_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_58: "f32[128]" = torch.ops.prims.convert_element_type.default(arg261_1, torch.float32);  arg261_1 = None
    convert_element_type_59: "f32[128]" = torch.ops.prims.convert_element_type.default(arg262_1, torch.float32);  arg262_1 = None
    add_67: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[128]" = torch.ops.aten.sqrt.default(add_67);  add_67 = None
    reciprocal_29: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_116, -1);  mul_116 = None
    unsqueeze_235: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  convolution_29 = unsqueeze_233 = None
    mul_117: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
    unsqueeze_237: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_118: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_117, unsqueeze_237);  mul_117 = unsqueeze_237 = None
    unsqueeze_238: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
    unsqueeze_239: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_68: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_239);  mul_118 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_29: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_68, 0)
    mul_119: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_68, 0.01)
    where_29: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_29, add_68, mul_119);  gt_29 = add_68 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_29, arg164_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_29 = arg164_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_60: "f32[128]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.float32);  arg263_1 = None
    convert_element_type_61: "f32[128]" = torch.ops.prims.convert_element_type.default(arg264_1, torch.float32);  arg264_1 = None
    add_69: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[128]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    reciprocal_30: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_120: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_243: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  convolution_30 = unsqueeze_241 = None
    mul_121: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
    unsqueeze_245: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_122: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_245);  mul_121 = unsqueeze_245 = None
    unsqueeze_246: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
    unsqueeze_247: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_70: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_247);  mul_122 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_30: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_70, 0)
    mul_123: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_70, 0.01)
    where_30: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_30, add_70, mul_123);  gt_30 = add_70 = mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_71: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_30, add_66);  where_30 = add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_31: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_71, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_62: "f32[128]" = torch.ops.prims.convert_element_type.default(arg265_1, torch.float32);  arg265_1 = None
    convert_element_type_63: "f32[128]" = torch.ops.prims.convert_element_type.default(arg266_1, torch.float32);  arg266_1 = None
    add_72: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[128]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_31: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_124: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_124, -1);  mul_124 = None
    unsqueeze_251: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  convolution_31 = unsqueeze_249 = None
    mul_125: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
    unsqueeze_253: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_126: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_125, unsqueeze_253);  mul_125 = unsqueeze_253 = None
    unsqueeze_254: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
    unsqueeze_255: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_73: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_255);  mul_126 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_31: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_73, 0)
    mul_127: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_73, 0.01)
    where_31: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_31, add_73, mul_127);  gt_31 = add_73 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(where_31, arg166_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_31 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_64: "f32[128]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.float32);  arg267_1 = None
    convert_element_type_65: "f32[128]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.float32);  arg268_1 = None
    add_74: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[128]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_32: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_128: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_128, -1);  mul_128 = None
    unsqueeze_259: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  convolution_32 = unsqueeze_257 = None
    mul_129: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
    unsqueeze_261: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_130: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_261);  mul_129 = unsqueeze_261 = None
    unsqueeze_262: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
    unsqueeze_263: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_75: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_130, unsqueeze_263);  mul_130 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_32: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_75, 0)
    mul_131: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_75, 0.01)
    where_32: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_32, add_75, mul_131);  gt_32 = add_75 = mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_76: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(where_32, add_71);  where_32 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(add_76, arg167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_76 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_66: "f32[128]" = torch.ops.prims.convert_element_type.default(arg269_1, torch.float32);  arg269_1 = None
    convert_element_type_67: "f32[128]" = torch.ops.prims.convert_element_type.default(arg270_1, torch.float32);  arg270_1 = None
    add_77: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[128]" = torch.ops.aten.sqrt.default(add_77);  add_77 = None
    reciprocal_33: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_132: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_267: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  convolution_33 = unsqueeze_265 = None
    mul_133: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
    unsqueeze_269: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_134: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_269);  mul_133 = unsqueeze_269 = None
    unsqueeze_270: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
    unsqueeze_271: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_78: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_271);  mul_134 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_33: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(add_78, 0)
    mul_135: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_78, 0.01)
    where_33: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_33, add_78, mul_135);  gt_33 = add_78 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(where_16, [128, 128], 1);  where_16 = None
    getitem_16: "f32[8, 128, 32, 32]" = split_with_sizes_8[0];  split_with_sizes_8 = None
    cat_2: "f32[8, 256, 32, 32]" = torch.ops.aten.cat.default([getitem_16, where_33], 1);  getitem_16 = where_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(cat_2, arg168_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_2 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_68: "f32[256]" = torch.ops.prims.convert_element_type.default(arg271_1, torch.float32);  arg271_1 = None
    convert_element_type_69: "f32[256]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.float32);  arg272_1 = None
    add_79: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[256]" = torch.ops.aten.sqrt.default(add_79);  add_79 = None
    reciprocal_34: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_136: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_136, -1);  mul_136 = None
    unsqueeze_275: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  convolution_34 = unsqueeze_273 = None
    mul_137: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
    unsqueeze_277: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_138: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_137, unsqueeze_277);  mul_137 = unsqueeze_277 = None
    unsqueeze_278: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
    unsqueeze_279: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_80: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_138, unsqueeze_279);  mul_138 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_34: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(add_80, 0)
    mul_139: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_80, 0.01)
    where_34: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_34, add_80, mul_139);  gt_34 = add_80 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_35: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(where_34, arg169_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where_34 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_70: "f32[512]" = torch.ops.prims.convert_element_type.default(arg273_1, torch.float32);  arg273_1 = None
    convert_element_type_71: "f32[512]" = torch.ops.prims.convert_element_type.default(arg274_1, torch.float32);  arg274_1 = None
    add_81: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[512]" = torch.ops.aten.sqrt.default(add_81);  add_81 = None
    reciprocal_35: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_140: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_140, -1);  mul_140 = None
    unsqueeze_283: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  convolution_35 = unsqueeze_281 = None
    mul_141: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
    unsqueeze_285: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_142: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_285);  mul_141 = unsqueeze_285 = None
    unsqueeze_286: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
    unsqueeze_287: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_82: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_287);  mul_142 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_35: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_82, 0)
    mul_143: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_82, 0.01)
    where_35: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_35, add_82, mul_143);  gt_35 = add_82 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_36: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(where_35, arg170_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_35 = arg170_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_72: "f32[512]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.float32);  arg275_1 = None
    convert_element_type_73: "f32[512]" = torch.ops.prims.convert_element_type.default(arg276_1, torch.float32);  arg276_1 = None
    add_83: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[512]" = torch.ops.aten.sqrt.default(add_83);  add_83 = None
    reciprocal_36: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_144: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_291: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  convolution_36 = unsqueeze_289 = None
    mul_145: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
    unsqueeze_293: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_146: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_293);  mul_145 = unsqueeze_293 = None
    unsqueeze_294: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
    unsqueeze_295: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_84: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_295);  mul_146 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_36: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_84, 0)
    mul_147: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_84, 0.01)
    where_36: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_36, add_84, mul_147);  gt_36 = add_84 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(where_36, [256, 256], 1)
    getitem_21: "f32[8, 256, 16, 16]" = split_with_sizes_10[1];  split_with_sizes_10 = None
    convolution_37: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(getitem_21, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg171_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_74: "f32[256]" = torch.ops.prims.convert_element_type.default(arg277_1, torch.float32);  arg277_1 = None
    convert_element_type_75: "f32[256]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.float32);  arg278_1 = None
    add_85: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[256]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
    reciprocal_37: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_148: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_148, -1);  mul_148 = None
    unsqueeze_299: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  convolution_37 = unsqueeze_297 = None
    mul_149: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
    unsqueeze_301: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_150: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_149, unsqueeze_301);  mul_149 = unsqueeze_301 = None
    unsqueeze_302: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
    unsqueeze_303: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_86: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_150, unsqueeze_303);  mul_150 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_37: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_86, 0)
    mul_151: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_86, 0.01)
    where_37: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_37, add_86, mul_151);  gt_37 = add_86 = mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_37, arg172_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_37 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_76: "f32[256]" = torch.ops.prims.convert_element_type.default(arg279_1, torch.float32);  arg279_1 = None
    convert_element_type_77: "f32[256]" = torch.ops.prims.convert_element_type.default(arg280_1, torch.float32);  arg280_1 = None
    add_87: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[256]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    reciprocal_38: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_152: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_152, -1);  mul_152 = None
    unsqueeze_307: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  convolution_38 = unsqueeze_305 = None
    mul_153: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
    unsqueeze_309: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_154: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_309);  mul_153 = unsqueeze_309 = None
    unsqueeze_310: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
    unsqueeze_311: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_88: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_311);  mul_154 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_38: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_88, 0)
    mul_155: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_88, 0.01)
    where_38: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_38, add_88, mul_155);  gt_38 = add_88 = mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_89: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_38, getitem_21);  where_38 = getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_89, arg173_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_78: "f32[256]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.float32);  arg281_1 = None
    convert_element_type_79: "f32[256]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.float32);  arg282_1 = None
    add_90: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[256]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_39: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_156: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_315: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  convolution_39 = unsqueeze_313 = None
    mul_157: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
    unsqueeze_317: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_158: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_317);  mul_157 = unsqueeze_317 = None
    unsqueeze_318: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
    unsqueeze_319: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_91: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_319);  mul_158 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_39: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_91, 0)
    mul_159: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_91, 0.01)
    where_39: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_39, add_91, mul_159);  gt_39 = add_91 = mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_40: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_39, arg174_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_39 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_80: "f32[256]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.float32);  arg283_1 = None
    convert_element_type_81: "f32[256]" = torch.ops.prims.convert_element_type.default(arg284_1, torch.float32);  arg284_1 = None
    add_92: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[256]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_40: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_160: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
    unsqueeze_323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  convolution_40 = unsqueeze_321 = None
    mul_161: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
    unsqueeze_325: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_162: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_325);  mul_161 = unsqueeze_325 = None
    unsqueeze_326: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
    unsqueeze_327: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_93: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_327);  mul_162 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_40: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_93, 0)
    mul_163: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_93, 0.01)
    where_40: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_40, add_93, mul_163);  gt_40 = add_93 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_94: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_40, add_89);  where_40 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_41: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_94, arg175_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg175_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_82: "f32[256]" = torch.ops.prims.convert_element_type.default(arg285_1, torch.float32);  arg285_1 = None
    convert_element_type_83: "f32[256]" = torch.ops.prims.convert_element_type.default(arg286_1, torch.float32);  arg286_1 = None
    add_95: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[256]" = torch.ops.aten.sqrt.default(add_95);  add_95 = None
    reciprocal_41: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_164: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_164, -1);  mul_164 = None
    unsqueeze_331: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
    mul_165: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
    unsqueeze_333: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_166: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_165, unsqueeze_333);  mul_165 = unsqueeze_333 = None
    unsqueeze_334: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
    unsqueeze_335: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_96: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_335);  mul_166 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_41: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_96, 0)
    mul_167: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_96, 0.01)
    where_41: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_41, add_96, mul_167);  gt_41 = add_96 = mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_42: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_41, arg176_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_41 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_84: "f32[256]" = torch.ops.prims.convert_element_type.default(arg287_1, torch.float32);  arg287_1 = None
    convert_element_type_85: "f32[256]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.float32);  arg288_1 = None
    add_97: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[256]" = torch.ops.aten.sqrt.default(add_97);  add_97 = None
    reciprocal_42: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_168: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
    unsqueeze_339: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
    mul_169: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
    unsqueeze_341: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_170: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_341);  mul_169 = unsqueeze_341 = None
    unsqueeze_342: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
    unsqueeze_343: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_98: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_343);  mul_170 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_42: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_98, 0)
    mul_171: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_98, 0.01)
    where_42: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_42, add_98, mul_171);  gt_42 = add_98 = mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_99: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_42, add_94);  where_42 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_43: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_99, arg177_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_86: "f32[256]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.float32);  arg289_1 = None
    convert_element_type_87: "f32[256]" = torch.ops.prims.convert_element_type.default(arg290_1, torch.float32);  arg290_1 = None
    add_100: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[256]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    reciprocal_43: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_172, -1);  mul_172 = None
    unsqueeze_347: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
    mul_173: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
    unsqueeze_349: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_174: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_349);  mul_173 = unsqueeze_349 = None
    unsqueeze_350: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
    unsqueeze_351: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_101: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_351);  mul_174 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_43: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_101, 0)
    mul_175: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_101, 0.01)
    where_43: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_43, add_101, mul_175);  gt_43 = add_101 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_43, arg178_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_43 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_88: "f32[256]" = torch.ops.prims.convert_element_type.default(arg291_1, torch.float32);  arg291_1 = None
    convert_element_type_89: "f32[256]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.float32);  arg292_1 = None
    add_102: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[256]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
    reciprocal_44: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_176: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_176, -1);  mul_176 = None
    unsqueeze_355: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
    mul_177: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
    unsqueeze_357: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_178: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_177, unsqueeze_357);  mul_177 = unsqueeze_357 = None
    unsqueeze_358: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
    unsqueeze_359: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_103: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_178, unsqueeze_359);  mul_178 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_44: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_103, 0)
    mul_179: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_103, 0.01)
    where_44: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_44, add_103, mul_179);  gt_44 = add_103 = mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_104: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_44, add_99);  where_44 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_104, arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_90: "f32[256]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.float32);  arg293_1 = None
    convert_element_type_91: "f32[256]" = torch.ops.prims.convert_element_type.default(arg294_1, torch.float32);  arg294_1 = None
    add_105: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[256]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    reciprocal_45: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_180: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
    unsqueeze_363: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
    mul_181: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
    unsqueeze_365: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_182: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_365);  mul_181 = unsqueeze_365 = None
    unsqueeze_366: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
    unsqueeze_367: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_106: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_367);  mul_182 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_45: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_106, 0)
    mul_183: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_106, 0.01)
    where_45: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_45, add_106, mul_183);  gt_45 = add_106 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_46: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_45, arg180_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_45 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_92: "f32[256]" = torch.ops.prims.convert_element_type.default(arg295_1, torch.float32);  arg295_1 = None
    convert_element_type_93: "f32[256]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.float32);  arg296_1 = None
    add_107: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[256]" = torch.ops.aten.sqrt.default(add_107);  add_107 = None
    reciprocal_46: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_184: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_184, -1);  mul_184 = None
    unsqueeze_371: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
    mul_185: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
    unsqueeze_373: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_186: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_185, unsqueeze_373);  mul_185 = unsqueeze_373 = None
    unsqueeze_374: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
    unsqueeze_375: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_108: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_186, unsqueeze_375);  mul_186 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_46: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_108, 0)
    mul_187: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_108, 0.01)
    where_46: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_46, add_108, mul_187);  gt_46 = add_108 = mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_109: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_46, add_104);  where_46 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_47: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_109, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_94: "f32[256]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.float32);  arg297_1 = None
    convert_element_type_95: "f32[256]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.float32);  arg298_1 = None
    add_110: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[256]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_47: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_188: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_188, -1);  mul_188 = None
    unsqueeze_379: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
    mul_189: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
    unsqueeze_381: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_190: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_381);  mul_189 = unsqueeze_381 = None
    unsqueeze_382: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
    unsqueeze_383: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_111: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_383);  mul_190 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_47: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_111, 0)
    mul_191: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_111, 0.01)
    where_47: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_47, add_111, mul_191);  gt_47 = add_111 = mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_47, arg182_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_47 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_96: "f32[256]" = torch.ops.prims.convert_element_type.default(arg299_1, torch.float32);  arg299_1 = None
    convert_element_type_97: "f32[256]" = torch.ops.prims.convert_element_type.default(arg300_1, torch.float32);  arg300_1 = None
    add_112: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[256]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
    reciprocal_48: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_192: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_387: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
    mul_193: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
    unsqueeze_389: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_194: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_389);  mul_193 = unsqueeze_389 = None
    unsqueeze_390: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
    unsqueeze_391: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_113: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_391);  mul_194 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_48: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_113, 0)
    mul_195: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_113, 0.01)
    where_48: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_48, add_113, mul_195);  gt_48 = add_113 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_114: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_48, add_109);  where_48 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_114, arg183_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_98: "f32[256]" = torch.ops.prims.convert_element_type.default(arg301_1, torch.float32);  arg301_1 = None
    convert_element_type_99: "f32[256]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.float32);  arg302_1 = None
    add_115: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[256]" = torch.ops.aten.sqrt.default(add_115);  add_115 = None
    reciprocal_49: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_196: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_196, -1);  mul_196 = None
    unsqueeze_395: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
    mul_197: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
    unsqueeze_397: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_198: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_197, unsqueeze_397);  mul_197 = unsqueeze_397 = None
    unsqueeze_398: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
    unsqueeze_399: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_116: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_198, unsqueeze_399);  mul_198 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_49: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_116, 0)
    mul_199: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_116, 0.01)
    where_49: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_49, add_116, mul_199);  gt_49 = add_116 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_49, arg184_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_49 = arg184_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_100: "f32[256]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.float32);  arg303_1 = None
    convert_element_type_101: "f32[256]" = torch.ops.prims.convert_element_type.default(arg304_1, torch.float32);  arg304_1 = None
    add_117: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[256]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
    reciprocal_50: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_200: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_200, -1);  mul_200 = None
    unsqueeze_403: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
    mul_201: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
    unsqueeze_405: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_202: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_201, unsqueeze_405);  mul_201 = unsqueeze_405 = None
    unsqueeze_406: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
    unsqueeze_407: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_118: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_407);  mul_202 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_50: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_118, 0)
    mul_203: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_118, 0.01)
    where_50: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_50, add_118, mul_203);  gt_50 = add_118 = mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_119: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_50, add_114);  where_50 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_51: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_119, arg185_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_102: "f32[256]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.float32);  arg305_1 = None
    convert_element_type_103: "f32[256]" = torch.ops.prims.convert_element_type.default(arg306_1, torch.float32);  arg306_1 = None
    add_120: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[256]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_51: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
    unsqueeze_411: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
    mul_205: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
    unsqueeze_413: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_206: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_413);  mul_205 = unsqueeze_413 = None
    unsqueeze_414: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
    unsqueeze_415: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_121: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_415);  mul_206 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_51: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_121, 0)
    mul_207: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_121, 0.01)
    where_51: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_51, add_121, mul_207);  gt_51 = add_121 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_52: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(where_51, arg186_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_51 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_104: "f32[256]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.float32);  arg307_1 = None
    convert_element_type_105: "f32[256]" = torch.ops.prims.convert_element_type.default(arg308_1, torch.float32);  arg308_1 = None
    add_122: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[256]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
    reciprocal_52: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_208: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
    unsqueeze_419: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
    mul_209: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
    unsqueeze_421: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_210: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_421);  mul_209 = unsqueeze_421 = None
    unsqueeze_422: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
    unsqueeze_423: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_123: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_423);  mul_210 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_52: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_123, 0)
    mul_211: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_123, 0.01)
    where_52: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_52, add_123, mul_211);  gt_52 = add_123 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_124: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(where_52, add_119);  where_52 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_53: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(add_124, arg187_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_124 = arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_106: "f32[256]" = torch.ops.prims.convert_element_type.default(arg309_1, torch.float32);  arg309_1 = None
    convert_element_type_107: "f32[256]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.float32);  arg310_1 = None
    add_125: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_107, 1e-05);  convert_element_type_107 = None
    sqrt_53: "f32[256]" = torch.ops.aten.sqrt.default(add_125);  add_125 = None
    reciprocal_53: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_212: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
    unsqueeze_425: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_212, -1);  mul_212 = None
    unsqueeze_427: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_425);  convolution_53 = unsqueeze_425 = None
    mul_213: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
    unsqueeze_429: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_214: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_429);  mul_213 = unsqueeze_429 = None
    unsqueeze_430: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
    unsqueeze_431: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_126: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_214, unsqueeze_431);  mul_214 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_53: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(add_126, 0)
    mul_215: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_126, 0.01)
    where_53: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_53, add_126, mul_215);  gt_53 = add_126 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(where_36, [256, 256], 1);  where_36 = None
    getitem_22: "f32[8, 256, 16, 16]" = split_with_sizes_11[0];  split_with_sizes_11 = None
    cat_3: "f32[8, 512, 16, 16]" = torch.ops.aten.cat.default([getitem_22, where_53], 1);  getitem_22 = where_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(cat_3, arg188_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_3 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_108: "f32[512]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.float32);  arg311_1 = None
    convert_element_type_109: "f32[512]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.float32);  arg312_1 = None
    add_127: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
    sqrt_54: "f32[512]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
    reciprocal_54: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_216: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
    unsqueeze_433: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
    unsqueeze_435: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  convolution_54 = unsqueeze_433 = None
    mul_217: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
    unsqueeze_437: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_218: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_437);  mul_217 = unsqueeze_437 = None
    unsqueeze_438: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
    unsqueeze_439: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_128: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_439);  mul_218 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_54: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(add_128, 0)
    mul_219: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_128, 0.01)
    where_54: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_54, add_128, mul_219);  gt_54 = add_128 = mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_55: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(where_54, arg189_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  where_54 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_110: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg313_1, torch.float32);  arg313_1 = None
    convert_element_type_111: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg314_1, torch.float32);  arg314_1 = None
    add_129: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_111, 1e-05);  convert_element_type_111 = None
    sqrt_55: "f32[1024]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
    reciprocal_55: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_220: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
    unsqueeze_441: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_220, -1);  mul_220 = None
    unsqueeze_443: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  convolution_55 = unsqueeze_441 = None
    mul_221: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
    unsqueeze_445: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_222: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_221, unsqueeze_445);  mul_221 = unsqueeze_445 = None
    unsqueeze_446: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
    unsqueeze_447: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_130: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_222, unsqueeze_447);  mul_222 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_55: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_130, 0)
    mul_223: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_130, 0.01)
    where_55: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_55, add_130, mul_223);  gt_55 = add_130 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_56: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(where_55, arg190_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  where_55 = arg190_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_112: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.float32);  arg315_1 = None
    convert_element_type_113: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.float32);  arg316_1 = None
    add_131: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_113, 1e-05);  convert_element_type_113 = None
    sqrt_56: "f32[1024]" = torch.ops.aten.sqrt.default(add_131);  add_131 = None
    reciprocal_56: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_224: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_448: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_112, -1);  convert_element_type_112 = None
    unsqueeze_449: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    unsqueeze_450: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_224, -1);  mul_224 = None
    unsqueeze_451: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    sub_56: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_449);  convolution_56 = unsqueeze_449 = None
    mul_225: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
    unsqueeze_453: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_226: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_453);  mul_225 = unsqueeze_453 = None
    unsqueeze_454: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
    unsqueeze_455: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_132: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_455);  mul_226 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_56: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_132, 0)
    mul_227: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_132, 0.01)
    where_56: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_56, add_132, mul_227);  gt_56 = add_132 = mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(where_56, [512, 512], 1)
    getitem_27: "f32[8, 512, 8, 8]" = split_with_sizes_13[1];  split_with_sizes_13 = None
    convolution_57: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(getitem_27, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_114: "f32[512]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.float32);  arg317_1 = None
    convert_element_type_115: "f32[512]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.float32);  arg318_1 = None
    add_133: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_115, 1e-05);  convert_element_type_115 = None
    sqrt_57: "f32[512]" = torch.ops.aten.sqrt.default(add_133);  add_133 = None
    reciprocal_57: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_228: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_456: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
    unsqueeze_457: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    unsqueeze_458: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
    unsqueeze_459: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    sub_57: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_457);  convolution_57 = unsqueeze_457 = None
    mul_229: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
    unsqueeze_461: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_230: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_461);  mul_229 = unsqueeze_461 = None
    unsqueeze_462: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
    unsqueeze_463: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_134: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_463);  mul_230 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_57: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_134, 0)
    mul_231: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_134, 0.01)
    where_57: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_57, add_134, mul_231);  gt_57 = add_134 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_58: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_57, arg192_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_57 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_116: "f32[512]" = torch.ops.prims.convert_element_type.default(arg319_1, torch.float32);  arg319_1 = None
    convert_element_type_117: "f32[512]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.float32);  arg320_1 = None
    add_135: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_117, 1e-05);  convert_element_type_117 = None
    sqrt_58: "f32[512]" = torch.ops.aten.sqrt.default(add_135);  add_135 = None
    reciprocal_58: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_232: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_464: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_116, -1);  convert_element_type_116 = None
    unsqueeze_465: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    unsqueeze_466: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_232, -1);  mul_232 = None
    unsqueeze_467: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    sub_58: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_465);  convolution_58 = unsqueeze_465 = None
    mul_233: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
    unsqueeze_469: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_234: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_469);  mul_233 = unsqueeze_469 = None
    unsqueeze_470: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
    unsqueeze_471: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_136: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_234, unsqueeze_471);  mul_234 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_58: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_136, 0)
    mul_235: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_136, 0.01)
    where_58: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_58, add_136, mul_235);  gt_58 = add_136 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_137: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_58, getitem_27);  where_58 = getitem_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_59: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_137, arg193_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_118: "f32[512]" = torch.ops.prims.convert_element_type.default(arg321_1, torch.float32);  arg321_1 = None
    convert_element_type_119: "f32[512]" = torch.ops.prims.convert_element_type.default(arg322_1, torch.float32);  arg322_1 = None
    add_138: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_119, 1e-05);  convert_element_type_119 = None
    sqrt_59: "f32[512]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
    reciprocal_59: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_236: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_472: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_118, -1);  convert_element_type_118 = None
    unsqueeze_473: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    unsqueeze_474: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_236, -1);  mul_236 = None
    unsqueeze_475: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    sub_59: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_473);  convolution_59 = unsqueeze_473 = None
    mul_237: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
    unsqueeze_477: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_238: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_477);  mul_237 = unsqueeze_477 = None
    unsqueeze_478: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
    unsqueeze_479: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_139: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_238, unsqueeze_479);  mul_238 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_59: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_139, 0)
    mul_239: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_139, 0.01)
    where_59: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_59, add_139, mul_239);  gt_59 = add_139 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_60: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_59, arg194_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_59 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_120: "f32[512]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.float32);  arg323_1 = None
    convert_element_type_121: "f32[512]" = torch.ops.prims.convert_element_type.default(arg324_1, torch.float32);  arg324_1 = None
    add_140: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_121, 1e-05);  convert_element_type_121 = None
    sqrt_60: "f32[512]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
    reciprocal_60: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_240: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_480: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_120, -1);  convert_element_type_120 = None
    unsqueeze_481: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    unsqueeze_482: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
    unsqueeze_483: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    sub_60: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_481);  convolution_60 = unsqueeze_481 = None
    mul_241: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
    unsqueeze_485: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_242: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_485);  mul_241 = unsqueeze_485 = None
    unsqueeze_486: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
    unsqueeze_487: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_141: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_487);  mul_242 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_60: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_141, 0)
    mul_243: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_141, 0.01)
    where_60: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_60, add_141, mul_243);  gt_60 = add_141 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_142: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_60, add_137);  where_60 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_61: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_142, arg195_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_122: "f32[512]" = torch.ops.prims.convert_element_type.default(arg325_1, torch.float32);  arg325_1 = None
    convert_element_type_123: "f32[512]" = torch.ops.prims.convert_element_type.default(arg326_1, torch.float32);  arg326_1 = None
    add_143: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_123, 1e-05);  convert_element_type_123 = None
    sqrt_61: "f32[512]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
    reciprocal_61: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_244: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_488: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_122, -1);  convert_element_type_122 = None
    unsqueeze_489: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    unsqueeze_490: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_244, -1);  mul_244 = None
    unsqueeze_491: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    sub_61: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_489);  convolution_61 = unsqueeze_489 = None
    mul_245: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
    unsqueeze_493: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_246: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_493);  mul_245 = unsqueeze_493 = None
    unsqueeze_494: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
    unsqueeze_495: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_144: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_495);  mul_246 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_61: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_144, 0)
    mul_247: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_144, 0.01)
    where_61: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_61, add_144, mul_247);  gt_61 = add_144 = mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_62: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_61, arg196_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_61 = arg196_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_124: "f32[512]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.float32);  arg327_1 = None
    convert_element_type_125: "f32[512]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.float32);  arg328_1 = None
    add_145: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_125, 1e-05);  convert_element_type_125 = None
    sqrt_62: "f32[512]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
    reciprocal_62: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
    mul_248: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
    unsqueeze_496: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_124, -1);  convert_element_type_124 = None
    unsqueeze_497: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    unsqueeze_498: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_248, -1);  mul_248 = None
    unsqueeze_499: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    sub_62: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_497);  convolution_62 = unsqueeze_497 = None
    mul_249: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
    unsqueeze_500: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
    unsqueeze_501: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_250: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_249, unsqueeze_501);  mul_249 = unsqueeze_501 = None
    unsqueeze_502: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
    unsqueeze_503: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_146: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_250, unsqueeze_503);  mul_250 = unsqueeze_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_62: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_146, 0)
    mul_251: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_146, 0.01)
    where_62: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_62, add_146, mul_251);  gt_62 = add_146 = mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_147: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_62, add_142);  where_62 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_63: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_147, arg197_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_126: "f32[512]" = torch.ops.prims.convert_element_type.default(arg329_1, torch.float32);  arg329_1 = None
    convert_element_type_127: "f32[512]" = torch.ops.prims.convert_element_type.default(arg330_1, torch.float32);  arg330_1 = None
    add_148: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_127, 1e-05);  convert_element_type_127 = None
    sqrt_63: "f32[512]" = torch.ops.aten.sqrt.default(add_148);  add_148 = None
    reciprocal_63: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
    mul_252: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
    unsqueeze_504: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_126, -1);  convert_element_type_126 = None
    unsqueeze_505: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    unsqueeze_506: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
    unsqueeze_507: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    sub_63: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_505);  convolution_63 = unsqueeze_505 = None
    mul_253: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
    unsqueeze_508: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
    unsqueeze_509: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_254: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_509);  mul_253 = unsqueeze_509 = None
    unsqueeze_510: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
    unsqueeze_511: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_149: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_511);  mul_254 = unsqueeze_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_63: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_149, 0)
    mul_255: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_149, 0.01)
    where_63: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_63, add_149, mul_255);  gt_63 = add_149 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_64: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(where_63, arg198_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  where_63 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_128: "f32[512]" = torch.ops.prims.convert_element_type.default(arg331_1, torch.float32);  arg331_1 = None
    convert_element_type_129: "f32[512]" = torch.ops.prims.convert_element_type.default(arg332_1, torch.float32);  arg332_1 = None
    add_150: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_129, 1e-05);  convert_element_type_129 = None
    sqrt_64: "f32[512]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
    reciprocal_64: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
    mul_256: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
    unsqueeze_512: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_128, -1);  convert_element_type_128 = None
    unsqueeze_513: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    unsqueeze_514: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_256, -1);  mul_256 = None
    unsqueeze_515: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    sub_64: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_513);  convolution_64 = unsqueeze_513 = None
    mul_257: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
    unsqueeze_516: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
    unsqueeze_517: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_258: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_257, unsqueeze_517);  mul_257 = unsqueeze_517 = None
    unsqueeze_518: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
    unsqueeze_519: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_151: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_519);  mul_258 = unsqueeze_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_64: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_151, 0)
    mul_259: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_151, 0.01)
    where_64: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_64, add_151, mul_259);  gt_64 = add_151 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:222, code: x = self.drop_path(x) + shortcut
    add_152: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(where_64, add_147);  where_64 = add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_65: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(add_152, arg199_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_152 = arg199_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_130: "f32[512]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.float32);  arg333_1 = None
    convert_element_type_131: "f32[512]" = torch.ops.prims.convert_element_type.default(arg334_1, torch.float32);  arg334_1 = None
    add_153: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_131, 1e-05);  convert_element_type_131 = None
    sqrt_65: "f32[512]" = torch.ops.aten.sqrt.default(add_153);  add_153 = None
    reciprocal_65: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
    mul_260: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
    unsqueeze_520: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_130, -1);  convert_element_type_130 = None
    unsqueeze_521: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    unsqueeze_522: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_260, -1);  mul_260 = None
    unsqueeze_523: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    sub_65: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_521);  convolution_65 = unsqueeze_521 = None
    mul_261: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
    unsqueeze_524: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
    unsqueeze_525: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_262: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_261, unsqueeze_525);  mul_261 = unsqueeze_525 = None
    unsqueeze_526: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
    unsqueeze_527: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_154: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_527);  mul_262 = unsqueeze_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_65: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(add_154, 0)
    mul_263: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_154, 0.01)
    where_65: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_65, add_154, mul_263);  gt_65 = add_154 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(where_56, [512, 512], 1);  where_56 = None
    getitem_28: "f32[8, 512, 8, 8]" = split_with_sizes_14[0];  split_with_sizes_14 = None
    cat_4: "f32[8, 1024, 8, 8]" = torch.ops.aten.cat.default([getitem_28, where_65], 1);  getitem_28 = where_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_66: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(cat_4, arg200_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_132: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg335_1, torch.float32);  arg335_1 = None
    convert_element_type_133: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg336_1, torch.float32);  arg336_1 = None
    add_155: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_133, 1e-05);  convert_element_type_133 = None
    sqrt_66: "f32[1024]" = torch.ops.aten.sqrt.default(add_155);  add_155 = None
    reciprocal_66: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
    mul_264: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
    unsqueeze_528: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_132, -1);  convert_element_type_132 = None
    unsqueeze_529: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    unsqueeze_530: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
    unsqueeze_531: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    sub_66: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_529);  convolution_66 = unsqueeze_529 = None
    mul_265: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
    unsqueeze_532: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
    unsqueeze_533: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_266: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_533);  mul_265 = unsqueeze_533 = None
    unsqueeze_534: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
    unsqueeze_535: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_156: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_535);  mul_266 = unsqueeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    gt_66: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(add_156, 0)
    mul_267: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(add_156, 0.01)
    where_66: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_66, add_156, mul_267);  gt_66 = add_156 = mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(where_66, [-1, -2], True);  where_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1024]" = torch.ops.aten.view.default(mean, [8, 1024]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone: "f32[8, 1024]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg202_1, clone, permute);  arg202_1 = clone = permute = None
    return (addmm,)
    