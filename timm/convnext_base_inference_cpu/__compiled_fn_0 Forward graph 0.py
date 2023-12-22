from __future__ import annotations



def forward(self, arg0_1: "f32[128]", arg1_1: "f32[128]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[128]", arg9_1: "f32[128]", arg10_1: "f32[128]", arg11_1: "f32[128]", arg12_1: "f32[128]", arg13_1: "f32[256]", arg14_1: "f32[256]", arg15_1: "f32[256]", arg16_1: "f32[256]", arg17_1: "f32[256]", arg18_1: "f32[256]", arg19_1: "f32[256]", arg20_1: "f32[256]", arg21_1: "f32[256]", arg22_1: "f32[256]", arg23_1: "f32[256]", arg24_1: "f32[512]", arg25_1: "f32[512]", arg26_1: "f32[512]", arg27_1: "f32[512]", arg28_1: "f32[512]", arg29_1: "f32[512]", arg30_1: "f32[512]", arg31_1: "f32[512]", arg32_1: "f32[512]", arg33_1: "f32[512]", arg34_1: "f32[512]", arg35_1: "f32[512]", arg36_1: "f32[512]", arg37_1: "f32[512]", arg38_1: "f32[512]", arg39_1: "f32[512]", arg40_1: "f32[512]", arg41_1: "f32[512]", arg42_1: "f32[512]", arg43_1: "f32[512]", arg44_1: "f32[512]", arg45_1: "f32[512]", arg46_1: "f32[512]", arg47_1: "f32[512]", arg48_1: "f32[512]", arg49_1: "f32[512]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[512]", arg53_1: "f32[512]", arg54_1: "f32[512]", arg55_1: "f32[512]", arg56_1: "f32[512]", arg57_1: "f32[512]", arg58_1: "f32[512]", arg59_1: "f32[512]", arg60_1: "f32[512]", arg61_1: "f32[512]", arg62_1: "f32[512]", arg63_1: "f32[512]", arg64_1: "f32[512]", arg65_1: "f32[512]", arg66_1: "f32[512]", arg67_1: "f32[512]", arg68_1: "f32[512]", arg69_1: "f32[512]", arg70_1: "f32[512]", arg71_1: "f32[512]", arg72_1: "f32[512]", arg73_1: "f32[512]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[512]", arg77_1: "f32[512]", arg78_1: "f32[512]", arg79_1: "f32[512]", arg80_1: "f32[512]", arg81_1: "f32[512]", arg82_1: "f32[512]", arg83_1: "f32[512]", arg84_1: "f32[512]", arg85_1: "f32[512]", arg86_1: "f32[512]", arg87_1: "f32[512]", arg88_1: "f32[512]", arg89_1: "f32[512]", arg90_1: "f32[512]", arg91_1: "f32[512]", arg92_1: "f32[512]", arg93_1: "f32[512]", arg94_1: "f32[512]", arg95_1: "f32[512]", arg96_1: "f32[512]", arg97_1: "f32[512]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[512]", arg101_1: "f32[512]", arg102_1: "f32[512]", arg103_1: "f32[512]", arg104_1: "f32[512]", arg105_1: "f32[512]", arg106_1: "f32[512]", arg107_1: "f32[1024]", arg108_1: "f32[1024]", arg109_1: "f32[1024]", arg110_1: "f32[1024]", arg111_1: "f32[1024]", arg112_1: "f32[1024]", arg113_1: "f32[1024]", arg114_1: "f32[1024]", arg115_1: "f32[1024]", arg116_1: "f32[1024]", arg117_1: "f32[1024]", arg118_1: "f32[128, 3, 4, 4]", arg119_1: "f32[128]", arg120_1: "f32[128, 1, 7, 7]", arg121_1: "f32[128]", arg122_1: "f32[512, 128]", arg123_1: "f32[512]", arg124_1: "f32[128, 512]", arg125_1: "f32[128]", arg126_1: "f32[128, 1, 7, 7]", arg127_1: "f32[128]", arg128_1: "f32[512, 128]", arg129_1: "f32[512]", arg130_1: "f32[128, 512]", arg131_1: "f32[128]", arg132_1: "f32[128, 1, 7, 7]", arg133_1: "f32[128]", arg134_1: "f32[512, 128]", arg135_1: "f32[512]", arg136_1: "f32[128, 512]", arg137_1: "f32[128]", arg138_1: "f32[256, 128, 2, 2]", arg139_1: "f32[256]", arg140_1: "f32[256, 1, 7, 7]", arg141_1: "f32[256]", arg142_1: "f32[1024, 256]", arg143_1: "f32[1024]", arg144_1: "f32[256, 1024]", arg145_1: "f32[256]", arg146_1: "f32[256, 1, 7, 7]", arg147_1: "f32[256]", arg148_1: "f32[1024, 256]", arg149_1: "f32[1024]", arg150_1: "f32[256, 1024]", arg151_1: "f32[256]", arg152_1: "f32[256, 1, 7, 7]", arg153_1: "f32[256]", arg154_1: "f32[1024, 256]", arg155_1: "f32[1024]", arg156_1: "f32[256, 1024]", arg157_1: "f32[256]", arg158_1: "f32[512, 256, 2, 2]", arg159_1: "f32[512]", arg160_1: "f32[512, 1, 7, 7]", arg161_1: "f32[512]", arg162_1: "f32[2048, 512]", arg163_1: "f32[2048]", arg164_1: "f32[512, 2048]", arg165_1: "f32[512]", arg166_1: "f32[512, 1, 7, 7]", arg167_1: "f32[512]", arg168_1: "f32[2048, 512]", arg169_1: "f32[2048]", arg170_1: "f32[512, 2048]", arg171_1: "f32[512]", arg172_1: "f32[512, 1, 7, 7]", arg173_1: "f32[512]", arg174_1: "f32[2048, 512]", arg175_1: "f32[2048]", arg176_1: "f32[512, 2048]", arg177_1: "f32[512]", arg178_1: "f32[512, 1, 7, 7]", arg179_1: "f32[512]", arg180_1: "f32[2048, 512]", arg181_1: "f32[2048]", arg182_1: "f32[512, 2048]", arg183_1: "f32[512]", arg184_1: "f32[512, 1, 7, 7]", arg185_1: "f32[512]", arg186_1: "f32[2048, 512]", arg187_1: "f32[2048]", arg188_1: "f32[512, 2048]", arg189_1: "f32[512]", arg190_1: "f32[512, 1, 7, 7]", arg191_1: "f32[512]", arg192_1: "f32[2048, 512]", arg193_1: "f32[2048]", arg194_1: "f32[512, 2048]", arg195_1: "f32[512]", arg196_1: "f32[512, 1, 7, 7]", arg197_1: "f32[512]", arg198_1: "f32[2048, 512]", arg199_1: "f32[2048]", arg200_1: "f32[512, 2048]", arg201_1: "f32[512]", arg202_1: "f32[512, 1, 7, 7]", arg203_1: "f32[512]", arg204_1: "f32[2048, 512]", arg205_1: "f32[2048]", arg206_1: "f32[512, 2048]", arg207_1: "f32[512]", arg208_1: "f32[512, 1, 7, 7]", arg209_1: "f32[512]", arg210_1: "f32[2048, 512]", arg211_1: "f32[2048]", arg212_1: "f32[512, 2048]", arg213_1: "f32[512]", arg214_1: "f32[512, 1, 7, 7]", arg215_1: "f32[512]", arg216_1: "f32[2048, 512]", arg217_1: "f32[2048]", arg218_1: "f32[512, 2048]", arg219_1: "f32[512]", arg220_1: "f32[512, 1, 7, 7]", arg221_1: "f32[512]", arg222_1: "f32[2048, 512]", arg223_1: "f32[2048]", arg224_1: "f32[512, 2048]", arg225_1: "f32[512]", arg226_1: "f32[512, 1, 7, 7]", arg227_1: "f32[512]", arg228_1: "f32[2048, 512]", arg229_1: "f32[2048]", arg230_1: "f32[512, 2048]", arg231_1: "f32[512]", arg232_1: "f32[512, 1, 7, 7]", arg233_1: "f32[512]", arg234_1: "f32[2048, 512]", arg235_1: "f32[2048]", arg236_1: "f32[512, 2048]", arg237_1: "f32[512]", arg238_1: "f32[512, 1, 7, 7]", arg239_1: "f32[512]", arg240_1: "f32[2048, 512]", arg241_1: "f32[2048]", arg242_1: "f32[512, 2048]", arg243_1: "f32[512]", arg244_1: "f32[512, 1, 7, 7]", arg245_1: "f32[512]", arg246_1: "f32[2048, 512]", arg247_1: "f32[2048]", arg248_1: "f32[512, 2048]", arg249_1: "f32[512]", arg250_1: "f32[512, 1, 7, 7]", arg251_1: "f32[512]", arg252_1: "f32[2048, 512]", arg253_1: "f32[2048]", arg254_1: "f32[512, 2048]", arg255_1: "f32[512]", arg256_1: "f32[512, 1, 7, 7]", arg257_1: "f32[512]", arg258_1: "f32[2048, 512]", arg259_1: "f32[2048]", arg260_1: "f32[512, 2048]", arg261_1: "f32[512]", arg262_1: "f32[512, 1, 7, 7]", arg263_1: "f32[512]", arg264_1: "f32[2048, 512]", arg265_1: "f32[2048]", arg266_1: "f32[512, 2048]", arg267_1: "f32[512]", arg268_1: "f32[512, 1, 7, 7]", arg269_1: "f32[512]", arg270_1: "f32[2048, 512]", arg271_1: "f32[2048]", arg272_1: "f32[512, 2048]", arg273_1: "f32[512]", arg274_1: "f32[512, 1, 7, 7]", arg275_1: "f32[512]", arg276_1: "f32[2048, 512]", arg277_1: "f32[2048]", arg278_1: "f32[512, 2048]", arg279_1: "f32[512]", arg280_1: "f32[512, 1, 7, 7]", arg281_1: "f32[512]", arg282_1: "f32[2048, 512]", arg283_1: "f32[2048]", arg284_1: "f32[512, 2048]", arg285_1: "f32[512]", arg286_1: "f32[512, 1, 7, 7]", arg287_1: "f32[512]", arg288_1: "f32[2048, 512]", arg289_1: "f32[2048]", arg290_1: "f32[512, 2048]", arg291_1: "f32[512]", arg292_1: "f32[512, 1, 7, 7]", arg293_1: "f32[512]", arg294_1: "f32[2048, 512]", arg295_1: "f32[2048]", arg296_1: "f32[512, 2048]", arg297_1: "f32[512]", arg298_1: "f32[512, 1, 7, 7]", arg299_1: "f32[512]", arg300_1: "f32[2048, 512]", arg301_1: "f32[2048]", arg302_1: "f32[512, 2048]", arg303_1: "f32[512]", arg304_1: "f32[512, 1, 7, 7]", arg305_1: "f32[512]", arg306_1: "f32[2048, 512]", arg307_1: "f32[2048]", arg308_1: "f32[512, 2048]", arg309_1: "f32[512]", arg310_1: "f32[512, 1, 7, 7]", arg311_1: "f32[512]", arg312_1: "f32[2048, 512]", arg313_1: "f32[2048]", arg314_1: "f32[512, 2048]", arg315_1: "f32[512]", arg316_1: "f32[512, 1, 7, 7]", arg317_1: "f32[512]", arg318_1: "f32[2048, 512]", arg319_1: "f32[2048]", arg320_1: "f32[512, 2048]", arg321_1: "f32[512]", arg322_1: "f32[1024, 512, 2, 2]", arg323_1: "f32[1024]", arg324_1: "f32[1024, 1, 7, 7]", arg325_1: "f32[1024]", arg326_1: "f32[4096, 1024]", arg327_1: "f32[4096]", arg328_1: "f32[1024, 4096]", arg329_1: "f32[1024]", arg330_1: "f32[1024, 1, 7, 7]", arg331_1: "f32[1024]", arg332_1: "f32[4096, 1024]", arg333_1: "f32[4096]", arg334_1: "f32[1024, 4096]", arg335_1: "f32[1024]", arg336_1: "f32[1024, 1, 7, 7]", arg337_1: "f32[1024]", arg338_1: "f32[4096, 1024]", arg339_1: "f32[4096]", arg340_1: "f32[1024, 4096]", arg341_1: "f32[1024]", arg342_1: "f32[1000, 1024]", arg343_1: "f32[1000]", arg344_1: "f32[8, 3, 288, 288]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:411, code: x = self.stem(x)
    convolution: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(arg344_1, arg118_1, arg119_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  arg344_1 = arg118_1 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    clone: "f32[8, 72, 72, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    var_mean = torch.ops.aten.var_mean.correction(clone, [3], correction = 0, keepdim = True)
    getitem: "f32[8, 72, 72, 1]" = var_mean[0]
    getitem_1: "f32[8, 72, 72, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = arg0_1 = None
    add_1: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_1: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(add_1, [0, 3, 1, 2]);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(permute_1, arg120_1, arg121_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  arg120_1 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_2: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(permute_2, [3], correction = 0, keepdim = True)
    getitem_2: "f32[8, 72, 72, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 72, 72, 1]" = var_mean_1[1];  var_mean_1 = None
    add_2: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(permute_2, getitem_3);  permute_2 = getitem_3 = None
    mul_2: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_3: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul_2, arg2_1);  mul_2 = arg2_1 = None
    add_3: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view: "f32[41472, 128]" = torch.ops.aten.view.default(add_3, [41472, 128]);  add_3 = None
    permute_3: "f32[128, 512]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm: "f32[41472, 512]" = torch.ops.aten.addmm.default(arg123_1, view, permute_3);  arg123_1 = view = permute_3 = None
    view_1: "f32[8, 72, 72, 512]" = torch.ops.aten.view.default(addmm, [8, 72, 72, 512]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_4: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
    mul_5: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476);  view_1 = None
    erf: "f32[8, 72, 72, 512]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_4: "f32[8, 72, 72, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_1: "f32[8, 72, 72, 512]" = torch.ops.aten.clone.default(mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_2: "f32[41472, 512]" = torch.ops.aten.view.default(clone_1, [41472, 512]);  clone_1 = None
    permute_4: "f32[512, 128]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_1: "f32[41472, 128]" = torch.ops.aten.addmm.default(arg125_1, view_2, permute_4);  arg125_1 = view_2 = permute_4 = None
    view_3: "f32[8, 72, 72, 128]" = torch.ops.aten.view.default(addmm_1, [8, 72, 72, 128]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_2: "f32[8, 72, 72, 128]" = torch.ops.aten.clone.default(view_3);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_5: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(clone_2, [0, 3, 1, 2]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_4: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(arg4_1, [1, -1, 1, 1]);  arg4_1 = None
    mul_7: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(permute_5, view_4);  permute_5 = view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_5: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_7, permute_1);  mul_7 = permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_2: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(add_5, arg126_1, arg127_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  arg126_1 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_6: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(convolution_2, [0, 2, 3, 1]);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(permute_6, [3], correction = 0, keepdim = True)
    getitem_4: "f32[8, 72, 72, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 72, 72, 1]" = var_mean_2[1];  var_mean_2 = None
    add_6: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_2: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(permute_6, getitem_5);  permute_6 = getitem_5 = None
    mul_8: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_9: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul_8, arg5_1);  mul_8 = arg5_1 = None
    add_7: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_9, arg6_1);  mul_9 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_5: "f32[41472, 128]" = torch.ops.aten.view.default(add_7, [41472, 128]);  add_7 = None
    permute_7: "f32[128, 512]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_2: "f32[41472, 512]" = torch.ops.aten.addmm.default(arg129_1, view_5, permute_7);  arg129_1 = view_5 = permute_7 = None
    view_6: "f32[8, 72, 72, 512]" = torch.ops.aten.view.default(addmm_2, [8, 72, 72, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_10: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_6, 0.5)
    mul_11: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_6, 0.7071067811865476);  view_6 = None
    erf_1: "f32[8, 72, 72, 512]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_8: "f32[8, 72, 72, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_12: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(mul_10, add_8);  mul_10 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_3: "f32[8, 72, 72, 512]" = torch.ops.aten.clone.default(mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_7: "f32[41472, 512]" = torch.ops.aten.view.default(clone_3, [41472, 512]);  clone_3 = None
    permute_8: "f32[512, 128]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_3: "f32[41472, 128]" = torch.ops.aten.addmm.default(arg131_1, view_7, permute_8);  arg131_1 = view_7 = permute_8 = None
    view_8: "f32[8, 72, 72, 128]" = torch.ops.aten.view.default(addmm_3, [8, 72, 72, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_4: "f32[8, 72, 72, 128]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_9: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(clone_4, [0, 3, 1, 2]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_9: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(arg7_1, [1, -1, 1, 1]);  arg7_1 = None
    mul_13: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(permute_9, view_9);  permute_9 = view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_9: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_13, add_5);  mul_13 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_3: "f32[8, 128, 72, 72]" = torch.ops.aten.convolution.default(add_9, arg132_1, arg133_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 128);  arg132_1 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_10: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(permute_10, [3], correction = 0, keepdim = True)
    getitem_6: "f32[8, 72, 72, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 72, 72, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(permute_10, getitem_7);  permute_10 = getitem_7 = None
    mul_14: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    mul_15: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul_14, arg8_1);  mul_14 = arg8_1 = None
    add_11: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_15, arg9_1);  mul_15 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_10: "f32[41472, 128]" = torch.ops.aten.view.default(add_11, [41472, 128]);  add_11 = None
    permute_11: "f32[128, 512]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_4: "f32[41472, 512]" = torch.ops.aten.addmm.default(arg135_1, view_10, permute_11);  arg135_1 = view_10 = permute_11 = None
    view_11: "f32[8, 72, 72, 512]" = torch.ops.aten.view.default(addmm_4, [8, 72, 72, 512]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_16: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
    mul_17: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(view_11, 0.7071067811865476);  view_11 = None
    erf_2: "f32[8, 72, 72, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_12: "f32[8, 72, 72, 512]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_18: "f32[8, 72, 72, 512]" = torch.ops.aten.mul.Tensor(mul_16, add_12);  mul_16 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_5: "f32[8, 72, 72, 512]" = torch.ops.aten.clone.default(mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_12: "f32[41472, 512]" = torch.ops.aten.view.default(clone_5, [41472, 512]);  clone_5 = None
    permute_12: "f32[512, 128]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_5: "f32[41472, 128]" = torch.ops.aten.addmm.default(arg137_1, view_12, permute_12);  arg137_1 = view_12 = permute_12 = None
    view_13: "f32[8, 72, 72, 128]" = torch.ops.aten.view.default(addmm_5, [8, 72, 72, 128]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_6: "f32[8, 72, 72, 128]" = torch.ops.aten.clone.default(view_13);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_13: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(clone_6, [0, 3, 1, 2]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_14: "f32[1, 128, 1, 1]" = torch.ops.aten.view.default(arg10_1, [1, -1, 1, 1]);  arg10_1 = None
    mul_19: "f32[8, 128, 72, 72]" = torch.ops.aten.mul.Tensor(permute_13, view_14);  permute_13 = view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_13: "f32[8, 128, 72, 72]" = torch.ops.aten.add.Tensor(mul_19, add_9);  mul_19 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_14: "f32[8, 72, 72, 128]" = torch.ops.aten.permute.default(add_13, [0, 2, 3, 1]);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(permute_14, [3], correction = 0, keepdim = True)
    getitem_8: "f32[8, 72, 72, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 72, 72, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 72, 72, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 72, 72, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 72, 72, 128]" = torch.ops.aten.sub.Tensor(permute_14, getitem_9);  permute_14 = getitem_9 = None
    mul_20: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    mul_21: "f32[8, 72, 72, 128]" = torch.ops.aten.mul.Tensor(mul_20, arg11_1);  mul_20 = arg11_1 = None
    add_15: "f32[8, 72, 72, 128]" = torch.ops.aten.add.Tensor(mul_21, arg12_1);  mul_21 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_15: "f32[8, 128, 72, 72]" = torch.ops.aten.permute.default(add_15, [0, 3, 1, 2]);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_4: "f32[8, 256, 36, 36]" = torch.ops.aten.convolution.default(permute_15, arg138_1, arg139_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_15 = arg138_1 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_5: "f32[8, 256, 36, 36]" = torch.ops.aten.convolution.default(convolution_4, arg140_1, arg141_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  arg140_1 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_16: "f32[8, 36, 36, 256]" = torch.ops.aten.permute.default(convolution_5, [0, 2, 3, 1]);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_5 = torch.ops.aten.var_mean.correction(permute_16, [3], correction = 0, keepdim = True)
    getitem_10: "f32[8, 36, 36, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 36, 36, 1]" = var_mean_5[1];  var_mean_5 = None
    add_16: "f32[8, 36, 36, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 36, 36, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_5: "f32[8, 36, 36, 256]" = torch.ops.aten.sub.Tensor(permute_16, getitem_11);  permute_16 = getitem_11 = None
    mul_22: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    mul_23: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(mul_22, arg13_1);  mul_22 = arg13_1 = None
    add_17: "f32[8, 36, 36, 256]" = torch.ops.aten.add.Tensor(mul_23, arg14_1);  mul_23 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_15: "f32[10368, 256]" = torch.ops.aten.view.default(add_17, [10368, 256]);  add_17 = None
    permute_17: "f32[256, 1024]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_6: "f32[10368, 1024]" = torch.ops.aten.addmm.default(arg143_1, view_15, permute_17);  arg143_1 = view_15 = permute_17 = None
    view_16: "f32[8, 36, 36, 1024]" = torch.ops.aten.view.default(addmm_6, [8, 36, 36, 1024]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_24: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
    mul_25: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476);  view_16 = None
    erf_3: "f32[8, 36, 36, 1024]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_18: "f32[8, 36, 36, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_26: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(mul_24, add_18);  mul_24 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_7: "f32[8, 36, 36, 1024]" = torch.ops.aten.clone.default(mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[10368, 1024]" = torch.ops.aten.view.default(clone_7, [10368, 1024]);  clone_7 = None
    permute_18: "f32[1024, 256]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_7: "f32[10368, 256]" = torch.ops.aten.addmm.default(arg145_1, view_17, permute_18);  arg145_1 = view_17 = permute_18 = None
    view_18: "f32[8, 36, 36, 256]" = torch.ops.aten.view.default(addmm_7, [8, 36, 36, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_8: "f32[8, 36, 36, 256]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_19: "f32[8, 256, 36, 36]" = torch.ops.aten.permute.default(clone_8, [0, 3, 1, 2]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_19: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(arg15_1, [1, -1, 1, 1]);  arg15_1 = None
    mul_27: "f32[8, 256, 36, 36]" = torch.ops.aten.mul.Tensor(permute_19, view_19);  permute_19 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_19: "f32[8, 256, 36, 36]" = torch.ops.aten.add.Tensor(mul_27, convolution_4);  mul_27 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_6: "f32[8, 256, 36, 36]" = torch.ops.aten.convolution.default(add_19, arg146_1, arg147_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  arg146_1 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_20: "f32[8, 36, 36, 256]" = torch.ops.aten.permute.default(convolution_6, [0, 2, 3, 1]);  convolution_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(permute_20, [3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 36, 36, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 36, 36, 1]" = var_mean_6[1];  var_mean_6 = None
    add_20: "f32[8, 36, 36, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 36, 36, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_6: "f32[8, 36, 36, 256]" = torch.ops.aten.sub.Tensor(permute_20, getitem_13);  permute_20 = getitem_13 = None
    mul_28: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    mul_29: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(mul_28, arg16_1);  mul_28 = arg16_1 = None
    add_21: "f32[8, 36, 36, 256]" = torch.ops.aten.add.Tensor(mul_29, arg17_1);  mul_29 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_20: "f32[10368, 256]" = torch.ops.aten.view.default(add_21, [10368, 256]);  add_21 = None
    permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_8: "f32[10368, 1024]" = torch.ops.aten.addmm.default(arg149_1, view_20, permute_21);  arg149_1 = view_20 = permute_21 = None
    view_21: "f32[8, 36, 36, 1024]" = torch.ops.aten.view.default(addmm_8, [8, 36, 36, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_30: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_31: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
    erf_4: "f32[8, 36, 36, 1024]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_22: "f32[8, 36, 36, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_32: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(mul_30, add_22);  mul_30 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_9: "f32[8, 36, 36, 1024]" = torch.ops.aten.clone.default(mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_22: "f32[10368, 1024]" = torch.ops.aten.view.default(clone_9, [10368, 1024]);  clone_9 = None
    permute_22: "f32[1024, 256]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_9: "f32[10368, 256]" = torch.ops.aten.addmm.default(arg151_1, view_22, permute_22);  arg151_1 = view_22 = permute_22 = None
    view_23: "f32[8, 36, 36, 256]" = torch.ops.aten.view.default(addmm_9, [8, 36, 36, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_10: "f32[8, 36, 36, 256]" = torch.ops.aten.clone.default(view_23);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_23: "f32[8, 256, 36, 36]" = torch.ops.aten.permute.default(clone_10, [0, 3, 1, 2]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_24: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(arg18_1, [1, -1, 1, 1]);  arg18_1 = None
    mul_33: "f32[8, 256, 36, 36]" = torch.ops.aten.mul.Tensor(permute_23, view_24);  permute_23 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_23: "f32[8, 256, 36, 36]" = torch.ops.aten.add.Tensor(mul_33, add_19);  mul_33 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_7: "f32[8, 256, 36, 36]" = torch.ops.aten.convolution.default(add_23, arg152_1, arg153_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 256);  arg152_1 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_24: "f32[8, 36, 36, 256]" = torch.ops.aten.permute.default(convolution_7, [0, 2, 3, 1]);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(permute_24, [3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 36, 36, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 36, 36, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 36, 36, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 36, 36, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_7: "f32[8, 36, 36, 256]" = torch.ops.aten.sub.Tensor(permute_24, getitem_15);  permute_24 = getitem_15 = None
    mul_34: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    mul_35: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(mul_34, arg19_1);  mul_34 = arg19_1 = None
    add_25: "f32[8, 36, 36, 256]" = torch.ops.aten.add.Tensor(mul_35, arg20_1);  mul_35 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_25: "f32[10368, 256]" = torch.ops.aten.view.default(add_25, [10368, 256]);  add_25 = None
    permute_25: "f32[256, 1024]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_10: "f32[10368, 1024]" = torch.ops.aten.addmm.default(arg155_1, view_25, permute_25);  arg155_1 = view_25 = permute_25 = None
    view_26: "f32[8, 36, 36, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 36, 36, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_36: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_26, 0.5)
    mul_37: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476);  view_26 = None
    erf_5: "f32[8, 36, 36, 1024]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
    add_26: "f32[8, 36, 36, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_38: "f32[8, 36, 36, 1024]" = torch.ops.aten.mul.Tensor(mul_36, add_26);  mul_36 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_11: "f32[8, 36, 36, 1024]" = torch.ops.aten.clone.default(mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_27: "f32[10368, 1024]" = torch.ops.aten.view.default(clone_11, [10368, 1024]);  clone_11 = None
    permute_26: "f32[1024, 256]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_11: "f32[10368, 256]" = torch.ops.aten.addmm.default(arg157_1, view_27, permute_26);  arg157_1 = view_27 = permute_26 = None
    view_28: "f32[8, 36, 36, 256]" = torch.ops.aten.view.default(addmm_11, [8, 36, 36, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_12: "f32[8, 36, 36, 256]" = torch.ops.aten.clone.default(view_28);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_27: "f32[8, 256, 36, 36]" = torch.ops.aten.permute.default(clone_12, [0, 3, 1, 2]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_29: "f32[1, 256, 1, 1]" = torch.ops.aten.view.default(arg21_1, [1, -1, 1, 1]);  arg21_1 = None
    mul_39: "f32[8, 256, 36, 36]" = torch.ops.aten.mul.Tensor(permute_27, view_29);  permute_27 = view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_27: "f32[8, 256, 36, 36]" = torch.ops.aten.add.Tensor(mul_39, add_23);  mul_39 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_28: "f32[8, 36, 36, 256]" = torch.ops.aten.permute.default(add_27, [0, 2, 3, 1]);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(permute_28, [3], correction = 0, keepdim = True)
    getitem_16: "f32[8, 36, 36, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 36, 36, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 36, 36, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 36, 36, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 36, 36, 256]" = torch.ops.aten.sub.Tensor(permute_28, getitem_17);  permute_28 = getitem_17 = None
    mul_40: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    mul_41: "f32[8, 36, 36, 256]" = torch.ops.aten.mul.Tensor(mul_40, arg22_1);  mul_40 = arg22_1 = None
    add_29: "f32[8, 36, 36, 256]" = torch.ops.aten.add.Tensor(mul_41, arg23_1);  mul_41 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_29: "f32[8, 256, 36, 36]" = torch.ops.aten.permute.default(add_29, [0, 3, 1, 2]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_8: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(permute_29, arg158_1, arg159_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_29 = arg158_1 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_9: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(convolution_8, arg160_1, arg161_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg160_1 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_30: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_9, [0, 2, 3, 1]);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_9 = torch.ops.aten.var_mean.correction(permute_30, [3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 18, 18, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 18, 18, 1]" = var_mean_9[1];  var_mean_9 = None
    add_30: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_9: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_30, getitem_19);  permute_30 = getitem_19 = None
    mul_42: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    mul_43: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_42, arg24_1);  mul_42 = arg24_1 = None
    add_31: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_43, arg25_1);  mul_43 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[2592, 512]" = torch.ops.aten.view.default(add_31, [2592, 512]);  add_31 = None
    permute_31: "f32[512, 2048]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    addmm_12: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg163_1, view_30, permute_31);  arg163_1 = view_30 = permute_31 = None
    view_31: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_12, [8, 18, 18, 2048]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_44: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    mul_45: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476);  view_31 = None
    erf_6: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
    add_32: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_46: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_44, add_32);  mul_44 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_13: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_46);  mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_32: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_13, [2592, 2048]);  clone_13 = None
    permute_32: "f32[2048, 512]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_13: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg165_1, view_32, permute_32);  arg165_1 = view_32 = permute_32 = None
    view_33: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_13, [8, 18, 18, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_14: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_33: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_14, [0, 3, 1, 2]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_34: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg26_1, [1, -1, 1, 1]);  arg26_1 = None
    mul_47: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_33, view_34);  permute_33 = view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_33: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_47, convolution_8);  mul_47 = convolution_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_10: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_33, arg166_1, arg167_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg166_1 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_34: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_10, [0, 2, 3, 1]);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_10 = torch.ops.aten.var_mean.correction(permute_34, [3], correction = 0, keepdim = True)
    getitem_20: "f32[8, 18, 18, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 18, 18, 1]" = var_mean_10[1];  var_mean_10 = None
    add_34: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_10: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_34, getitem_21);  permute_34 = getitem_21 = None
    mul_48: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    mul_49: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_48, arg27_1);  mul_48 = arg27_1 = None
    add_35: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_49, arg28_1);  mul_49 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_35: "f32[2592, 512]" = torch.ops.aten.view.default(add_35, [2592, 512]);  add_35 = None
    permute_35: "f32[512, 2048]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_14: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg169_1, view_35, permute_35);  arg169_1 = view_35 = permute_35 = None
    view_36: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_14, [8, 18, 18, 2048]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_50: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_36, 0.5)
    mul_51: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_36, 0.7071067811865476);  view_36 = None
    erf_7: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_36: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_52: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_50, add_36);  mul_50 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_37: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_15, [2592, 2048]);  clone_15 = None
    permute_36: "f32[2048, 512]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_15: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg171_1, view_37, permute_36);  arg171_1 = view_37 = permute_36 = None
    view_38: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_15, [8, 18, 18, 512]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_38);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_37: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_16, [0, 3, 1, 2]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_39: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg29_1, [1, -1, 1, 1]);  arg29_1 = None
    mul_53: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_37, view_39);  permute_37 = view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_37: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_53, add_33);  mul_53 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_11: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_37, arg172_1, arg173_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg172_1 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_38: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(permute_38, [3], correction = 0, keepdim = True)
    getitem_22: "f32[8, 18, 18, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 18, 18, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_11: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_38, getitem_23);  permute_38 = getitem_23 = None
    mul_54: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    mul_55: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_54, arg30_1);  mul_54 = arg30_1 = None
    add_39: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_55, arg31_1);  mul_55 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_40: "f32[2592, 512]" = torch.ops.aten.view.default(add_39, [2592, 512]);  add_39 = None
    permute_39: "f32[512, 2048]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_16: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg175_1, view_40, permute_39);  arg175_1 = view_40 = permute_39 = None
    view_41: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_16, [8, 18, 18, 2048]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_56: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_57: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_8: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
    add_40: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_58: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_56, add_40);  mul_56 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_17: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_58);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_42: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_17, [2592, 2048]);  clone_17 = None
    permute_40: "f32[2048, 512]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_17: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg177_1, view_42, permute_40);  arg177_1 = view_42 = permute_40 = None
    view_43: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_17, [8, 18, 18, 512]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_18: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_43);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_41: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_18, [0, 3, 1, 2]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_44: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg32_1, [1, -1, 1, 1]);  arg32_1 = None
    mul_59: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_41, view_44);  permute_41 = view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_41: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_59, add_37);  mul_59 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_12: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_41, arg178_1, arg179_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg178_1 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_42: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_12, [0, 2, 3, 1]);  convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(permute_42, [3], correction = 0, keepdim = True)
    getitem_24: "f32[8, 18, 18, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 18, 18, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_42, getitem_25);  permute_42 = getitem_25 = None
    mul_60: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    mul_61: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_60, arg33_1);  mul_60 = arg33_1 = None
    add_43: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_61, arg34_1);  mul_61 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[2592, 512]" = torch.ops.aten.view.default(add_43, [2592, 512]);  add_43 = None
    permute_43: "f32[512, 2048]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_18: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg181_1, view_45, permute_43);  arg181_1 = view_45 = permute_43 = None
    view_46: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 18, 18, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_62: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
    mul_63: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476);  view_46 = None
    erf_9: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_44: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_64: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_62, add_44);  mul_62 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_19: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_19, [2592, 2048]);  clone_19 = None
    permute_44: "f32[2048, 512]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_19: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg183_1, view_47, permute_44);  arg183_1 = view_47 = permute_44 = None
    view_48: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_19, [8, 18, 18, 512]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_20: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_45: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_20, [0, 3, 1, 2]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_49: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg35_1, [1, -1, 1, 1]);  arg35_1 = None
    mul_65: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_45, view_49);  permute_45 = view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_45: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_65, add_41);  mul_65 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_13: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_45, arg184_1, arg185_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg184_1 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_46: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_13, [0, 2, 3, 1]);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(permute_46, [3], correction = 0, keepdim = True)
    getitem_26: "f32[8, 18, 18, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 18, 18, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_13: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_46, getitem_27);  permute_46 = getitem_27 = None
    mul_66: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    mul_67: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_66, arg36_1);  mul_66 = arg36_1 = None
    add_47: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_67, arg37_1);  mul_67 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[2592, 512]" = torch.ops.aten.view.default(add_47, [2592, 512]);  add_47 = None
    permute_47: "f32[512, 2048]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_20: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg187_1, view_50, permute_47);  arg187_1 = view_50 = permute_47 = None
    view_51: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_20, [8, 18, 18, 2048]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_68: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_69: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476);  view_51 = None
    erf_10: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_48: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_70: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_68, add_48);  mul_68 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_21: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_21, [2592, 2048]);  clone_21 = None
    permute_48: "f32[2048, 512]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    addmm_21: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg189_1, view_52, permute_48);  arg189_1 = view_52 = permute_48 = None
    view_53: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_21, [8, 18, 18, 512]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_22: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_53);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_49: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_22, [0, 3, 1, 2]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_54: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg38_1, [1, -1, 1, 1]);  arg38_1 = None
    mul_71: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_49, view_54);  permute_49 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_49: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_71, add_45);  mul_71 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_14: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_49, arg190_1, arg191_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg190_1 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_50: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_14, [0, 2, 3, 1]);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(permute_50, [3], correction = 0, keepdim = True)
    getitem_28: "f32[8, 18, 18, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 18, 18, 1]" = var_mean_14[1];  var_mean_14 = None
    add_50: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_14: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_50, getitem_29);  permute_50 = getitem_29 = None
    mul_72: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    mul_73: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_72, arg39_1);  mul_72 = arg39_1 = None
    add_51: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_73, arg40_1);  mul_73 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_55: "f32[2592, 512]" = torch.ops.aten.view.default(add_51, [2592, 512]);  add_51 = None
    permute_51: "f32[512, 2048]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_22: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg193_1, view_55, permute_51);  arg193_1 = view_55 = permute_51 = None
    view_56: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 18, 18, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_74: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_56, 0.5)
    mul_75: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476);  view_56 = None
    erf_11: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_52: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_76: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_74, add_52);  mul_74 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_57: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_23, [2592, 2048]);  clone_23 = None
    permute_52: "f32[2048, 512]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    addmm_23: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg195_1, view_57, permute_52);  arg195_1 = view_57 = permute_52 = None
    view_58: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_23, [8, 18, 18, 512]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_58);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_53: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_24, [0, 3, 1, 2]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_59: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg41_1, [1, -1, 1, 1]);  arg41_1 = None
    mul_77: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_53, view_59);  permute_53 = view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_53: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_77, add_49);  mul_77 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_53, arg196_1, arg197_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg196_1 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_54: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_15 = torch.ops.aten.var_mean.correction(permute_54, [3], correction = 0, keepdim = True)
    getitem_30: "f32[8, 18, 18, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 18, 18, 1]" = var_mean_15[1];  var_mean_15 = None
    add_54: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_15: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_54, getitem_31);  permute_54 = getitem_31 = None
    mul_78: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    mul_79: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_78, arg42_1);  mul_78 = arg42_1 = None
    add_55: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_79, arg43_1);  mul_79 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_60: "f32[2592, 512]" = torch.ops.aten.view.default(add_55, [2592, 512]);  add_55 = None
    permute_55: "f32[512, 2048]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    addmm_24: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg199_1, view_60, permute_55);  arg199_1 = view_60 = permute_55 = None
    view_61: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_24, [8, 18, 18, 2048]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_80: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
    mul_81: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
    erf_12: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_56: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_82: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_80, add_56);  mul_80 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_25: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_82);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_62: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_25, [2592, 2048]);  clone_25 = None
    permute_56: "f32[2048, 512]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
    addmm_25: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg201_1, view_62, permute_56);  arg201_1 = view_62 = permute_56 = None
    view_63: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_25, [8, 18, 18, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_26: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_63);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_57: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_26, [0, 3, 1, 2]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_64: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg44_1, [1, -1, 1, 1]);  arg44_1 = None
    mul_83: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_57, view_64);  permute_57 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_57: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_83, add_53);  mul_83 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_16: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_57, arg202_1, arg203_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg202_1 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_58: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_16, [0, 2, 3, 1]);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(permute_58, [3], correction = 0, keepdim = True)
    getitem_32: "f32[8, 18, 18, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 18, 18, 1]" = var_mean_16[1];  var_mean_16 = None
    add_58: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_16: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_58, getitem_33);  permute_58 = getitem_33 = None
    mul_84: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    mul_85: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_84, arg45_1);  mul_84 = arg45_1 = None
    add_59: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_85, arg46_1);  mul_85 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_65: "f32[2592, 512]" = torch.ops.aten.view.default(add_59, [2592, 512]);  add_59 = None
    permute_59: "f32[512, 2048]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    addmm_26: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg205_1, view_65, permute_59);  arg205_1 = view_65 = permute_59 = None
    view_66: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 18, 18, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_86: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
    mul_87: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476);  view_66 = None
    erf_13: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_60: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_88: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_86, add_60);  mul_86 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_27: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_67: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_27, [2592, 2048]);  clone_27 = None
    permute_60: "f32[2048, 512]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    addmm_27: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg207_1, view_67, permute_60);  arg207_1 = view_67 = permute_60 = None
    view_68: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_27, [8, 18, 18, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_28: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_61: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_28, [0, 3, 1, 2]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_69: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg47_1, [1, -1, 1, 1]);  arg47_1 = None
    mul_89: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_61, view_69);  permute_61 = view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_61: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_89, add_57);  mul_89 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_17: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_61, arg208_1, arg209_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg208_1 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_62: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_17, [0, 2, 3, 1]);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(permute_62, [3], correction = 0, keepdim = True)
    getitem_34: "f32[8, 18, 18, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 18, 18, 1]" = var_mean_17[1];  var_mean_17 = None
    add_62: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_17: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_62, getitem_35);  permute_62 = getitem_35 = None
    mul_90: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    mul_91: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_90, arg48_1);  mul_90 = arg48_1 = None
    add_63: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_91, arg49_1);  mul_91 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_70: "f32[2592, 512]" = torch.ops.aten.view.default(add_63, [2592, 512]);  add_63 = None
    permute_63: "f32[512, 2048]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    addmm_28: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg211_1, view_70, permute_63);  arg211_1 = view_70 = permute_63 = None
    view_71: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_28, [8, 18, 18, 2048]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_92: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    mul_93: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_71, 0.7071067811865476);  view_71 = None
    erf_14: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
    add_64: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_94: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_92, add_64);  mul_92 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_94);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_72: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_29, [2592, 2048]);  clone_29 = None
    permute_64: "f32[2048, 512]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
    addmm_29: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg213_1, view_72, permute_64);  arg213_1 = view_72 = permute_64 = None
    view_73: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_29, [8, 18, 18, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_73);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_65: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_30, [0, 3, 1, 2]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_74: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg50_1, [1, -1, 1, 1]);  arg50_1 = None
    mul_95: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_65, view_74);  permute_65 = view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_65: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_95, add_61);  mul_95 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_18: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_65, arg214_1, arg215_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg214_1 = arg215_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_66: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1]);  convolution_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(permute_66, [3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 18, 18, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 18, 18, 1]" = var_mean_18[1];  var_mean_18 = None
    add_66: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_18: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_66, getitem_37);  permute_66 = getitem_37 = None
    mul_96: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    mul_97: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_96, arg51_1);  mul_96 = arg51_1 = None
    add_67: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_97, arg52_1);  mul_97 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_75: "f32[2592, 512]" = torch.ops.aten.view.default(add_67, [2592, 512]);  add_67 = None
    permute_67: "f32[512, 2048]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
    addmm_30: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg217_1, view_75, permute_67);  arg217_1 = view_75 = permute_67 = None
    view_76: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 18, 18, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_98: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_76, 0.5)
    mul_99: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476);  view_76 = None
    erf_15: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_68: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_100: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_98, add_68);  mul_98 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_100);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_77: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_31, [2592, 2048]);  clone_31 = None
    permute_68: "f32[2048, 512]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
    addmm_31: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg219_1, view_77, permute_68);  arg219_1 = view_77 = permute_68 = None
    view_78: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_31, [8, 18, 18, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_78);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_69: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_32, [0, 3, 1, 2]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_79: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg53_1, [1, -1, 1, 1]);  arg53_1 = None
    mul_101: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_69, view_79);  permute_69 = view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_69: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_101, add_65);  mul_101 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_19: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_69, arg220_1, arg221_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg220_1 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_70: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_19, [0, 2, 3, 1]);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(permute_70, [3], correction = 0, keepdim = True)
    getitem_38: "f32[8, 18, 18, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 18, 18, 1]" = var_mean_19[1];  var_mean_19 = None
    add_70: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_19: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_70, getitem_39);  permute_70 = getitem_39 = None
    mul_102: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    mul_103: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_102, arg54_1);  mul_102 = arg54_1 = None
    add_71: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_103, arg55_1);  mul_103 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_80: "f32[2592, 512]" = torch.ops.aten.view.default(add_71, [2592, 512]);  add_71 = None
    permute_71: "f32[512, 2048]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
    addmm_32: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg223_1, view_80, permute_71);  arg223_1 = view_80 = permute_71 = None
    view_81: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_32, [8, 18, 18, 2048]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_104: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
    mul_105: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_81, 0.7071067811865476);  view_81 = None
    erf_16: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_72: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_106: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_104, add_72);  mul_104 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_33: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_82: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_33, [2592, 2048]);  clone_33 = None
    permute_72: "f32[2048, 512]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    addmm_33: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg225_1, view_82, permute_72);  arg225_1 = view_82 = permute_72 = None
    view_83: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_33, [8, 18, 18, 512]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_34: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_73: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_34, [0, 3, 1, 2]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_84: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg56_1, [1, -1, 1, 1]);  arg56_1 = None
    mul_107: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_73, view_84);  permute_73 = view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_73: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_107, add_69);  mul_107 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_20: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_73, arg226_1, arg227_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg226_1 = arg227_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_74: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_20, [0, 2, 3, 1]);  convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(permute_74, [3], correction = 0, keepdim = True)
    getitem_40: "f32[8, 18, 18, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 18, 18, 1]" = var_mean_20[1];  var_mean_20 = None
    add_74: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_20: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_74, getitem_41);  permute_74 = getitem_41 = None
    mul_108: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    mul_109: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_108, arg57_1);  mul_108 = arg57_1 = None
    add_75: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_109, arg58_1);  mul_109 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_85: "f32[2592, 512]" = torch.ops.aten.view.default(add_75, [2592, 512]);  add_75 = None
    permute_75: "f32[512, 2048]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    addmm_34: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg229_1, view_85, permute_75);  arg229_1 = view_85 = permute_75 = None
    view_86: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 18, 18, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_110: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_86, 0.5)
    mul_111: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476);  view_86 = None
    erf_17: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_76: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_112: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_110, add_76);  mul_110 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_35: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_112);  mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_87: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_35, [2592, 2048]);  clone_35 = None
    permute_76: "f32[2048, 512]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    addmm_35: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg231_1, view_87, permute_76);  arg231_1 = view_87 = permute_76 = None
    view_88: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_35, [8, 18, 18, 512]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_36: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_88);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_77: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_36, [0, 3, 1, 2]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_89: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg59_1, [1, -1, 1, 1]);  arg59_1 = None
    mul_113: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_77, view_89);  permute_77 = view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_77: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_113, add_73);  mul_113 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_21: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_77, arg232_1, arg233_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg232_1 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_78: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(permute_78, [3], correction = 0, keepdim = True)
    getitem_42: "f32[8, 18, 18, 1]" = var_mean_21[0]
    getitem_43: "f32[8, 18, 18, 1]" = var_mean_21[1];  var_mean_21 = None
    add_78: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_21: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_21: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_78, getitem_43);  permute_78 = getitem_43 = None
    mul_114: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    mul_115: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_114, arg60_1);  mul_114 = arg60_1 = None
    add_79: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_115, arg61_1);  mul_115 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_90: "f32[2592, 512]" = torch.ops.aten.view.default(add_79, [2592, 512]);  add_79 = None
    permute_79: "f32[512, 2048]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    addmm_36: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg235_1, view_90, permute_79);  arg235_1 = view_90 = permute_79 = None
    view_91: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_36, [8, 18, 18, 2048]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_116: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_91, 0.5)
    mul_117: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_91, 0.7071067811865476);  view_91 = None
    erf_18: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
    add_80: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_118: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_116, add_80);  mul_116 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_37: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_118);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_92: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_37, [2592, 2048]);  clone_37 = None
    permute_80: "f32[2048, 512]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    addmm_37: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg237_1, view_92, permute_80);  arg237_1 = view_92 = permute_80 = None
    view_93: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_37, [8, 18, 18, 512]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_38: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_93);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_81: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_38, [0, 3, 1, 2]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_94: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg62_1, [1, -1, 1, 1]);  arg62_1 = None
    mul_119: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_81, view_94);  permute_81 = view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_81: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_119, add_77);  mul_119 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_22: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_81, arg238_1, arg239_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg238_1 = arg239_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_82: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_22, [0, 2, 3, 1]);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(permute_82, [3], correction = 0, keepdim = True)
    getitem_44: "f32[8, 18, 18, 1]" = var_mean_22[0]
    getitem_45: "f32[8, 18, 18, 1]" = var_mean_22[1];  var_mean_22 = None
    add_82: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_22: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_22: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_82, getitem_45);  permute_82 = getitem_45 = None
    mul_120: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    mul_121: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_120, arg63_1);  mul_120 = arg63_1 = None
    add_83: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_121, arg64_1);  mul_121 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_95: "f32[2592, 512]" = torch.ops.aten.view.default(add_83, [2592, 512]);  add_83 = None
    permute_83: "f32[512, 2048]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    addmm_38: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg241_1, view_95, permute_83);  arg241_1 = view_95 = permute_83 = None
    view_96: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_38, [8, 18, 18, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_122: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_96, 0.5)
    mul_123: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_96, 0.7071067811865476);  view_96 = None
    erf_19: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_84: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_124: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_122, add_84);  mul_122 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_97: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_39, [2592, 2048]);  clone_39 = None
    permute_84: "f32[2048, 512]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
    addmm_39: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg243_1, view_97, permute_84);  arg243_1 = view_97 = permute_84 = None
    view_98: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_39, [8, 18, 18, 512]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_98);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_85: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_40, [0, 3, 1, 2]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_99: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg65_1, [1, -1, 1, 1]);  arg65_1 = None
    mul_125: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_85, view_99);  permute_85 = view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_85: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_125, add_81);  mul_125 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_23: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_85, arg244_1, arg245_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg244_1 = arg245_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_86: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_23, [0, 2, 3, 1]);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(permute_86, [3], correction = 0, keepdim = True)
    getitem_46: "f32[8, 18, 18, 1]" = var_mean_23[0]
    getitem_47: "f32[8, 18, 18, 1]" = var_mean_23[1];  var_mean_23 = None
    add_86: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_23: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_23: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_86, getitem_47);  permute_86 = getitem_47 = None
    mul_126: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    mul_127: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_126, arg66_1);  mul_126 = arg66_1 = None
    add_87: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_127, arg67_1);  mul_127 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_100: "f32[2592, 512]" = torch.ops.aten.view.default(add_87, [2592, 512]);  add_87 = None
    permute_87: "f32[512, 2048]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
    addmm_40: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg247_1, view_100, permute_87);  arg247_1 = view_100 = permute_87 = None
    view_101: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_40, [8, 18, 18, 2048]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_128: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_101, 0.5)
    mul_129: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476);  view_101 = None
    erf_20: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_88: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_130: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_128, add_88);  mul_128 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_41: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_130);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_102: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_41, [2592, 2048]);  clone_41 = None
    permute_88: "f32[2048, 512]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
    addmm_41: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg249_1, view_102, permute_88);  arg249_1 = view_102 = permute_88 = None
    view_103: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_41, [8, 18, 18, 512]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_42: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_103);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_89: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_42, [0, 3, 1, 2]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_104: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg68_1, [1, -1, 1, 1]);  arg68_1 = None
    mul_131: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_89, view_104);  permute_89 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_89: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_131, add_85);  mul_131 = add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_24: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_89, arg250_1, arg251_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg250_1 = arg251_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_90: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(permute_90, [3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 18, 18, 1]" = var_mean_24[0]
    getitem_49: "f32[8, 18, 18, 1]" = var_mean_24[1];  var_mean_24 = None
    add_90: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_24: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_24: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_90, getitem_49);  permute_90 = getitem_49 = None
    mul_132: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    mul_133: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_132, arg69_1);  mul_132 = arg69_1 = None
    add_91: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_133, arg70_1);  mul_133 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[2592, 512]" = torch.ops.aten.view.default(add_91, [2592, 512]);  add_91 = None
    permute_91: "f32[512, 2048]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
    addmm_42: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg253_1, view_105, permute_91);  arg253_1 = view_105 = permute_91 = None
    view_106: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_42, [8, 18, 18, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_134: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_135: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476);  view_106 = None
    erf_21: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_92: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_136: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_134, add_92);  mul_134 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_43: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_136);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_43, [2592, 2048]);  clone_43 = None
    permute_92: "f32[2048, 512]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_43: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg255_1, view_107, permute_92);  arg255_1 = view_107 = permute_92 = None
    view_108: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_43, [8, 18, 18, 512]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_44: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_93: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_44, [0, 3, 1, 2]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_109: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg71_1, [1, -1, 1, 1]);  arg71_1 = None
    mul_137: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_93, view_109);  permute_93 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_93: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_137, add_89);  mul_137 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_25: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_93, arg256_1, arg257_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg256_1 = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_94: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_25, [0, 2, 3, 1]);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_25 = torch.ops.aten.var_mean.correction(permute_94, [3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 18, 18, 1]" = var_mean_25[0]
    getitem_51: "f32[8, 18, 18, 1]" = var_mean_25[1];  var_mean_25 = None
    add_94: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_25: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_25: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_94, getitem_51);  permute_94 = getitem_51 = None
    mul_138: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
    mul_139: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_138, arg72_1);  mul_138 = arg72_1 = None
    add_95: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_139, arg73_1);  mul_139 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_110: "f32[2592, 512]" = torch.ops.aten.view.default(add_95, [2592, 512]);  add_95 = None
    permute_95: "f32[512, 2048]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
    addmm_44: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg259_1, view_110, permute_95);  arg259_1 = view_110 = permute_95 = None
    view_111: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_44, [8, 18, 18, 2048]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_140: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    mul_141: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
    erf_22: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_96: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_142: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_140, add_96);  mul_140 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_45: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_142);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_112: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_45, [2592, 2048]);  clone_45 = None
    permute_96: "f32[2048, 512]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    addmm_45: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg261_1, view_112, permute_96);  arg261_1 = view_112 = permute_96 = None
    view_113: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_45, [8, 18, 18, 512]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_46: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_113);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_97: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_46, [0, 3, 1, 2]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_114: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg74_1, [1, -1, 1, 1]);  arg74_1 = None
    mul_143: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_97, view_114);  permute_97 = view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_97: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_143, add_93);  mul_143 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_26: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_97, arg262_1, arg263_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg262_1 = arg263_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_98: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_26, [0, 2, 3, 1]);  convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_26 = torch.ops.aten.var_mean.correction(permute_98, [3], correction = 0, keepdim = True)
    getitem_52: "f32[8, 18, 18, 1]" = var_mean_26[0]
    getitem_53: "f32[8, 18, 18, 1]" = var_mean_26[1];  var_mean_26 = None
    add_98: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_26: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_26: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_98, getitem_53);  permute_98 = getitem_53 = None
    mul_144: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
    mul_145: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_144, arg75_1);  mul_144 = arg75_1 = None
    add_99: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_145, arg76_1);  mul_145 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_115: "f32[2592, 512]" = torch.ops.aten.view.default(add_99, [2592, 512]);  add_99 = None
    permute_99: "f32[512, 2048]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
    addmm_46: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg265_1, view_115, permute_99);  arg265_1 = view_115 = permute_99 = None
    view_116: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_46, [8, 18, 18, 2048]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_146: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_147: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476);  view_116 = None
    erf_23: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_100: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_148: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_146, add_100);  mul_146 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_117: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_47, [2592, 2048]);  clone_47 = None
    permute_100: "f32[2048, 512]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
    addmm_47: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg267_1, view_117, permute_100);  arg267_1 = view_117 = permute_100 = None
    view_118: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_47, [8, 18, 18, 512]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_118);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_101: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_48, [0, 3, 1, 2]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_119: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg77_1, [1, -1, 1, 1]);  arg77_1 = None
    mul_149: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_101, view_119);  permute_101 = view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_101: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_149, add_97);  mul_149 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_27: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_101, arg268_1, arg269_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg268_1 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_102: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_27 = torch.ops.aten.var_mean.correction(permute_102, [3], correction = 0, keepdim = True)
    getitem_54: "f32[8, 18, 18, 1]" = var_mean_27[0]
    getitem_55: "f32[8, 18, 18, 1]" = var_mean_27[1];  var_mean_27 = None
    add_102: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_27: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_27: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_102, getitem_55);  permute_102 = getitem_55 = None
    mul_150: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
    mul_151: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_150, arg78_1);  mul_150 = arg78_1 = None
    add_103: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_151, arg79_1);  mul_151 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_120: "f32[2592, 512]" = torch.ops.aten.view.default(add_103, [2592, 512]);  add_103 = None
    permute_103: "f32[512, 2048]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
    addmm_48: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg271_1, view_120, permute_103);  arg271_1 = view_120 = permute_103 = None
    view_121: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_48, [8, 18, 18, 2048]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_152: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_121, 0.5)
    mul_153: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_121, 0.7071067811865476);  view_121 = None
    erf_24: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_104: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_154: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_152, add_104);  mul_152 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_49: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_154);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_122: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_49, [2592, 2048]);  clone_49 = None
    permute_104: "f32[2048, 512]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
    addmm_49: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg273_1, view_122, permute_104);  arg273_1 = view_122 = permute_104 = None
    view_123: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_49, [8, 18, 18, 512]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_50: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_123);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_105: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_50, [0, 3, 1, 2]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_124: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg80_1, [1, -1, 1, 1]);  arg80_1 = None
    mul_155: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_105, view_124);  permute_105 = view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_105: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_155, add_101);  mul_155 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_28: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_105, arg274_1, arg275_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg274_1 = arg275_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_106: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_28, [0, 2, 3, 1]);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_28 = torch.ops.aten.var_mean.correction(permute_106, [3], correction = 0, keepdim = True)
    getitem_56: "f32[8, 18, 18, 1]" = var_mean_28[0]
    getitem_57: "f32[8, 18, 18, 1]" = var_mean_28[1];  var_mean_28 = None
    add_106: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_28: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_28: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_106, getitem_57);  permute_106 = getitem_57 = None
    mul_156: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
    mul_157: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_156, arg81_1);  mul_156 = arg81_1 = None
    add_107: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_157, arg82_1);  mul_157 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[2592, 512]" = torch.ops.aten.view.default(add_107, [2592, 512]);  add_107 = None
    permute_107: "f32[512, 2048]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
    addmm_50: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg277_1, view_125, permute_107);  arg277_1 = view_125 = permute_107 = None
    view_126: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_50, [8, 18, 18, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_158: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    mul_159: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476);  view_126 = None
    erf_25: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
    add_108: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_160: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_158, add_108);  mul_158 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_51: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_160);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_127: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_51, [2592, 2048]);  clone_51 = None
    permute_108: "f32[2048, 512]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
    addmm_51: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg279_1, view_127, permute_108);  arg279_1 = view_127 = permute_108 = None
    view_128: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_51, [8, 18, 18, 512]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_52: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_128);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_109: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_52, [0, 3, 1, 2]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_129: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg83_1, [1, -1, 1, 1]);  arg83_1 = None
    mul_161: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_109, view_129);  permute_109 = view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_109: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_161, add_105);  mul_161 = add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_29: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_109, arg280_1, arg281_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg280_1 = arg281_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_110: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_29, [0, 2, 3, 1]);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_29 = torch.ops.aten.var_mean.correction(permute_110, [3], correction = 0, keepdim = True)
    getitem_58: "f32[8, 18, 18, 1]" = var_mean_29[0]
    getitem_59: "f32[8, 18, 18, 1]" = var_mean_29[1];  var_mean_29 = None
    add_110: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
    rsqrt_29: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_29: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_110, getitem_59);  permute_110 = getitem_59 = None
    mul_162: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
    mul_163: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_162, arg84_1);  mul_162 = arg84_1 = None
    add_111: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_163, arg85_1);  mul_163 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_130: "f32[2592, 512]" = torch.ops.aten.view.default(add_111, [2592, 512]);  add_111 = None
    permute_111: "f32[512, 2048]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
    addmm_52: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg283_1, view_130, permute_111);  arg283_1 = view_130 = permute_111 = None
    view_131: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_52, [8, 18, 18, 2048]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_164: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_165: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
    erf_26: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_165);  mul_165 = None
    add_112: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_166: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_164, add_112);  mul_164 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_53: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_166);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_132: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_53, [2592, 2048]);  clone_53 = None
    permute_112: "f32[2048, 512]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
    addmm_53: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg285_1, view_132, permute_112);  arg285_1 = view_132 = permute_112 = None
    view_133: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_53, [8, 18, 18, 512]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_54: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_133);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_113: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_54, [0, 3, 1, 2]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_134: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg86_1, [1, -1, 1, 1]);  arg86_1 = None
    mul_167: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_113, view_134);  permute_113 = view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_113: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_167, add_109);  mul_167 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_30: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_113, arg286_1, arg287_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg286_1 = arg287_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_114: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1]);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_30 = torch.ops.aten.var_mean.correction(permute_114, [3], correction = 0, keepdim = True)
    getitem_60: "f32[8, 18, 18, 1]" = var_mean_30[0]
    getitem_61: "f32[8, 18, 18, 1]" = var_mean_30[1];  var_mean_30 = None
    add_114: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_30: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_30: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_114, getitem_61);  permute_114 = getitem_61 = None
    mul_168: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
    mul_169: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_168, arg87_1);  mul_168 = arg87_1 = None
    add_115: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_169, arg88_1);  mul_169 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_135: "f32[2592, 512]" = torch.ops.aten.view.default(add_115, [2592, 512]);  add_115 = None
    permute_115: "f32[512, 2048]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
    addmm_54: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg289_1, view_135, permute_115);  arg289_1 = view_135 = permute_115 = None
    view_136: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_54, [8, 18, 18, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_170: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_136, 0.5)
    mul_171: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_136, 0.7071067811865476);  view_136 = None
    erf_27: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_116: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_172: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_170, add_116);  mul_170 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_55: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_172);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_137: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_55, [2592, 2048]);  clone_55 = None
    permute_116: "f32[2048, 512]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
    addmm_55: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg291_1, view_137, permute_116);  arg291_1 = view_137 = permute_116 = None
    view_138: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_55, [8, 18, 18, 512]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_138);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_117: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_56, [0, 3, 1, 2]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_139: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg89_1, [1, -1, 1, 1]);  arg89_1 = None
    mul_173: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_117, view_139);  permute_117 = view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_117: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_173, add_113);  mul_173 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_31: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_117, arg292_1, arg293_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg292_1 = arg293_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_118: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_31, [0, 2, 3, 1]);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_31 = torch.ops.aten.var_mean.correction(permute_118, [3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 18, 18, 1]" = var_mean_31[0]
    getitem_63: "f32[8, 18, 18, 1]" = var_mean_31[1];  var_mean_31 = None
    add_118: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_31: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_31: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_118, getitem_63);  permute_118 = getitem_63 = None
    mul_174: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
    mul_175: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_174, arg90_1);  mul_174 = arg90_1 = None
    add_119: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_175, arg91_1);  mul_175 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_140: "f32[2592, 512]" = torch.ops.aten.view.default(add_119, [2592, 512]);  add_119 = None
    permute_119: "f32[512, 2048]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
    addmm_56: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg295_1, view_140, permute_119);  arg295_1 = view_140 = permute_119 = None
    view_141: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_56, [8, 18, 18, 2048]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_176: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
    mul_177: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
    erf_28: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_120: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_178: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_176, add_120);  mul_176 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_57: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_178);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_142: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_57, [2592, 2048]);  clone_57 = None
    permute_120: "f32[2048, 512]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
    addmm_57: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg297_1, view_142, permute_120);  arg297_1 = view_142 = permute_120 = None
    view_143: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_57, [8, 18, 18, 512]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_58: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_143);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_121: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_58, [0, 3, 1, 2]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_144: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg92_1, [1, -1, 1, 1]);  arg92_1 = None
    mul_179: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_121, view_144);  permute_121 = view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_121: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_179, add_117);  mul_179 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_32: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_121, arg298_1, arg299_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg298_1 = arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_122: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_32, [0, 2, 3, 1]);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_32 = torch.ops.aten.var_mean.correction(permute_122, [3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 18, 18, 1]" = var_mean_32[0]
    getitem_65: "f32[8, 18, 18, 1]" = var_mean_32[1];  var_mean_32 = None
    add_122: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_32: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_32: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_122, getitem_65);  permute_122 = getitem_65 = None
    mul_180: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
    mul_181: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_180, arg93_1);  mul_180 = arg93_1 = None
    add_123: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_181, arg94_1);  mul_181 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_145: "f32[2592, 512]" = torch.ops.aten.view.default(add_123, [2592, 512]);  add_123 = None
    permute_123: "f32[512, 2048]" = torch.ops.aten.permute.default(arg300_1, [1, 0]);  arg300_1 = None
    addmm_58: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg301_1, view_145, permute_123);  arg301_1 = view_145 = permute_123 = None
    view_146: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_58, [8, 18, 18, 2048]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_182: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_146, 0.5)
    mul_183: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476);  view_146 = None
    erf_29: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_183);  mul_183 = None
    add_124: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_184: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_182, add_124);  mul_182 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_59: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_184);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_147: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_59, [2592, 2048]);  clone_59 = None
    permute_124: "f32[2048, 512]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
    addmm_59: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg303_1, view_147, permute_124);  arg303_1 = view_147 = permute_124 = None
    view_148: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_59, [8, 18, 18, 512]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_60: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_148);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_125: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_60, [0, 3, 1, 2]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_149: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg95_1, [1, -1, 1, 1]);  arg95_1 = None
    mul_185: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_125, view_149);  permute_125 = view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_125: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_185, add_121);  mul_185 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_33: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_125, arg304_1, arg305_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg304_1 = arg305_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_126: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_33 = torch.ops.aten.var_mean.correction(permute_126, [3], correction = 0, keepdim = True)
    getitem_66: "f32[8, 18, 18, 1]" = var_mean_33[0]
    getitem_67: "f32[8, 18, 18, 1]" = var_mean_33[1];  var_mean_33 = None
    add_126: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_33: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_33: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_126, getitem_67);  permute_126 = getitem_67 = None
    mul_186: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
    mul_187: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_186, arg96_1);  mul_186 = arg96_1 = None
    add_127: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_187, arg97_1);  mul_187 = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_150: "f32[2592, 512]" = torch.ops.aten.view.default(add_127, [2592, 512]);  add_127 = None
    permute_127: "f32[512, 2048]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
    addmm_60: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg307_1, view_150, permute_127);  arg307_1 = view_150 = permute_127 = None
    view_151: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_60, [8, 18, 18, 2048]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_188: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_189: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_30: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_189);  mul_189 = None
    add_128: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_190: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_188, add_128);  mul_188 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_61: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_190);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_152: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_61, [2592, 2048]);  clone_61 = None
    permute_128: "f32[2048, 512]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
    addmm_61: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg309_1, view_152, permute_128);  arg309_1 = view_152 = permute_128 = None
    view_153: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_61, [8, 18, 18, 512]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_62: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_153);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_129: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_62, [0, 3, 1, 2]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_154: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg98_1, [1, -1, 1, 1]);  arg98_1 = None
    mul_191: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_129, view_154);  permute_129 = view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_129: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_191, add_125);  mul_191 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_34: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_129, arg310_1, arg311_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg310_1 = arg311_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_130: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_34, [0, 2, 3, 1]);  convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_34 = torch.ops.aten.var_mean.correction(permute_130, [3], correction = 0, keepdim = True)
    getitem_68: "f32[8, 18, 18, 1]" = var_mean_34[0]
    getitem_69: "f32[8, 18, 18, 1]" = var_mean_34[1];  var_mean_34 = None
    add_130: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_34: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_34: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_130, getitem_69);  permute_130 = getitem_69 = None
    mul_192: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
    mul_193: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_192, arg99_1);  mul_192 = arg99_1 = None
    add_131: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_193, arg100_1);  mul_193 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_155: "f32[2592, 512]" = torch.ops.aten.view.default(add_131, [2592, 512]);  add_131 = None
    permute_131: "f32[512, 2048]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
    addmm_62: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg313_1, view_155, permute_131);  arg313_1 = view_155 = permute_131 = None
    view_156: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_62, [8, 18, 18, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_194: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_156, 0.5)
    mul_195: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_156, 0.7071067811865476);  view_156 = None
    erf_31: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_132: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_196: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_194, add_132);  mul_194 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_63: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_196);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_157: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_63, [2592, 2048]);  clone_63 = None
    permute_132: "f32[2048, 512]" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
    addmm_63: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg315_1, view_157, permute_132);  arg315_1 = view_157 = permute_132 = None
    view_158: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_63, [8, 18, 18, 512]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_64: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_158);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_133: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_64, [0, 3, 1, 2]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_159: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg101_1, [1, -1, 1, 1]);  arg101_1 = None
    mul_197: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_133, view_159);  permute_133 = view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_133: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_197, add_129);  mul_197 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_35: "f32[8, 512, 18, 18]" = torch.ops.aten.convolution.default(add_133, arg316_1, arg317_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 512);  arg316_1 = arg317_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_134: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(convolution_35, [0, 2, 3, 1]);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_35 = torch.ops.aten.var_mean.correction(permute_134, [3], correction = 0, keepdim = True)
    getitem_70: "f32[8, 18, 18, 1]" = var_mean_35[0]
    getitem_71: "f32[8, 18, 18, 1]" = var_mean_35[1];  var_mean_35 = None
    add_134: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_35: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_35: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_134, getitem_71);  permute_134 = getitem_71 = None
    mul_198: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
    mul_199: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_198, arg102_1);  mul_198 = arg102_1 = None
    add_135: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_199, arg103_1);  mul_199 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_160: "f32[2592, 512]" = torch.ops.aten.view.default(add_135, [2592, 512]);  add_135 = None
    permute_135: "f32[512, 2048]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
    addmm_64: "f32[2592, 2048]" = torch.ops.aten.addmm.default(arg319_1, view_160, permute_135);  arg319_1 = view_160 = permute_135 = None
    view_161: "f32[8, 18, 18, 2048]" = torch.ops.aten.view.default(addmm_64, [8, 18, 18, 2048]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_200: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    mul_201: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(view_161, 0.7071067811865476);  view_161 = None
    erf_32: "f32[8, 18, 18, 2048]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_136: "f32[8, 18, 18, 2048]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_202: "f32[8, 18, 18, 2048]" = torch.ops.aten.mul.Tensor(mul_200, add_136);  mul_200 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_65: "f32[8, 18, 18, 2048]" = torch.ops.aten.clone.default(mul_202);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_162: "f32[2592, 2048]" = torch.ops.aten.view.default(clone_65, [2592, 2048]);  clone_65 = None
    permute_136: "f32[2048, 512]" = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
    addmm_65: "f32[2592, 512]" = torch.ops.aten.addmm.default(arg321_1, view_162, permute_136);  arg321_1 = view_162 = permute_136 = None
    view_163: "f32[8, 18, 18, 512]" = torch.ops.aten.view.default(addmm_65, [8, 18, 18, 512]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_66: "f32[8, 18, 18, 512]" = torch.ops.aten.clone.default(view_163);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_137: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(clone_66, [0, 3, 1, 2]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_164: "f32[1, 512, 1, 1]" = torch.ops.aten.view.default(arg104_1, [1, -1, 1, 1]);  arg104_1 = None
    mul_203: "f32[8, 512, 18, 18]" = torch.ops.aten.mul.Tensor(permute_137, view_164);  permute_137 = view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_137: "f32[8, 512, 18, 18]" = torch.ops.aten.add.Tensor(mul_203, add_133);  mul_203 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_138: "f32[8, 18, 18, 512]" = torch.ops.aten.permute.default(add_137, [0, 2, 3, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_36 = torch.ops.aten.var_mean.correction(permute_138, [3], correction = 0, keepdim = True)
    getitem_72: "f32[8, 18, 18, 1]" = var_mean_36[0]
    getitem_73: "f32[8, 18, 18, 1]" = var_mean_36[1];  var_mean_36 = None
    add_138: "f32[8, 18, 18, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_36: "f32[8, 18, 18, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_36: "f32[8, 18, 18, 512]" = torch.ops.aten.sub.Tensor(permute_138, getitem_73);  permute_138 = getitem_73 = None
    mul_204: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
    mul_205: "f32[8, 18, 18, 512]" = torch.ops.aten.mul.Tensor(mul_204, arg105_1);  mul_204 = arg105_1 = None
    add_139: "f32[8, 18, 18, 512]" = torch.ops.aten.add.Tensor(mul_205, arg106_1);  mul_205 = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_139: "f32[8, 512, 18, 18]" = torch.ops.aten.permute.default(add_139, [0, 3, 1, 2]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_36: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(permute_139, arg322_1, arg323_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_139 = arg322_1 = arg323_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_37: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(convolution_36, arg324_1, arg325_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  arg324_1 = arg325_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_140: "f32[8, 9, 9, 1024]" = torch.ops.aten.permute.default(convolution_37, [0, 2, 3, 1]);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_37 = torch.ops.aten.var_mean.correction(permute_140, [3], correction = 0, keepdim = True)
    getitem_74: "f32[8, 9, 9, 1]" = var_mean_37[0]
    getitem_75: "f32[8, 9, 9, 1]" = var_mean_37[1];  var_mean_37 = None
    add_140: "f32[8, 9, 9, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
    rsqrt_37: "f32[8, 9, 9, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_37: "f32[8, 9, 9, 1024]" = torch.ops.aten.sub.Tensor(permute_140, getitem_75);  permute_140 = getitem_75 = None
    mul_206: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
    mul_207: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(mul_206, arg107_1);  mul_206 = arg107_1 = None
    add_141: "f32[8, 9, 9, 1024]" = torch.ops.aten.add.Tensor(mul_207, arg108_1);  mul_207 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_165: "f32[648, 1024]" = torch.ops.aten.view.default(add_141, [648, 1024]);  add_141 = None
    permute_141: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
    addmm_66: "f32[648, 4096]" = torch.ops.aten.addmm.default(arg327_1, view_165, permute_141);  arg327_1 = view_165 = permute_141 = None
    view_166: "f32[8, 9, 9, 4096]" = torch.ops.aten.view.default(addmm_66, [8, 9, 9, 4096]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_208: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_166, 0.5)
    mul_209: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_166, 0.7071067811865476);  view_166 = None
    erf_33: "f32[8, 9, 9, 4096]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_142: "f32[8, 9, 9, 4096]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_210: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(mul_208, add_142);  mul_208 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_67: "f32[8, 9, 9, 4096]" = torch.ops.aten.clone.default(mul_210);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_167: "f32[648, 4096]" = torch.ops.aten.view.default(clone_67, [648, 4096]);  clone_67 = None
    permute_142: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
    addmm_67: "f32[648, 1024]" = torch.ops.aten.addmm.default(arg329_1, view_167, permute_142);  arg329_1 = view_167 = permute_142 = None
    view_168: "f32[8, 9, 9, 1024]" = torch.ops.aten.view.default(addmm_67, [8, 9, 9, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_68: "f32[8, 9, 9, 1024]" = torch.ops.aten.clone.default(view_168);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_143: "f32[8, 1024, 9, 9]" = torch.ops.aten.permute.default(clone_68, [0, 3, 1, 2]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_169: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(arg109_1, [1, -1, 1, 1]);  arg109_1 = None
    mul_211: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(permute_143, view_169);  permute_143 = view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_143: "f32[8, 1024, 9, 9]" = torch.ops.aten.add.Tensor(mul_211, convolution_36);  mul_211 = convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_38: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(add_143, arg330_1, arg331_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  arg330_1 = arg331_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_144: "f32[8, 9, 9, 1024]" = torch.ops.aten.permute.default(convolution_38, [0, 2, 3, 1]);  convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_38 = torch.ops.aten.var_mean.correction(permute_144, [3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 9, 9, 1]" = var_mean_38[0]
    getitem_77: "f32[8, 9, 9, 1]" = var_mean_38[1];  var_mean_38 = None
    add_144: "f32[8, 9, 9, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_38: "f32[8, 9, 9, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_38: "f32[8, 9, 9, 1024]" = torch.ops.aten.sub.Tensor(permute_144, getitem_77);  permute_144 = getitem_77 = None
    mul_212: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
    mul_213: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(mul_212, arg110_1);  mul_212 = arg110_1 = None
    add_145: "f32[8, 9, 9, 1024]" = torch.ops.aten.add.Tensor(mul_213, arg111_1);  mul_213 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_170: "f32[648, 1024]" = torch.ops.aten.view.default(add_145, [648, 1024]);  add_145 = None
    permute_145: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
    addmm_68: "f32[648, 4096]" = torch.ops.aten.addmm.default(arg333_1, view_170, permute_145);  arg333_1 = view_170 = permute_145 = None
    view_171: "f32[8, 9, 9, 4096]" = torch.ops.aten.view.default(addmm_68, [8, 9, 9, 4096]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_214: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_171, 0.5)
    mul_215: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_171, 0.7071067811865476);  view_171 = None
    erf_34: "f32[8, 9, 9, 4096]" = torch.ops.aten.erf.default(mul_215);  mul_215 = None
    add_146: "f32[8, 9, 9, 4096]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_216: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(mul_214, add_146);  mul_214 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_69: "f32[8, 9, 9, 4096]" = torch.ops.aten.clone.default(mul_216);  mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_172: "f32[648, 4096]" = torch.ops.aten.view.default(clone_69, [648, 4096]);  clone_69 = None
    permute_146: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
    addmm_69: "f32[648, 1024]" = torch.ops.aten.addmm.default(arg335_1, view_172, permute_146);  arg335_1 = view_172 = permute_146 = None
    view_173: "f32[8, 9, 9, 1024]" = torch.ops.aten.view.default(addmm_69, [8, 9, 9, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_70: "f32[8, 9, 9, 1024]" = torch.ops.aten.clone.default(view_173);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_147: "f32[8, 1024, 9, 9]" = torch.ops.aten.permute.default(clone_70, [0, 3, 1, 2]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_174: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(arg112_1, [1, -1, 1, 1]);  arg112_1 = None
    mul_217: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(permute_147, view_174);  permute_147 = view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_147: "f32[8, 1024, 9, 9]" = torch.ops.aten.add.Tensor(mul_217, add_143);  mul_217 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_39: "f32[8, 1024, 9, 9]" = torch.ops.aten.convolution.default(add_147, arg336_1, arg337_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 1024);  arg336_1 = arg337_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_148: "f32[8, 9, 9, 1024]" = torch.ops.aten.permute.default(convolution_39, [0, 2, 3, 1]);  convolution_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_39 = torch.ops.aten.var_mean.correction(permute_148, [3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 9, 9, 1]" = var_mean_39[0]
    getitem_79: "f32[8, 9, 9, 1]" = var_mean_39[1];  var_mean_39 = None
    add_148: "f32[8, 9, 9, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_39: "f32[8, 9, 9, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_39: "f32[8, 9, 9, 1024]" = torch.ops.aten.sub.Tensor(permute_148, getitem_79);  permute_148 = getitem_79 = None
    mul_218: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
    mul_219: "f32[8, 9, 9, 1024]" = torch.ops.aten.mul.Tensor(mul_218, arg113_1);  mul_218 = arg113_1 = None
    add_149: "f32[8, 9, 9, 1024]" = torch.ops.aten.add.Tensor(mul_219, arg114_1);  mul_219 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_175: "f32[648, 1024]" = torch.ops.aten.view.default(add_149, [648, 1024]);  add_149 = None
    permute_149: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
    addmm_70: "f32[648, 4096]" = torch.ops.aten.addmm.default(arg339_1, view_175, permute_149);  arg339_1 = view_175 = permute_149 = None
    view_176: "f32[8, 9, 9, 4096]" = torch.ops.aten.view.default(addmm_70, [8, 9, 9, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_220: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_176, 0.5)
    mul_221: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476);  view_176 = None
    erf_35: "f32[8, 9, 9, 4096]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_150: "f32[8, 9, 9, 4096]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_222: "f32[8, 9, 9, 4096]" = torch.ops.aten.mul.Tensor(mul_220, add_150);  mul_220 = add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_71: "f32[8, 9, 9, 4096]" = torch.ops.aten.clone.default(mul_222);  mul_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_177: "f32[648, 4096]" = torch.ops.aten.view.default(clone_71, [648, 4096]);  clone_71 = None
    permute_150: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
    addmm_71: "f32[648, 1024]" = torch.ops.aten.addmm.default(arg341_1, view_177, permute_150);  arg341_1 = view_177 = permute_150 = None
    view_178: "f32[8, 9, 9, 1024]" = torch.ops.aten.view.default(addmm_71, [8, 9, 9, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_72: "f32[8, 9, 9, 1024]" = torch.ops.aten.clone.default(view_178);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_151: "f32[8, 1024, 9, 9]" = torch.ops.aten.permute.default(clone_72, [0, 3, 1, 2]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_179: "f32[1, 1024, 1, 1]" = torch.ops.aten.view.default(arg115_1, [1, -1, 1, 1]);  arg115_1 = None
    mul_223: "f32[8, 1024, 9, 9]" = torch.ops.aten.mul.Tensor(permute_151, view_179);  permute_151 = view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:164, code: x = self.drop_path(x) + self.shortcut(shortcut)
    add_151: "f32[8, 1024, 9, 9]" = torch.ops.aten.add.Tensor(mul_223, add_147);  mul_223 = add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(add_151, [-1, -2], True);  add_151 = None
    as_strided: "f32[8, 1024, 1, 1]" = torch.ops.aten.as_strided.default(mean, [8, 1024, 1, 1], [1024, 1, 1024, 1024]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_152: "f32[8, 1, 1, 1024]" = torch.ops.aten.permute.default(as_strided, [0, 2, 3, 1]);  as_strided = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_40 = torch.ops.aten.var_mean.correction(permute_152, [3], correction = 0, keepdim = True)
    getitem_80: "f32[8, 1, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[8, 1, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_152: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_40: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_40: "f32[8, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(permute_152, getitem_81);  permute_152 = getitem_81 = None
    mul_224: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
    mul_225: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_224, arg116_1);  mul_224 = arg116_1 = None
    add_153: "f32[8, 1, 1, 1024]" = torch.ops.aten.add.Tensor(mul_225, arg117_1);  mul_225 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_153: "f32[8, 1024, 1, 1]" = torch.ops.aten.permute.default(add_153, [0, 3, 1, 2]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:202, code: x = self.flatten(x)
    view_180: "f32[8, 1024]" = torch.ops.aten.view.default(permute_153, [8, 1024]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:204, code: x = self.drop(x)
    clone_73: "f32[8, 1024]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:207, code: x = self.fc(x)
    permute_154: "f32[1024, 1000]" = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
    addmm_72: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg343_1, clone_73, permute_154);  arg343_1 = clone_73 = permute_154 = None
    return (addmm_72,)
    