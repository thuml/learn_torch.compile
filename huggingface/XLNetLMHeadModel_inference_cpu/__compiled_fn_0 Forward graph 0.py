from __future__ import annotations



def forward(self, arg0_1: "f32[1024, 16, 64]", arg1_1: "f32[1024, 16, 64]", arg2_1: "f32[1024, 16, 64]", arg3_1: "f32[1024, 16, 64]", arg4_1: "f32[16, 64]", arg5_1: "f32[16, 64]", arg6_1: "f32[1024, 16, 64]", arg7_1: "f32[1024, 16, 64]", arg8_1: "f32[1024, 16, 64]", arg9_1: "f32[1024, 16, 64]", arg10_1: "f32[1024, 16, 64]", arg11_1: "f32[16, 64]", arg12_1: "f32[16, 64]", arg13_1: "f32[1024, 16, 64]", arg14_1: "f32[1024, 16, 64]", arg15_1: "f32[1024, 16, 64]", arg16_1: "f32[1024, 16, 64]", arg17_1: "f32[1024, 16, 64]", arg18_1: "f32[16, 64]", arg19_1: "f32[16, 64]", arg20_1: "f32[1024, 16, 64]", arg21_1: "f32[1024, 16, 64]", arg22_1: "f32[1024, 16, 64]", arg23_1: "f32[1024, 16, 64]", arg24_1: "f32[1024, 16, 64]", arg25_1: "f32[16, 64]", arg26_1: "f32[16, 64]", arg27_1: "f32[1024, 16, 64]", arg28_1: "f32[1024, 16, 64]", arg29_1: "f32[1024, 16, 64]", arg30_1: "f32[1024, 16, 64]", arg31_1: "f32[1024, 16, 64]", arg32_1: "f32[16, 64]", arg33_1: "f32[16, 64]", arg34_1: "f32[1024, 16, 64]", arg35_1: "f32[1024, 16, 64]", arg36_1: "f32[1024, 16, 64]", arg37_1: "f32[1024, 16, 64]", arg38_1: "f32[1024, 16, 64]", arg39_1: "f32[16, 64]", arg40_1: "f32[16, 64]", arg41_1: "f32[1024, 16, 64]", arg42_1: "f32[1024, 16, 64]", arg43_1: "f32[1024, 16, 64]", arg44_1: "f32[1024, 16, 64]", arg45_1: "f32[1024, 16, 64]", arg46_1: "f32[16, 64]", arg47_1: "f32[16, 64]", arg48_1: "f32[1024, 16, 64]", arg49_1: "f32[1024, 16, 64]", arg50_1: "f32[1024, 16, 64]", arg51_1: "f32[1024, 16, 64]", arg52_1: "f32[1024, 16, 64]", arg53_1: "f32[16, 64]", arg54_1: "f32[16, 64]", arg55_1: "f32[1024, 16, 64]", arg56_1: "f32[1024, 16, 64]", arg57_1: "f32[1024, 16, 64]", arg58_1: "f32[1024, 16, 64]", arg59_1: "f32[1024, 16, 64]", arg60_1: "f32[16, 64]", arg61_1: "f32[16, 64]", arg62_1: "f32[1024, 16, 64]", arg63_1: "f32[1024, 16, 64]", arg64_1: "f32[1024, 16, 64]", arg65_1: "f32[1024, 16, 64]", arg66_1: "f32[1024, 16, 64]", arg67_1: "f32[16, 64]", arg68_1: "f32[16, 64]", arg69_1: "f32[1024, 16, 64]", arg70_1: "f32[1024, 16, 64]", arg71_1: "f32[1024, 16, 64]", arg72_1: "f32[1024, 16, 64]", arg73_1: "f32[1024, 16, 64]", arg74_1: "f32[16, 64]", arg75_1: "f32[16, 64]", arg76_1: "f32[1024, 16, 64]", arg77_1: "f32[1024, 16, 64]", arg78_1: "f32[1024, 16, 64]", arg79_1: "f32[1024, 16, 64]", arg80_1: "f32[1024, 16, 64]", arg81_1: "f32[16, 64]", arg82_1: "f32[16, 64]", arg83_1: "f32[1024, 16, 64]", arg84_1: "f32[1024, 16, 64]", arg85_1: "f32[1024, 16, 64]", arg86_1: "f32[1024, 16, 64]", arg87_1: "f32[1024, 16, 64]", arg88_1: "f32[16, 64]", arg89_1: "f32[16, 64]", arg90_1: "f32[1024, 16, 64]", arg91_1: "f32[1024, 16, 64]", arg92_1: "f32[1024, 16, 64]", arg93_1: "f32[1024, 16, 64]", arg94_1: "f32[1024, 16, 64]", arg95_1: "f32[16, 64]", arg96_1: "f32[16, 64]", arg97_1: "f32[1024, 16, 64]", arg98_1: "f32[1024, 16, 64]", arg99_1: "f32[1024, 16, 64]", arg100_1: "f32[1024, 16, 64]", arg101_1: "f32[1024, 16, 64]", arg102_1: "f32[16, 64]", arg103_1: "f32[16, 64]", arg104_1: "f32[1024, 16, 64]", arg105_1: "f32[1024, 16, 64]", arg106_1: "f32[1024, 16, 64]", arg107_1: "f32[1024, 16, 64]", arg108_1: "f32[1024, 16, 64]", arg109_1: "f32[16, 64]", arg110_1: "f32[16, 64]", arg111_1: "f32[1024, 16, 64]", arg112_1: "f32[1024, 16, 64]", arg113_1: "f32[1024, 16, 64]", arg114_1: "f32[1024, 16, 64]", arg115_1: "f32[1024, 16, 64]", arg116_1: "f32[16, 64]", arg117_1: "f32[16, 64]", arg118_1: "f32[1024, 16, 64]", arg119_1: "f32[1024, 16, 64]", arg120_1: "f32[1024, 16, 64]", arg121_1: "f32[1024, 16, 64]", arg122_1: "f32[1024, 16, 64]", arg123_1: "f32[16, 64]", arg124_1: "f32[16, 64]", arg125_1: "f32[1024, 16, 64]", arg126_1: "f32[1024, 16, 64]", arg127_1: "f32[1024, 16, 64]", arg128_1: "f32[1024, 16, 64]", arg129_1: "f32[1024, 16, 64]", arg130_1: "f32[16, 64]", arg131_1: "f32[16, 64]", arg132_1: "f32[1024, 16, 64]", arg133_1: "f32[1024, 16, 64]", arg134_1: "f32[1024, 16, 64]", arg135_1: "f32[1024, 16, 64]", arg136_1: "f32[1024, 16, 64]", arg137_1: "f32[16, 64]", arg138_1: "f32[16, 64]", arg139_1: "f32[1024, 16, 64]", arg140_1: "f32[1024, 16, 64]", arg141_1: "f32[1024, 16, 64]", arg142_1: "f32[1024, 16, 64]", arg143_1: "f32[1024, 16, 64]", arg144_1: "f32[16, 64]", arg145_1: "f32[16, 64]", arg146_1: "f32[1024, 16, 64]", arg147_1: "f32[1024, 16, 64]", arg148_1: "f32[1024, 16, 64]", arg149_1: "f32[1024, 16, 64]", arg150_1: "f32[1024, 16, 64]", arg151_1: "f32[16, 64]", arg152_1: "f32[16, 64]", arg153_1: "f32[1024, 16, 64]", arg154_1: "f32[1024, 16, 64]", arg155_1: "f32[1024, 16, 64]", arg156_1: "f32[1024, 16, 64]", arg157_1: "f32[1024, 16, 64]", arg158_1: "f32[16, 64]", arg159_1: "f32[16, 64]", arg160_1: "f32[1024, 16, 64]", arg161_1: "f32[1024, 16, 64]", arg162_1: "f32[1024, 16, 64]", arg163_1: "f32[1024, 16, 64]", arg164_1: "f32[1024, 16, 64]", arg165_1: "f32[16, 64]", arg166_1: "f32[16, 64]", arg167_1: "f32[1024, 16, 64]", arg168_1: "f32[32000, 1024]", arg169_1: "f32[1024]", arg170_1: "f32[1024]", arg171_1: "f32[4096, 1024]", arg172_1: "f32[4096]", arg173_1: "f32[1024, 4096]", arg174_1: "f32[1024]", arg175_1: "f32[1024]", arg176_1: "f32[1024]", arg177_1: "f32[1024]", arg178_1: "f32[1024]", arg179_1: "f32[4096, 1024]", arg180_1: "f32[4096]", arg181_1: "f32[1024, 4096]", arg182_1: "f32[1024]", arg183_1: "f32[1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024]", arg186_1: "f32[1024]", arg187_1: "f32[4096, 1024]", arg188_1: "f32[4096]", arg189_1: "f32[1024, 4096]", arg190_1: "f32[1024]", arg191_1: "f32[1024]", arg192_1: "f32[1024]", arg193_1: "f32[1024]", arg194_1: "f32[1024]", arg195_1: "f32[4096, 1024]", arg196_1: "f32[4096]", arg197_1: "f32[1024, 4096]", arg198_1: "f32[1024]", arg199_1: "f32[1024]", arg200_1: "f32[1024]", arg201_1: "f32[1024]", arg202_1: "f32[1024]", arg203_1: "f32[4096, 1024]", arg204_1: "f32[4096]", arg205_1: "f32[1024, 4096]", arg206_1: "f32[1024]", arg207_1: "f32[1024]", arg208_1: "f32[1024]", arg209_1: "f32[1024]", arg210_1: "f32[1024]", arg211_1: "f32[4096, 1024]", arg212_1: "f32[4096]", arg213_1: "f32[1024, 4096]", arg214_1: "f32[1024]", arg215_1: "f32[1024]", arg216_1: "f32[1024]", arg217_1: "f32[1024]", arg218_1: "f32[1024]", arg219_1: "f32[4096, 1024]", arg220_1: "f32[4096]", arg221_1: "f32[1024, 4096]", arg222_1: "f32[1024]", arg223_1: "f32[1024]", arg224_1: "f32[1024]", arg225_1: "f32[1024]", arg226_1: "f32[1024]", arg227_1: "f32[4096, 1024]", arg228_1: "f32[4096]", arg229_1: "f32[1024, 4096]", arg230_1: "f32[1024]", arg231_1: "f32[1024]", arg232_1: "f32[1024]", arg233_1: "f32[1024]", arg234_1: "f32[1024]", arg235_1: "f32[4096, 1024]", arg236_1: "f32[4096]", arg237_1: "f32[1024, 4096]", arg238_1: "f32[1024]", arg239_1: "f32[1024]", arg240_1: "f32[1024]", arg241_1: "f32[1024]", arg242_1: "f32[1024]", arg243_1: "f32[4096, 1024]", arg244_1: "f32[4096]", arg245_1: "f32[1024, 4096]", arg246_1: "f32[1024]", arg247_1: "f32[1024]", arg248_1: "f32[1024]", arg249_1: "f32[1024]", arg250_1: "f32[1024]", arg251_1: "f32[4096, 1024]", arg252_1: "f32[4096]", arg253_1: "f32[1024, 4096]", arg254_1: "f32[1024]", arg255_1: "f32[1024]", arg256_1: "f32[1024]", arg257_1: "f32[1024]", arg258_1: "f32[1024]", arg259_1: "f32[4096, 1024]", arg260_1: "f32[4096]", arg261_1: "f32[1024, 4096]", arg262_1: "f32[1024]", arg263_1: "f32[1024]", arg264_1: "f32[1024]", arg265_1: "f32[1024]", arg266_1: "f32[1024]", arg267_1: "f32[4096, 1024]", arg268_1: "f32[4096]", arg269_1: "f32[1024, 4096]", arg270_1: "f32[1024]", arg271_1: "f32[1024]", arg272_1: "f32[1024]", arg273_1: "f32[1024]", arg274_1: "f32[1024]", arg275_1: "f32[4096, 1024]", arg276_1: "f32[4096]", arg277_1: "f32[1024, 4096]", arg278_1: "f32[1024]", arg279_1: "f32[1024]", arg280_1: "f32[1024]", arg281_1: "f32[1024]", arg282_1: "f32[1024]", arg283_1: "f32[4096, 1024]", arg284_1: "f32[4096]", arg285_1: "f32[1024, 4096]", arg286_1: "f32[1024]", arg287_1: "f32[1024]", arg288_1: "f32[1024]", arg289_1: "f32[1024]", arg290_1: "f32[1024]", arg291_1: "f32[4096, 1024]", arg292_1: "f32[4096]", arg293_1: "f32[1024, 4096]", arg294_1: "f32[1024]", arg295_1: "f32[1024]", arg296_1: "f32[1024]", arg297_1: "f32[1024]", arg298_1: "f32[1024]", arg299_1: "f32[4096, 1024]", arg300_1: "f32[4096]", arg301_1: "f32[1024, 4096]", arg302_1: "f32[1024]", arg303_1: "f32[1024]", arg304_1: "f32[1024]", arg305_1: "f32[1024]", arg306_1: "f32[1024]", arg307_1: "f32[4096, 1024]", arg308_1: "f32[4096]", arg309_1: "f32[1024, 4096]", arg310_1: "f32[1024]", arg311_1: "f32[1024]", arg312_1: "f32[1024]", arg313_1: "f32[1024]", arg314_1: "f32[1024]", arg315_1: "f32[4096, 1024]", arg316_1: "f32[4096]", arg317_1: "f32[1024, 4096]", arg318_1: "f32[1024]", arg319_1: "f32[1024]", arg320_1: "f32[1024]", arg321_1: "f32[1024]", arg322_1: "f32[1024]", arg323_1: "f32[4096, 1024]", arg324_1: "f32[4096]", arg325_1: "f32[1024, 4096]", arg326_1: "f32[1024]", arg327_1: "f32[1024]", arg328_1: "f32[1024]", arg329_1: "f32[1024]", arg330_1: "f32[1024]", arg331_1: "f32[4096, 1024]", arg332_1: "f32[4096]", arg333_1: "f32[1024, 4096]", arg334_1: "f32[1024]", arg335_1: "f32[1024]", arg336_1: "f32[1024]", arg337_1: "f32[1024]", arg338_1: "f32[1024]", arg339_1: "f32[4096, 1024]", arg340_1: "f32[4096]", arg341_1: "f32[1024, 4096]", arg342_1: "f32[1024]", arg343_1: "f32[1024]", arg344_1: "f32[1024]", arg345_1: "f32[1024]", arg346_1: "f32[1024]", arg347_1: "f32[4096, 1024]", arg348_1: "f32[4096]", arg349_1: "f32[1024, 4096]", arg350_1: "f32[1024]", arg351_1: "f32[1024]", arg352_1: "f32[1024]", arg353_1: "f32[1024]", arg354_1: "f32[1024]", arg355_1: "f32[4096, 1024]", arg356_1: "f32[4096]", arg357_1: "f32[1024, 4096]", arg358_1: "f32[1024]", arg359_1: "f32[1024]", arg360_1: "f32[1024]", arg361_1: "f32[32000, 1024]", arg362_1: "f32[32000]", arg363_1: "i64[1, 512]", arg364_1: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1107, code: input_ids = input_ids.transpose(0, 1).contiguous()
    permute: "i64[512, 1]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1176, code: word_emb_k = self.word_embedding(input_ids)
    embedding: "f32[512, 1, 1024]" = torch.ops.aten.embedding.default(arg168_1, permute);  arg168_1 = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1177, code: output_h = self.dropout(word_emb_k)
    clone: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(embedding);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1023, code: freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    convert_element_type: "f64[512]" = torch.ops.prims.convert_element_type.default(iota, torch.float64);  iota = None
    mul: "f64[512]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
    add: "f64[512]" = torch.ops.aten.add.Tensor(mul, 0);  mul = None
    convert_element_type_1: "f32[512]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1024, code: inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))
    div: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_1, 1024);  convert_element_type_1 = None
    pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
    reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
    mul_1: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1052, code: fwd_pos_seq = torch.arange(beg, end, -1.0)
    iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    convert_element_type_2: "f64[1024]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float64);  iota_1 = None
    mul_2: "f64[1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, -1.0);  convert_element_type_2 = None
    add_1: "f64[1024]" = torch.ops.aten.add.Tensor(mul_2, 512);  mul_2 = None
    convert_element_type_3: "f32[1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1012, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
    unsqueeze: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
    permute_1: "f32[1024, 1]" = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None
    unsqueeze_1: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_1, 1);  mul_1 = None
    permute_2: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
    mul_3: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(permute_1, permute_2);  permute_1 = permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1013, code: pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    sin: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_3)
    cos: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_3);  mul_3 = None
    cat: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], 1);  sin = cos = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1014, code: pos_emb = pos_emb[:, None, :]
    slice_1: "f32[1024, 1024]" = torch.ops.aten.slice.Tensor(cat, 0, 0, 9223372036854775807);  cat = None
    unsqueeze_2: "f32[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
    slice_2: "f32[1024, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_2, 2, 0, 9223372036854775807);  unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1017, code: pos_emb = pos_emb.expand(-1, bsz, -1)
    expand: "f32[1024, 1, 1024]" = torch.ops.aten.expand.default(slice_2, [-1, 1, -1]);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1205, code: pos_emb = self.dropout(pos_emb)
    clone_1: "f32[1024, 1, 1024]" = torch.ops.aten.clone.default(expand);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_3: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(clone, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_3);  slice_3 = None
    alias_1: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_3: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone, 3)
    unsqueeze_4: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
    permute_3: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_4, [0, 1, 3, 4, 2]);  unsqueeze_4 = None
    unsqueeze_5: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 3);  arg0_1 = None
    unsqueeze_6: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
    permute_4: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_6, [3, 4, 1, 2, 0]);  unsqueeze_6 = None
    permute_5: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_3, [0, 4, 1, 2, 3]);  permute_3 = None
    view: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_5, [1, 512, 1024]);  permute_5 = None
    permute_6: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_4, [4, 1, 2, 3, 0]);  permute_4 = None
    view_1: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_6, [1, 1024, 1024]);  permute_6 = None
    bmm: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm, [512, 1, 1, 16, 64]);  bmm = None
    permute_7: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 2, 3, 4, 1]);  view_2 = None
    view_3: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_7, [512, 1, 16, 64]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_7: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone, 3)
    unsqueeze_8: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 4);  unsqueeze_7 = None
    permute_8: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_8, [0, 1, 3, 4, 2]);  unsqueeze_8 = None
    unsqueeze_9: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 3);  arg1_1 = None
    unsqueeze_10: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 4);  unsqueeze_9 = None
    permute_9: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_10, [3, 4, 1, 2, 0]);  unsqueeze_10 = None
    permute_10: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_8, [0, 4, 1, 2, 3]);  permute_8 = None
    view_4: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_10, [1, 512, 1024]);  permute_10 = None
    permute_11: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_9, [4, 1, 2, 3, 0]);  permute_9 = None
    view_5: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_11, [1, 1024, 1024]);  permute_11 = None
    bmm_1: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = view_5 = None
    view_6: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_1, [512, 1, 1, 16, 64]);  bmm_1 = None
    permute_12: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_6, [0, 2, 3, 4, 1]);  view_6 = None
    view_7: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_12, [512, 1, 16, 64]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_11: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone, 3)
    unsqueeze_12: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
    permute_13: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_12, [0, 1, 3, 4, 2]);  unsqueeze_12 = None
    unsqueeze_13: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, 3);  arg2_1 = None
    unsqueeze_14: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 4);  unsqueeze_13 = None
    permute_14: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_14, [3, 4, 1, 2, 0]);  unsqueeze_14 = None
    permute_15: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_13, [0, 4, 1, 2, 3]);  permute_13 = None
    view_8: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_15, [1, 512, 1024]);  permute_15 = None
    permute_16: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_14, [4, 1, 2, 3, 0]);  permute_14 = None
    view_9: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_16, [1, 1024, 1024]);  permute_16 = None
    bmm_2: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = view_9 = None
    view_10: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_2, [512, 1, 1, 16, 64]);  bmm_2 = None
    permute_17: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_10, [0, 2, 3, 4, 1]);  view_10 = None
    view_11: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_17, [512, 1, 16, 64]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_15: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_16: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 4);  unsqueeze_15 = None
    permute_18: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_16, [0, 1, 3, 4, 2]);  unsqueeze_16 = None
    unsqueeze_17: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, 3);  arg3_1 = None
    unsqueeze_18: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
    permute_19: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_18, [3, 4, 1, 2, 0]);  unsqueeze_18 = None
    permute_20: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_18, [0, 4, 1, 2, 3]);  permute_18 = None
    view_12: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_20, [1, 1024, 1024]);  permute_20 = None
    permute_21: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_19, [4, 1, 2, 3, 0]);  permute_19 = None
    view_13: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_21, [1, 1024, 1024]);  permute_21 = None
    bmm_3: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
    view_14: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_3, [1024, 1, 1, 16, 64]);  bmm_3 = None
    permute_22: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_14, [0, 2, 3, 4, 1]);  view_14 = None
    view_15: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_22, [1024, 1, 16, 64]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_2: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg4_1);  arg4_1 = None
    unsqueeze_19: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_2, 4);  add_2 = None
    permute_23: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_19, [1, 2, 0, 4, 3]);  unsqueeze_19 = None
    unsqueeze_20: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_7, 4);  view_7 = None
    permute_24: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_20, [1, 2, 4, 0, 3]);  unsqueeze_20 = None
    permute_25: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_23, [1, 2, 4, 0, 3]);  permute_23 = None
    view_16: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_25, [16, 512, 64]);  permute_25 = None
    permute_26: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_24, [1, 4, 0, 3, 2]);  permute_24 = None
    view_17: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_26, [16, 64, 512]);  permute_26 = None
    bmm_4: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = view_17 = None
    view_18: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_4, [16, 512, 1, 1, 512]);  bmm_4 = None
    permute_27: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_18, [3, 0, 1, 4, 2]);  view_18 = None
    view_19: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_27, [1, 16, 512, 512]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_3: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg5_1);  view_3 = arg5_1 = None
    unsqueeze_21: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_3, 4);  add_3 = None
    permute_28: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_21, [1, 2, 0, 4, 3]);  unsqueeze_21 = None
    unsqueeze_22: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_15, 4);  view_15 = None
    permute_29: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_22, [1, 2, 4, 0, 3]);  unsqueeze_22 = None
    permute_30: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_28, [1, 2, 4, 0, 3]);  permute_28 = None
    view_20: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_30, [16, 512, 64]);  permute_30 = None
    permute_31: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_29, [1, 4, 0, 3, 2]);  permute_29 = None
    view_21: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_31, [16, 64, 1024]);  permute_31 = None
    bmm_5: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_20, view_21);  view_20 = view_21 = None
    view_22: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_5, [16, 512, 1, 1, 1024]);  bmm_5 = None
    permute_32: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_22, [3, 0, 1, 4, 2]);  view_22 = None
    view_23: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_32, [1, 16, 512, 1024]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_24: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_23, [1, 16, 1024, 512]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_4: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_24, 0, 0, 9223372036854775807);  view_24 = None
    slice_5: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 9223372036854775807);  slice_4 = None
    slice_6: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_5, 2, 1, 9223372036854775807);  slice_5 = None
    slice_7: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 9223372036854775807);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_25: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_7, [1, 16, 512, 1023]);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_8: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_25, 0, 0, 9223372036854775807);  view_25 = None
    slice_9: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_8, 1, 0, 9223372036854775807);  slice_8 = None
    slice_10: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_9, 2, 0, 9223372036854775807);  slice_9 = None
    index: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_10, [None, None, None, iota_2]);  slice_10 = iota_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_4: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_19, index);  view_19 = index = None
    add_5: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_4, 0);  add_4 = None
    mul_4: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_5, 0.125);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_4, [3], True)
    sub: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_4, amax);  mul_4 = amax = None
    exp: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
    sum_1: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
    div_1: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_2: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_23: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_2, 4);  clone_2 = None
    permute_33: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_23, [2, 0, 1, 4, 3]);  unsqueeze_23 = None
    unsqueeze_24: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_11, 4);  view_11 = None
    permute_34: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_24, [4, 1, 2, 3, 0]);  unsqueeze_24 = None
    permute_35: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_33, [2, 0, 4, 1, 3]);  permute_33 = None
    view_26: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_35, [16, 512, 512]);  permute_35 = None
    permute_36: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_34, [2, 4, 1, 3, 0]);  permute_34 = None
    view_27: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_36, [16, 512, 64]);  permute_36 = None
    bmm_6: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_26, view_27);  view_26 = view_27 = None
    view_28: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_6, [16, 512, 1, 1, 64]);  bmm_6 = None
    permute_37: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_28, [1, 3, 0, 4, 2]);  view_28 = None
    view_29: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_37, [512, 1, 16, 64]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_25: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_29, 4);  view_29 = None
    permute_38: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_25, [0, 1, 4, 3, 2]);  unsqueeze_25 = None
    unsqueeze_26: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, 3);  arg6_1 = None
    unsqueeze_27: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 4);  unsqueeze_26 = None
    permute_39: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_27, [3, 4, 0, 2, 1]);  unsqueeze_27 = None
    permute_40: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_38, [0, 3, 4, 1, 2]);  permute_38 = None
    clone_3: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_30: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_3, [1, 512, 1024]);  clone_3 = None
    permute_41: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_39, [3, 4, 1, 2, 0]);  permute_39 = None
    clone_4: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_31: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_4, [1, 1024, 1024]);  clone_4 = None
    bmm_7: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_30, view_31);  view_30 = view_31 = None
    view_32: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_7, [512, 1, 1, 1, 1024]);  bmm_7 = None
    permute_42: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_32, [0, 3, 4, 1, 2]);  view_32 = None
    view_33: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_42, [512, 1, 1024]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_5: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_6: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_5, clone);  clone_5 = clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem: "f32[512, 1, 1]" = var_mean[0]
    getitem_1: "f32[512, 1, 1]" = var_mean[1];  var_mean = None
    add_7: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_1: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_1);  add_6 = getitem_1 = None
    mul_5: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
    mul_6: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_5, arg169_1);  mul_5 = arg169_1 = None
    add_8: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_6, arg170_1);  mul_6 = arg170_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_34: "f32[512, 1024]" = torch.ops.aten.view.default(add_8, [512, 1024])
    permute_43: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    addmm: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg172_1, view_34, permute_43);  arg172_1 = view_34 = permute_43 = None
    view_35: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm, [512, 1, 4096]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_7: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    mul_8: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476);  view_35 = None
    erf: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_9: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_6: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_36: "f32[512, 4096]" = torch.ops.aten.view.default(clone_6, [512, 4096]);  clone_6 = None
    permute_44: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_1: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg174_1, view_36, permute_44);  arg174_1 = view_36 = permute_44 = None
    view_37: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_1, [512, 1, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_7: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_37);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_10: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_7, add_8);  clone_7 = add_8 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_2: "f32[512, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[512, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_11: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_10, getitem_3);  add_10 = getitem_3 = None
    mul_10: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_11: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_10, arg175_1);  mul_10 = arg175_1 = None
    add_12: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_11, arg176_1);  mul_11 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_11: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_12, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_2: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_11);  slice_11 = None
    alias_3: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_28: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_12, 3)
    unsqueeze_29: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 4);  unsqueeze_28 = None
    permute_45: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_29, [0, 1, 3, 4, 2]);  unsqueeze_29 = None
    unsqueeze_30: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, 3);  arg7_1 = None
    unsqueeze_31: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 4);  unsqueeze_30 = None
    permute_46: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_31, [3, 4, 1, 2, 0]);  unsqueeze_31 = None
    permute_47: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_45, [0, 4, 1, 2, 3]);  permute_45 = None
    view_38: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_47, [1, 512, 1024]);  permute_47 = None
    permute_48: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_46, [4, 1, 2, 3, 0]);  permute_46 = None
    view_39: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_48, [1, 1024, 1024]);  permute_48 = None
    bmm_8: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_38, view_39);  view_38 = view_39 = None
    view_40: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_8, [512, 1, 1, 16, 64]);  bmm_8 = None
    permute_49: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_40, [0, 2, 3, 4, 1]);  view_40 = None
    view_41: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_49, [512, 1, 16, 64]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_32: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_12, 3)
    unsqueeze_33: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 4);  unsqueeze_32 = None
    permute_50: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_33, [0, 1, 3, 4, 2]);  unsqueeze_33 = None
    unsqueeze_34: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 3);  arg8_1 = None
    unsqueeze_35: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 4);  unsqueeze_34 = None
    permute_51: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_35, [3, 4, 1, 2, 0]);  unsqueeze_35 = None
    permute_52: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_50, [0, 4, 1, 2, 3]);  permute_50 = None
    view_42: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_52, [1, 512, 1024]);  permute_52 = None
    permute_53: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_51, [4, 1, 2, 3, 0]);  permute_51 = None
    view_43: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_53, [1, 1024, 1024]);  permute_53 = None
    bmm_9: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_42, view_43);  view_42 = view_43 = None
    view_44: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_9, [512, 1, 1, 16, 64]);  bmm_9 = None
    permute_54: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_44, [0, 2, 3, 4, 1]);  view_44 = None
    view_45: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_54, [512, 1, 16, 64]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_36: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_12, 3)
    unsqueeze_37: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 4);  unsqueeze_36 = None
    permute_55: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_37, [0, 1, 3, 4, 2]);  unsqueeze_37 = None
    unsqueeze_38: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, 3);  arg9_1 = None
    unsqueeze_39: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 4);  unsqueeze_38 = None
    permute_56: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_39, [3, 4, 1, 2, 0]);  unsqueeze_39 = None
    permute_57: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_55, [0, 4, 1, 2, 3]);  permute_55 = None
    view_46: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_57, [1, 512, 1024]);  permute_57 = None
    permute_58: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_56, [4, 1, 2, 3, 0]);  permute_56 = None
    view_47: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_58, [1, 1024, 1024]);  permute_58 = None
    bmm_10: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_46, view_47);  view_46 = view_47 = None
    view_48: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_10, [512, 1, 1, 16, 64]);  bmm_10 = None
    permute_59: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_48, [0, 2, 3, 4, 1]);  view_48 = None
    view_49: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_59, [512, 1, 16, 64]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_40: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_41: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 4);  unsqueeze_40 = None
    permute_60: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_41, [0, 1, 3, 4, 2]);  unsqueeze_41 = None
    unsqueeze_42: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, 3);  arg10_1 = None
    unsqueeze_43: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 4);  unsqueeze_42 = None
    permute_61: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_43, [3, 4, 1, 2, 0]);  unsqueeze_43 = None
    permute_62: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_60, [0, 4, 1, 2, 3]);  permute_60 = None
    view_50: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_62, [1, 1024, 1024]);  permute_62 = None
    permute_63: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_61, [4, 1, 2, 3, 0]);  permute_61 = None
    view_51: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_63, [1, 1024, 1024]);  permute_63 = None
    bmm_11: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_50, view_51);  view_50 = view_51 = None
    view_52: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_11, [1024, 1, 1, 16, 64]);  bmm_11 = None
    permute_64: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_52, [0, 2, 3, 4, 1]);  view_52 = None
    view_53: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_64, [1024, 1, 16, 64]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_13: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_41, arg11_1);  arg11_1 = None
    unsqueeze_44: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_13, 4);  add_13 = None
    permute_65: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_44, [1, 2, 0, 4, 3]);  unsqueeze_44 = None
    unsqueeze_45: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_45, 4);  view_45 = None
    permute_66: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_45, [1, 2, 4, 0, 3]);  unsqueeze_45 = None
    permute_67: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_65, [1, 2, 4, 0, 3]);  permute_65 = None
    view_54: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_67, [16, 512, 64]);  permute_67 = None
    permute_68: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_66, [1, 4, 0, 3, 2]);  permute_66 = None
    view_55: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_68, [16, 64, 512]);  permute_68 = None
    bmm_12: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_54, view_55);  view_54 = view_55 = None
    view_56: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_12, [16, 512, 1, 1, 512]);  bmm_12 = None
    permute_69: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_56, [3, 0, 1, 4, 2]);  view_56 = None
    view_57: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_69, [1, 16, 512, 512]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_14: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_41, arg12_1);  view_41 = arg12_1 = None
    unsqueeze_46: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_14, 4);  add_14 = None
    permute_70: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_46, [1, 2, 0, 4, 3]);  unsqueeze_46 = None
    unsqueeze_47: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_53, 4);  view_53 = None
    permute_71: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_47, [1, 2, 4, 0, 3]);  unsqueeze_47 = None
    permute_72: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_70, [1, 2, 4, 0, 3]);  permute_70 = None
    view_58: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_72, [16, 512, 64]);  permute_72 = None
    permute_73: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_71, [1, 4, 0, 3, 2]);  permute_71 = None
    view_59: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_73, [16, 64, 1024]);  permute_73 = None
    bmm_13: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
    view_60: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_13, [16, 512, 1, 1, 1024]);  bmm_13 = None
    permute_74: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_60, [3, 0, 1, 4, 2]);  view_60 = None
    view_61: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_74, [1, 16, 512, 1024]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_62: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_61, [1, 16, 1024, 512]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_12: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_62, 0, 0, 9223372036854775807);  view_62 = None
    slice_13: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_12, 1, 0, 9223372036854775807);  slice_12 = None
    slice_14: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_13, 2, 1, 9223372036854775807);  slice_13 = None
    slice_15: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 9223372036854775807);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_63: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_15, [1, 16, 512, 1023]);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_3: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_16: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_63, 0, 0, 9223372036854775807);  view_63 = None
    slice_17: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 9223372036854775807);  slice_16 = None
    slice_18: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_17, 2, 0, 9223372036854775807);  slice_17 = None
    index_1: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_18, [None, None, None, iota_3]);  slice_18 = iota_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_15: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_57, index_1);  view_57 = index_1 = None
    add_16: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_15, 0);  add_15 = None
    mul_12: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_16, 0.125);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_1: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_12, [3], True)
    sub_3: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_12, amax_1);  mul_12 = amax_1 = None
    exp_1: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [3], True)
    div_2: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_8: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_48: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_8, 4);  clone_8 = None
    permute_75: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_48, [2, 0, 1, 4, 3]);  unsqueeze_48 = None
    unsqueeze_49: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_49, 4);  view_49 = None
    permute_76: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_49, [4, 1, 2, 3, 0]);  unsqueeze_49 = None
    permute_77: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_75, [2, 0, 4, 1, 3]);  permute_75 = None
    view_64: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_77, [16, 512, 512]);  permute_77 = None
    permute_78: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_76, [2, 4, 1, 3, 0]);  permute_76 = None
    view_65: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_78, [16, 512, 64]);  permute_78 = None
    bmm_14: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_64, view_65);  view_64 = view_65 = None
    view_66: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_14, [16, 512, 1, 1, 64]);  bmm_14 = None
    permute_79: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_66, [1, 3, 0, 4, 2]);  view_66 = None
    view_67: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_79, [512, 1, 16, 64]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_50: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_67, 4);  view_67 = None
    permute_80: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_50, [0, 1, 4, 3, 2]);  unsqueeze_50 = None
    unsqueeze_51: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, 3);  arg13_1 = None
    unsqueeze_52: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 4);  unsqueeze_51 = None
    permute_81: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_52, [3, 4, 0, 2, 1]);  unsqueeze_52 = None
    permute_82: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_80, [0, 3, 4, 1, 2]);  permute_80 = None
    clone_9: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_68: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_9, [1, 512, 1024]);  clone_9 = None
    permute_83: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_81, [3, 4, 1, 2, 0]);  permute_81 = None
    clone_10: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    view_69: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_10, [1, 1024, 1024]);  clone_10 = None
    bmm_15: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_68, view_69);  view_68 = view_69 = None
    view_70: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_15, [512, 1, 1, 1, 1024]);  bmm_15 = None
    permute_84: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_70, [0, 3, 4, 1, 2]);  view_70 = None
    view_71: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_84, [512, 1, 1024]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_11: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_71);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_17: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_11, add_12);  clone_11 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_4: "f32[512, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[512, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_18: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_4: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_5);  add_17 = getitem_5 = None
    mul_13: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
    mul_14: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_13, arg177_1);  mul_13 = arg177_1 = None
    add_19: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_14, arg178_1);  mul_14 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_72: "f32[512, 1024]" = torch.ops.aten.view.default(add_19, [512, 1024])
    permute_85: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_2: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg180_1, view_72, permute_85);  arg180_1 = view_72 = permute_85 = None
    view_73: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_2, [512, 1, 4096]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_15: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_73, 0.5)
    mul_16: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476);  view_73 = None
    erf_1: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
    add_20: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_17: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_15, add_20);  mul_15 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_12: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_17);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_74: "f32[512, 4096]" = torch.ops.aten.view.default(clone_12, [512, 4096]);  clone_12 = None
    permute_86: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_3: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg182_1, view_74, permute_86);  arg182_1 = view_74 = permute_86 = None
    view_75: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_3, [512, 1, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_13: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_75);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_21: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_13, add_19);  clone_13 = add_19 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_6: "f32[512, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[512, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_22: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_5: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_21, getitem_7);  add_21 = getitem_7 = None
    mul_18: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_19: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_18, arg183_1);  mul_18 = arg183_1 = None
    add_23: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_19, arg184_1);  mul_19 = arg184_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_19: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_23, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_4: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_19);  slice_19 = None
    alias_5: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_53: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_23, 3)
    unsqueeze_54: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 4);  unsqueeze_53 = None
    permute_87: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_54, [0, 1, 3, 4, 2]);  unsqueeze_54 = None
    unsqueeze_55: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, 3);  arg14_1 = None
    unsqueeze_56: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
    permute_88: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_56, [3, 4, 1, 2, 0]);  unsqueeze_56 = None
    permute_89: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_87, [0, 4, 1, 2, 3]);  permute_87 = None
    view_76: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_89, [1, 512, 1024]);  permute_89 = None
    permute_90: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_88, [4, 1, 2, 3, 0]);  permute_88 = None
    view_77: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_90, [1, 1024, 1024]);  permute_90 = None
    bmm_16: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_76, view_77);  view_76 = view_77 = None
    view_78: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_16, [512, 1, 1, 16, 64]);  bmm_16 = None
    permute_91: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_78, [0, 2, 3, 4, 1]);  view_78 = None
    view_79: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_91, [512, 1, 16, 64]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_57: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_23, 3)
    unsqueeze_58: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 4);  unsqueeze_57 = None
    permute_92: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_58, [0, 1, 3, 4, 2]);  unsqueeze_58 = None
    unsqueeze_59: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, 3);  arg15_1 = None
    unsqueeze_60: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 4);  unsqueeze_59 = None
    permute_93: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_60, [3, 4, 1, 2, 0]);  unsqueeze_60 = None
    permute_94: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_92, [0, 4, 1, 2, 3]);  permute_92 = None
    view_80: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_94, [1, 512, 1024]);  permute_94 = None
    permute_95: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_93, [4, 1, 2, 3, 0]);  permute_93 = None
    view_81: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_95, [1, 1024, 1024]);  permute_95 = None
    bmm_17: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
    view_82: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_17, [512, 1, 1, 16, 64]);  bmm_17 = None
    permute_96: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_82, [0, 2, 3, 4, 1]);  view_82 = None
    view_83: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_96, [512, 1, 16, 64]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_61: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_23, 3)
    unsqueeze_62: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 4);  unsqueeze_61 = None
    permute_97: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_62, [0, 1, 3, 4, 2]);  unsqueeze_62 = None
    unsqueeze_63: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, 3);  arg16_1 = None
    unsqueeze_64: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 4);  unsqueeze_63 = None
    permute_98: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_64, [3, 4, 1, 2, 0]);  unsqueeze_64 = None
    permute_99: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_97, [0, 4, 1, 2, 3]);  permute_97 = None
    view_84: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_99, [1, 512, 1024]);  permute_99 = None
    permute_100: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_98, [4, 1, 2, 3, 0]);  permute_98 = None
    view_85: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_100, [1, 1024, 1024]);  permute_100 = None
    bmm_18: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_84, view_85);  view_84 = view_85 = None
    view_86: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_18, [512, 1, 1, 16, 64]);  bmm_18 = None
    permute_101: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_86, [0, 2, 3, 4, 1]);  view_86 = None
    view_87: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_101, [512, 1, 16, 64]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_65: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_66: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 4);  unsqueeze_65 = None
    permute_102: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_66, [0, 1, 3, 4, 2]);  unsqueeze_66 = None
    unsqueeze_67: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, 3);  arg17_1 = None
    unsqueeze_68: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 4);  unsqueeze_67 = None
    permute_103: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_68, [3, 4, 1, 2, 0]);  unsqueeze_68 = None
    permute_104: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_102, [0, 4, 1, 2, 3]);  permute_102 = None
    view_88: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_104, [1, 1024, 1024]);  permute_104 = None
    permute_105: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_103, [4, 1, 2, 3, 0]);  permute_103 = None
    view_89: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_105, [1, 1024, 1024]);  permute_105 = None
    bmm_19: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_88, view_89);  view_88 = view_89 = None
    view_90: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_19, [1024, 1, 1, 16, 64]);  bmm_19 = None
    permute_106: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_90, [0, 2, 3, 4, 1]);  view_90 = None
    view_91: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_106, [1024, 1, 16, 64]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_24: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_79, arg18_1);  arg18_1 = None
    unsqueeze_69: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_24, 4);  add_24 = None
    permute_107: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_69, [1, 2, 0, 4, 3]);  unsqueeze_69 = None
    unsqueeze_70: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_83, 4);  view_83 = None
    permute_108: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_70, [1, 2, 4, 0, 3]);  unsqueeze_70 = None
    permute_109: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_107, [1, 2, 4, 0, 3]);  permute_107 = None
    view_92: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_109, [16, 512, 64]);  permute_109 = None
    permute_110: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_108, [1, 4, 0, 3, 2]);  permute_108 = None
    view_93: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_110, [16, 64, 512]);  permute_110 = None
    bmm_20: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_92, view_93);  view_92 = view_93 = None
    view_94: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_20, [16, 512, 1, 1, 512]);  bmm_20 = None
    permute_111: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_94, [3, 0, 1, 4, 2]);  view_94 = None
    view_95: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_111, [1, 16, 512, 512]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_25: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_79, arg19_1);  view_79 = arg19_1 = None
    unsqueeze_71: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_25, 4);  add_25 = None
    permute_112: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_71, [1, 2, 0, 4, 3]);  unsqueeze_71 = None
    unsqueeze_72: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_91, 4);  view_91 = None
    permute_113: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_72, [1, 2, 4, 0, 3]);  unsqueeze_72 = None
    permute_114: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_112, [1, 2, 4, 0, 3]);  permute_112 = None
    view_96: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_114, [16, 512, 64]);  permute_114 = None
    permute_115: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_113, [1, 4, 0, 3, 2]);  permute_113 = None
    view_97: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_115, [16, 64, 1024]);  permute_115 = None
    bmm_21: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_96, view_97);  view_96 = view_97 = None
    view_98: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_21, [16, 512, 1, 1, 1024]);  bmm_21 = None
    permute_116: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_98, [3, 0, 1, 4, 2]);  view_98 = None
    view_99: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_116, [1, 16, 512, 1024]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_100: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_99, [1, 16, 1024, 512]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_20: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_100, 0, 0, 9223372036854775807);  view_100 = None
    slice_21: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_20, 1, 0, 9223372036854775807);  slice_20 = None
    slice_22: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_21, 2, 1, 9223372036854775807);  slice_21 = None
    slice_23: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 9223372036854775807);  slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_101: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_23, [1, 16, 512, 1023]);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_4: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_24: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_101, 0, 0, 9223372036854775807);  view_101 = None
    slice_25: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_24, 1, 0, 9223372036854775807);  slice_24 = None
    slice_26: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_25, 2, 0, 9223372036854775807);  slice_25 = None
    index_2: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_26, [None, None, None, iota_4]);  slice_26 = iota_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_26: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_95, index_2);  view_95 = index_2 = None
    add_27: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_26, 0);  add_26 = None
    mul_20: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_27, 0.125);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_2: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_20, [3], True)
    sub_6: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_20, amax_2);  mul_20 = amax_2 = None
    exp_2: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_3: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [3], True)
    div_3: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_14: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_73: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_14, 4);  clone_14 = None
    permute_117: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_73, [2, 0, 1, 4, 3]);  unsqueeze_73 = None
    unsqueeze_74: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_87, 4);  view_87 = None
    permute_118: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_74, [4, 1, 2, 3, 0]);  unsqueeze_74 = None
    permute_119: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_117, [2, 0, 4, 1, 3]);  permute_117 = None
    view_102: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_119, [16, 512, 512]);  permute_119 = None
    permute_120: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_118, [2, 4, 1, 3, 0]);  permute_118 = None
    view_103: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_120, [16, 512, 64]);  permute_120 = None
    bmm_22: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
    view_104: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_22, [16, 512, 1, 1, 64]);  bmm_22 = None
    permute_121: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_104, [1, 3, 0, 4, 2]);  view_104 = None
    view_105: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_121, [512, 1, 16, 64]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_75: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_105, 4);  view_105 = None
    permute_122: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 3, 2]);  unsqueeze_75 = None
    unsqueeze_76: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, 3);  arg20_1 = None
    unsqueeze_77: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 4);  unsqueeze_76 = None
    permute_123: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_77, [3, 4, 0, 2, 1]);  unsqueeze_77 = None
    permute_124: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_122, [0, 3, 4, 1, 2]);  permute_122 = None
    clone_15: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    view_106: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_15, [1, 512, 1024]);  clone_15 = None
    permute_125: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_123, [3, 4, 1, 2, 0]);  permute_123 = None
    clone_16: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    view_107: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_16, [1, 1024, 1024]);  clone_16 = None
    bmm_23: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
    view_108: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_23, [512, 1, 1, 1, 1024]);  bmm_23 = None
    permute_126: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_108, [0, 3, 4, 1, 2]);  view_108 = None
    view_109: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_126, [512, 1, 1024]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_17: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_109);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_28: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_17, add_23);  clone_17 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_8: "f32[512, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[512, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_29: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_7: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_28, getitem_9);  add_28 = getitem_9 = None
    mul_21: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
    mul_22: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_21, arg185_1);  mul_21 = arg185_1 = None
    add_30: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_22, arg186_1);  mul_22 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_110: "f32[512, 1024]" = torch.ops.aten.view.default(add_30, [512, 1024])
    permute_127: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    addmm_4: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg188_1, view_110, permute_127);  arg188_1 = view_110 = permute_127 = None
    view_111: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_4, [512, 1, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_23: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    mul_24: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
    erf_2: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_31: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_23, add_31);  mul_23 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_18: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_25);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_112: "f32[512, 4096]" = torch.ops.aten.view.default(clone_18, [512, 4096]);  clone_18 = None
    permute_128: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_5: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg190_1, view_112, permute_128);  arg190_1 = view_112 = permute_128 = None
    view_113: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_5, [512, 1, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_19: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_113);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_32: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_19, add_30);  clone_19 = add_30 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_10: "f32[512, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[512, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_33: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_8: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_32, getitem_11);  add_32 = getitem_11 = None
    mul_26: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_27: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_26, arg191_1);  mul_26 = arg191_1 = None
    add_34: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_27, arg192_1);  mul_27 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_27: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_34, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_6: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_27);  slice_27 = None
    alias_7: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_78: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_34, 3)
    unsqueeze_79: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 4);  unsqueeze_78 = None
    permute_129: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_79, [0, 1, 3, 4, 2]);  unsqueeze_79 = None
    unsqueeze_80: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, 3);  arg21_1 = None
    unsqueeze_81: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 4);  unsqueeze_80 = None
    permute_130: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_81, [3, 4, 1, 2, 0]);  unsqueeze_81 = None
    permute_131: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_129, [0, 4, 1, 2, 3]);  permute_129 = None
    view_114: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_131, [1, 512, 1024]);  permute_131 = None
    permute_132: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_130, [4, 1, 2, 3, 0]);  permute_130 = None
    view_115: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_132, [1, 1024, 1024]);  permute_132 = None
    bmm_24: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
    view_116: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_24, [512, 1, 1, 16, 64]);  bmm_24 = None
    permute_133: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_116, [0, 2, 3, 4, 1]);  view_116 = None
    view_117: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_133, [512, 1, 16, 64]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_82: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_34, 3)
    unsqueeze_83: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 4);  unsqueeze_82 = None
    permute_134: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_83, [0, 1, 3, 4, 2]);  unsqueeze_83 = None
    unsqueeze_84: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, 3);  arg22_1 = None
    unsqueeze_85: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 4);  unsqueeze_84 = None
    permute_135: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_85, [3, 4, 1, 2, 0]);  unsqueeze_85 = None
    permute_136: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_134, [0, 4, 1, 2, 3]);  permute_134 = None
    view_118: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_136, [1, 512, 1024]);  permute_136 = None
    permute_137: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_135, [4, 1, 2, 3, 0]);  permute_135 = None
    view_119: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_137, [1, 1024, 1024]);  permute_137 = None
    bmm_25: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_118, view_119);  view_118 = view_119 = None
    view_120: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_25, [512, 1, 1, 16, 64]);  bmm_25 = None
    permute_138: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_120, [0, 2, 3, 4, 1]);  view_120 = None
    view_121: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_138, [512, 1, 16, 64]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_86: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_34, 3)
    unsqueeze_87: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, 4);  unsqueeze_86 = None
    permute_139: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_87, [0, 1, 3, 4, 2]);  unsqueeze_87 = None
    unsqueeze_88: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, 3);  arg23_1 = None
    unsqueeze_89: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 4);  unsqueeze_88 = None
    permute_140: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_89, [3, 4, 1, 2, 0]);  unsqueeze_89 = None
    permute_141: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_139, [0, 4, 1, 2, 3]);  permute_139 = None
    view_122: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_141, [1, 512, 1024]);  permute_141 = None
    permute_142: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_140, [4, 1, 2, 3, 0]);  permute_140 = None
    view_123: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_142, [1, 1024, 1024]);  permute_142 = None
    bmm_26: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
    view_124: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_26, [512, 1, 1, 16, 64]);  bmm_26 = None
    permute_143: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_124, [0, 2, 3, 4, 1]);  view_124 = None
    view_125: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_143, [512, 1, 16, 64]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_90: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_91: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 4);  unsqueeze_90 = None
    permute_144: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_91, [0, 1, 3, 4, 2]);  unsqueeze_91 = None
    unsqueeze_92: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, 3);  arg24_1 = None
    unsqueeze_93: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 4);  unsqueeze_92 = None
    permute_145: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_93, [3, 4, 1, 2, 0]);  unsqueeze_93 = None
    permute_146: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_144, [0, 4, 1, 2, 3]);  permute_144 = None
    view_126: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_146, [1, 1024, 1024]);  permute_146 = None
    permute_147: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_145, [4, 1, 2, 3, 0]);  permute_145 = None
    view_127: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_147, [1, 1024, 1024]);  permute_147 = None
    bmm_27: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_126, view_127);  view_126 = view_127 = None
    view_128: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_27, [1024, 1, 1, 16, 64]);  bmm_27 = None
    permute_148: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_128, [0, 2, 3, 4, 1]);  view_128 = None
    view_129: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_148, [1024, 1, 16, 64]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_35: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_117, arg25_1);  arg25_1 = None
    unsqueeze_94: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_35, 4);  add_35 = None
    permute_149: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_94, [1, 2, 0, 4, 3]);  unsqueeze_94 = None
    unsqueeze_95: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_121, 4);  view_121 = None
    permute_150: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_95, [1, 2, 4, 0, 3]);  unsqueeze_95 = None
    permute_151: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_149, [1, 2, 4, 0, 3]);  permute_149 = None
    view_130: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_151, [16, 512, 64]);  permute_151 = None
    permute_152: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_150, [1, 4, 0, 3, 2]);  permute_150 = None
    view_131: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_152, [16, 64, 512]);  permute_152 = None
    bmm_28: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
    view_132: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_28, [16, 512, 1, 1, 512]);  bmm_28 = None
    permute_153: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_132, [3, 0, 1, 4, 2]);  view_132 = None
    view_133: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_153, [1, 16, 512, 512]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_36: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_117, arg26_1);  view_117 = arg26_1 = None
    unsqueeze_96: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_36, 4);  add_36 = None
    permute_154: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_96, [1, 2, 0, 4, 3]);  unsqueeze_96 = None
    unsqueeze_97: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_129, 4);  view_129 = None
    permute_155: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_97, [1, 2, 4, 0, 3]);  unsqueeze_97 = None
    permute_156: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_154, [1, 2, 4, 0, 3]);  permute_154 = None
    view_134: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_156, [16, 512, 64]);  permute_156 = None
    permute_157: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_155, [1, 4, 0, 3, 2]);  permute_155 = None
    view_135: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_157, [16, 64, 1024]);  permute_157 = None
    bmm_29: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_134, view_135);  view_134 = view_135 = None
    view_136: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_29, [16, 512, 1, 1, 1024]);  bmm_29 = None
    permute_158: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_136, [3, 0, 1, 4, 2]);  view_136 = None
    view_137: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_158, [1, 16, 512, 1024]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_138: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_137, [1, 16, 1024, 512]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_28: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_138, 0, 0, 9223372036854775807);  view_138 = None
    slice_29: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_28, 1, 0, 9223372036854775807);  slice_28 = None
    slice_30: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_29, 2, 1, 9223372036854775807);  slice_29 = None
    slice_31: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_30, 3, 0, 9223372036854775807);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_139: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_31, [1, 16, 512, 1023]);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_5: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_32: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_139, 0, 0, 9223372036854775807);  view_139 = None
    slice_33: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_32, 1, 0, 9223372036854775807);  slice_32 = None
    slice_34: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_33, 2, 0, 9223372036854775807);  slice_33 = None
    index_3: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_34, [None, None, None, iota_5]);  slice_34 = iota_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_37: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_133, index_3);  view_133 = index_3 = None
    add_38: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_37, 0);  add_37 = None
    mul_28: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_38, 0.125);  add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_3: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_28, [3], True)
    sub_9: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_28, amax_3);  mul_28 = amax_3 = None
    exp_3: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_4: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [3], True)
    div_4: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_20: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_98: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_20, 4);  clone_20 = None
    permute_159: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_98, [2, 0, 1, 4, 3]);  unsqueeze_98 = None
    unsqueeze_99: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_125, 4);  view_125 = None
    permute_160: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_99, [4, 1, 2, 3, 0]);  unsqueeze_99 = None
    permute_161: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_159, [2, 0, 4, 1, 3]);  permute_159 = None
    view_140: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_161, [16, 512, 512]);  permute_161 = None
    permute_162: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_160, [2, 4, 1, 3, 0]);  permute_160 = None
    view_141: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_162, [16, 512, 64]);  permute_162 = None
    bmm_30: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
    view_142: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_30, [16, 512, 1, 1, 64]);  bmm_30 = None
    permute_163: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_142, [1, 3, 0, 4, 2]);  view_142 = None
    view_143: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_163, [512, 1, 16, 64]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_100: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_143, 4);  view_143 = None
    permute_164: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_100, [0, 1, 4, 3, 2]);  unsqueeze_100 = None
    unsqueeze_101: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg27_1, 3);  arg27_1 = None
    unsqueeze_102: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 4);  unsqueeze_101 = None
    permute_165: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_102, [3, 4, 0, 2, 1]);  unsqueeze_102 = None
    permute_166: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_164, [0, 3, 4, 1, 2]);  permute_164 = None
    clone_21: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    view_144: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_21, [1, 512, 1024]);  clone_21 = None
    permute_167: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_165, [3, 4, 1, 2, 0]);  permute_165 = None
    clone_22: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    view_145: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_22, [1, 1024, 1024]);  clone_22 = None
    bmm_31: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
    view_146: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_31, [512, 1, 1, 1, 1024]);  bmm_31 = None
    permute_168: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_146, [0, 3, 4, 1, 2]);  view_146 = None
    view_147: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_168, [512, 1, 1024]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_23: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_147);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_39: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_23, add_34);  clone_23 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_12: "f32[512, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[512, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_40: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_10: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_39, getitem_13);  add_39 = getitem_13 = None
    mul_29: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
    mul_30: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_29, arg193_1);  mul_29 = arg193_1 = None
    add_41: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_30, arg194_1);  mul_30 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_148: "f32[512, 1024]" = torch.ops.aten.view.default(add_41, [512, 1024])
    permute_169: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_6: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg196_1, view_148, permute_169);  arg196_1 = view_148 = permute_169 = None
    view_149: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_6, [512, 1, 4096]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_31: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
    mul_32: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
    erf_3: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_32);  mul_32 = None
    add_42: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_33: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_31, add_42);  mul_31 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_24: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_33);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_150: "f32[512, 4096]" = torch.ops.aten.view.default(clone_24, [512, 4096]);  clone_24 = None
    permute_170: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    addmm_7: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg198_1, view_150, permute_170);  arg198_1 = view_150 = permute_170 = None
    view_151: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_7, [512, 1, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_25: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_43: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_25, add_41);  clone_25 = add_41 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_14: "f32[512, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[512, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_44: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_11: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_43, getitem_15);  add_43 = getitem_15 = None
    mul_34: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_35: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_34, arg199_1);  mul_34 = arg199_1 = None
    add_45: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_35, arg200_1);  mul_35 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_35: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_45, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_8: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_35);  slice_35 = None
    alias_9: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_103: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_45, 3)
    unsqueeze_104: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 4);  unsqueeze_103 = None
    permute_171: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_104, [0, 1, 3, 4, 2]);  unsqueeze_104 = None
    unsqueeze_105: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, 3);  arg28_1 = None
    unsqueeze_106: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 4);  unsqueeze_105 = None
    permute_172: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_106, [3, 4, 1, 2, 0]);  unsqueeze_106 = None
    permute_173: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_171, [0, 4, 1, 2, 3]);  permute_171 = None
    view_152: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_173, [1, 512, 1024]);  permute_173 = None
    permute_174: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_172, [4, 1, 2, 3, 0]);  permute_172 = None
    view_153: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_174, [1, 1024, 1024]);  permute_174 = None
    bmm_32: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_152, view_153);  view_152 = view_153 = None
    view_154: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_32, [512, 1, 1, 16, 64]);  bmm_32 = None
    permute_175: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_154, [0, 2, 3, 4, 1]);  view_154 = None
    view_155: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_175, [512, 1, 16, 64]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_107: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_45, 3)
    unsqueeze_108: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 4);  unsqueeze_107 = None
    permute_176: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_108, [0, 1, 3, 4, 2]);  unsqueeze_108 = None
    unsqueeze_109: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, 3);  arg29_1 = None
    unsqueeze_110: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 4);  unsqueeze_109 = None
    permute_177: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_110, [3, 4, 1, 2, 0]);  unsqueeze_110 = None
    permute_178: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_176, [0, 4, 1, 2, 3]);  permute_176 = None
    view_156: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_178, [1, 512, 1024]);  permute_178 = None
    permute_179: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_177, [4, 1, 2, 3, 0]);  permute_177 = None
    view_157: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_179, [1, 1024, 1024]);  permute_179 = None
    bmm_33: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_156, view_157);  view_156 = view_157 = None
    view_158: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_33, [512, 1, 1, 16, 64]);  bmm_33 = None
    permute_180: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_158, [0, 2, 3, 4, 1]);  view_158 = None
    view_159: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_180, [512, 1, 16, 64]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_111: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_45, 3)
    unsqueeze_112: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 4);  unsqueeze_111 = None
    permute_181: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_112, [0, 1, 3, 4, 2]);  unsqueeze_112 = None
    unsqueeze_113: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, 3);  arg30_1 = None
    unsqueeze_114: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 4);  unsqueeze_113 = None
    permute_182: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_114, [3, 4, 1, 2, 0]);  unsqueeze_114 = None
    permute_183: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_181, [0, 4, 1, 2, 3]);  permute_181 = None
    view_160: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_183, [1, 512, 1024]);  permute_183 = None
    permute_184: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_182, [4, 1, 2, 3, 0]);  permute_182 = None
    view_161: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_184, [1, 1024, 1024]);  permute_184 = None
    bmm_34: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
    view_162: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_34, [512, 1, 1, 16, 64]);  bmm_34 = None
    permute_185: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_162, [0, 2, 3, 4, 1]);  view_162 = None
    view_163: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_185, [512, 1, 16, 64]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_115: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_116: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 4);  unsqueeze_115 = None
    permute_186: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_116, [0, 1, 3, 4, 2]);  unsqueeze_116 = None
    unsqueeze_117: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, 3);  arg31_1 = None
    unsqueeze_118: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 4);  unsqueeze_117 = None
    permute_187: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_118, [3, 4, 1, 2, 0]);  unsqueeze_118 = None
    permute_188: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_186, [0, 4, 1, 2, 3]);  permute_186 = None
    view_164: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_188, [1, 1024, 1024]);  permute_188 = None
    permute_189: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_187, [4, 1, 2, 3, 0]);  permute_187 = None
    view_165: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_189, [1, 1024, 1024]);  permute_189 = None
    bmm_35: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_164, view_165);  view_164 = view_165 = None
    view_166: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_35, [1024, 1, 1, 16, 64]);  bmm_35 = None
    permute_190: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_166, [0, 2, 3, 4, 1]);  view_166 = None
    view_167: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_190, [1024, 1, 16, 64]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_46: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_155, arg32_1);  arg32_1 = None
    unsqueeze_119: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_46, 4);  add_46 = None
    permute_191: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_119, [1, 2, 0, 4, 3]);  unsqueeze_119 = None
    unsqueeze_120: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_159, 4);  view_159 = None
    permute_192: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_120, [1, 2, 4, 0, 3]);  unsqueeze_120 = None
    permute_193: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_191, [1, 2, 4, 0, 3]);  permute_191 = None
    view_168: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_193, [16, 512, 64]);  permute_193 = None
    permute_194: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_192, [1, 4, 0, 3, 2]);  permute_192 = None
    view_169: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_194, [16, 64, 512]);  permute_194 = None
    bmm_36: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
    view_170: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_36, [16, 512, 1, 1, 512]);  bmm_36 = None
    permute_195: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_170, [3, 0, 1, 4, 2]);  view_170 = None
    view_171: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_195, [1, 16, 512, 512]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_47: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_155, arg33_1);  view_155 = arg33_1 = None
    unsqueeze_121: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_47, 4);  add_47 = None
    permute_196: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_121, [1, 2, 0, 4, 3]);  unsqueeze_121 = None
    unsqueeze_122: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_167, 4);  view_167 = None
    permute_197: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_122, [1, 2, 4, 0, 3]);  unsqueeze_122 = None
    permute_198: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_196, [1, 2, 4, 0, 3]);  permute_196 = None
    view_172: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_198, [16, 512, 64]);  permute_198 = None
    permute_199: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_197, [1, 4, 0, 3, 2]);  permute_197 = None
    view_173: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_199, [16, 64, 1024]);  permute_199 = None
    bmm_37: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_172, view_173);  view_172 = view_173 = None
    view_174: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_37, [16, 512, 1, 1, 1024]);  bmm_37 = None
    permute_200: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_174, [3, 0, 1, 4, 2]);  view_174 = None
    view_175: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_200, [1, 16, 512, 1024]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_176: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_175, [1, 16, 1024, 512]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_36: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_176, 0, 0, 9223372036854775807);  view_176 = None
    slice_37: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_36, 1, 0, 9223372036854775807);  slice_36 = None
    slice_38: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_37, 2, 1, 9223372036854775807);  slice_37 = None
    slice_39: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 9223372036854775807);  slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_177: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_39, [1, 16, 512, 1023]);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_6: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_40: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_177, 0, 0, 9223372036854775807);  view_177 = None
    slice_41: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_40, 1, 0, 9223372036854775807);  slice_40 = None
    slice_42: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_41, 2, 0, 9223372036854775807);  slice_41 = None
    index_4: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_42, [None, None, None, iota_6]);  slice_42 = iota_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_48: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_171, index_4);  view_171 = index_4 = None
    add_49: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_48, 0);  add_48 = None
    mul_36: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_49, 0.125);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_4: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_36, [3], True)
    sub_12: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_36, amax_4);  mul_36 = amax_4 = None
    exp_4: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_5: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [3], True)
    div_5: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_26: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_123: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_26, 4);  clone_26 = None
    permute_201: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_123, [2, 0, 1, 4, 3]);  unsqueeze_123 = None
    unsqueeze_124: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_163, 4);  view_163 = None
    permute_202: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_124, [4, 1, 2, 3, 0]);  unsqueeze_124 = None
    permute_203: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_201, [2, 0, 4, 1, 3]);  permute_201 = None
    view_178: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_203, [16, 512, 512]);  permute_203 = None
    permute_204: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_202, [2, 4, 1, 3, 0]);  permute_202 = None
    view_179: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_204, [16, 512, 64]);  permute_204 = None
    bmm_38: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_178, view_179);  view_178 = view_179 = None
    view_180: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_38, [16, 512, 1, 1, 64]);  bmm_38 = None
    permute_205: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_180, [1, 3, 0, 4, 2]);  view_180 = None
    view_181: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_205, [512, 1, 16, 64]);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_125: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_181, 4);  view_181 = None
    permute_206: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_125, [0, 1, 4, 3, 2]);  unsqueeze_125 = None
    unsqueeze_126: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, 3);  arg34_1 = None
    unsqueeze_127: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 4);  unsqueeze_126 = None
    permute_207: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_127, [3, 4, 0, 2, 1]);  unsqueeze_127 = None
    permute_208: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_206, [0, 3, 4, 1, 2]);  permute_206 = None
    clone_27: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
    view_182: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_27, [1, 512, 1024]);  clone_27 = None
    permute_209: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_207, [3, 4, 1, 2, 0]);  permute_207 = None
    clone_28: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    view_183: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_28, [1, 1024, 1024]);  clone_28 = None
    bmm_39: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_182, view_183);  view_182 = view_183 = None
    view_184: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_39, [512, 1, 1, 1, 1024]);  bmm_39 = None
    permute_210: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_184, [0, 3, 4, 1, 2]);  view_184 = None
    view_185: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_210, [512, 1, 1024]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_29: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_185);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_50: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_29, add_45);  clone_29 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_16: "f32[512, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[512, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_51: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_13: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_50, getitem_17);  add_50 = getitem_17 = None
    mul_37: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
    mul_38: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_37, arg201_1);  mul_37 = arg201_1 = None
    add_52: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_38, arg202_1);  mul_38 = arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_186: "f32[512, 1024]" = torch.ops.aten.view.default(add_52, [512, 1024])
    permute_211: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
    addmm_8: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg204_1, view_186, permute_211);  arg204_1 = view_186 = permute_211 = None
    view_187: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_8, [512, 1, 4096]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_39: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_187, 0.5)
    mul_40: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_187, 0.7071067811865476);  view_187 = None
    erf_4: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_53: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_41: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_39, add_53);  mul_39 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_30: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_41);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_188: "f32[512, 4096]" = torch.ops.aten.view.default(clone_30, [512, 4096]);  clone_30 = None
    permute_212: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
    addmm_9: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg206_1, view_188, permute_212);  arg206_1 = view_188 = permute_212 = None
    view_189: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_9, [512, 1, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_31: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_189);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_54: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_31, add_52);  clone_31 = add_52 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
    getitem_18: "f32[512, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[512, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_55: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_14: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_54, getitem_19);  add_54 = getitem_19 = None
    mul_42: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_43: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_42, arg207_1);  mul_42 = arg207_1 = None
    add_56: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_43, arg208_1);  mul_43 = arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_43: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_56, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_10: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_43);  slice_43 = None
    alias_11: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_128: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_56, 3)
    unsqueeze_129: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 4);  unsqueeze_128 = None
    permute_213: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_129, [0, 1, 3, 4, 2]);  unsqueeze_129 = None
    unsqueeze_130: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, 3);  arg35_1 = None
    unsqueeze_131: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 4);  unsqueeze_130 = None
    permute_214: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_131, [3, 4, 1, 2, 0]);  unsqueeze_131 = None
    permute_215: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_213, [0, 4, 1, 2, 3]);  permute_213 = None
    view_190: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_215, [1, 512, 1024]);  permute_215 = None
    permute_216: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_214, [4, 1, 2, 3, 0]);  permute_214 = None
    view_191: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_216, [1, 1024, 1024]);  permute_216 = None
    bmm_40: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
    view_192: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_40, [512, 1, 1, 16, 64]);  bmm_40 = None
    permute_217: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_192, [0, 2, 3, 4, 1]);  view_192 = None
    view_193: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_217, [512, 1, 16, 64]);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_132: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_56, 3)
    unsqueeze_133: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 4);  unsqueeze_132 = None
    permute_218: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_133, [0, 1, 3, 4, 2]);  unsqueeze_133 = None
    unsqueeze_134: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, 3);  arg36_1 = None
    unsqueeze_135: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 4);  unsqueeze_134 = None
    permute_219: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_135, [3, 4, 1, 2, 0]);  unsqueeze_135 = None
    permute_220: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_218, [0, 4, 1, 2, 3]);  permute_218 = None
    view_194: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_220, [1, 512, 1024]);  permute_220 = None
    permute_221: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_219, [4, 1, 2, 3, 0]);  permute_219 = None
    view_195: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_221, [1, 1024, 1024]);  permute_221 = None
    bmm_41: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_194, view_195);  view_194 = view_195 = None
    view_196: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_41, [512, 1, 1, 16, 64]);  bmm_41 = None
    permute_222: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_196, [0, 2, 3, 4, 1]);  view_196 = None
    view_197: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_222, [512, 1, 16, 64]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_136: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_56, 3)
    unsqueeze_137: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 4);  unsqueeze_136 = None
    permute_223: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_137, [0, 1, 3, 4, 2]);  unsqueeze_137 = None
    unsqueeze_138: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, 3);  arg37_1 = None
    unsqueeze_139: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 4);  unsqueeze_138 = None
    permute_224: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_139, [3, 4, 1, 2, 0]);  unsqueeze_139 = None
    permute_225: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_223, [0, 4, 1, 2, 3]);  permute_223 = None
    view_198: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_225, [1, 512, 1024]);  permute_225 = None
    permute_226: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_224, [4, 1, 2, 3, 0]);  permute_224 = None
    view_199: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_226, [1, 1024, 1024]);  permute_226 = None
    bmm_42: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_198, view_199);  view_198 = view_199 = None
    view_200: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_42, [512, 1, 1, 16, 64]);  bmm_42 = None
    permute_227: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_200, [0, 2, 3, 4, 1]);  view_200 = None
    view_201: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_227, [512, 1, 16, 64]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_140: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_141: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 4);  unsqueeze_140 = None
    permute_228: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_141, [0, 1, 3, 4, 2]);  unsqueeze_141 = None
    unsqueeze_142: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, 3);  arg38_1 = None
    unsqueeze_143: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 4);  unsqueeze_142 = None
    permute_229: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_143, [3, 4, 1, 2, 0]);  unsqueeze_143 = None
    permute_230: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_228, [0, 4, 1, 2, 3]);  permute_228 = None
    view_202: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_230, [1, 1024, 1024]);  permute_230 = None
    permute_231: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_229, [4, 1, 2, 3, 0]);  permute_229 = None
    view_203: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_231, [1, 1024, 1024]);  permute_231 = None
    bmm_43: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_202, view_203);  view_202 = view_203 = None
    view_204: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_43, [1024, 1, 1, 16, 64]);  bmm_43 = None
    permute_232: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_204, [0, 2, 3, 4, 1]);  view_204 = None
    view_205: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_232, [1024, 1, 16, 64]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_57: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_193, arg39_1);  arg39_1 = None
    unsqueeze_144: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_57, 4);  add_57 = None
    permute_233: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_144, [1, 2, 0, 4, 3]);  unsqueeze_144 = None
    unsqueeze_145: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_197, 4);  view_197 = None
    permute_234: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_145, [1, 2, 4, 0, 3]);  unsqueeze_145 = None
    permute_235: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_233, [1, 2, 4, 0, 3]);  permute_233 = None
    view_206: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_235, [16, 512, 64]);  permute_235 = None
    permute_236: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_234, [1, 4, 0, 3, 2]);  permute_234 = None
    view_207: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_236, [16, 64, 512]);  permute_236 = None
    bmm_44: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_206, view_207);  view_206 = view_207 = None
    view_208: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_44, [16, 512, 1, 1, 512]);  bmm_44 = None
    permute_237: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_208, [3, 0, 1, 4, 2]);  view_208 = None
    view_209: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_237, [1, 16, 512, 512]);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_58: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_193, arg40_1);  view_193 = arg40_1 = None
    unsqueeze_146: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_58, 4);  add_58 = None
    permute_238: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_146, [1, 2, 0, 4, 3]);  unsqueeze_146 = None
    unsqueeze_147: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_205, 4);  view_205 = None
    permute_239: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_147, [1, 2, 4, 0, 3]);  unsqueeze_147 = None
    permute_240: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_238, [1, 2, 4, 0, 3]);  permute_238 = None
    view_210: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_240, [16, 512, 64]);  permute_240 = None
    permute_241: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_239, [1, 4, 0, 3, 2]);  permute_239 = None
    view_211: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_241, [16, 64, 1024]);  permute_241 = None
    bmm_45: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
    view_212: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_45, [16, 512, 1, 1, 1024]);  bmm_45 = None
    permute_242: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_212, [3, 0, 1, 4, 2]);  view_212 = None
    view_213: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_242, [1, 16, 512, 1024]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_214: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_213, [1, 16, 1024, 512]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_44: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_214, 0, 0, 9223372036854775807);  view_214 = None
    slice_45: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_44, 1, 0, 9223372036854775807);  slice_44 = None
    slice_46: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_45, 2, 1, 9223372036854775807);  slice_45 = None
    slice_47: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 9223372036854775807);  slice_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_215: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_47, [1, 16, 512, 1023]);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_7: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_48: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_215, 0, 0, 9223372036854775807);  view_215 = None
    slice_49: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_48, 1, 0, 9223372036854775807);  slice_48 = None
    slice_50: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_49, 2, 0, 9223372036854775807);  slice_49 = None
    index_5: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_50, [None, None, None, iota_7]);  slice_50 = iota_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_59: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_209, index_5);  view_209 = index_5 = None
    add_60: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_59, 0);  add_59 = None
    mul_44: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_60, 0.125);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_5: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_44, [3], True)
    sub_15: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_44, amax_5);  mul_44 = amax_5 = None
    exp_5: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_6: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [3], True)
    div_6: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_32: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_148: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_32, 4);  clone_32 = None
    permute_243: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_148, [2, 0, 1, 4, 3]);  unsqueeze_148 = None
    unsqueeze_149: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_201, 4);  view_201 = None
    permute_244: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_149, [4, 1, 2, 3, 0]);  unsqueeze_149 = None
    permute_245: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_243, [2, 0, 4, 1, 3]);  permute_243 = None
    view_216: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_245, [16, 512, 512]);  permute_245 = None
    permute_246: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_244, [2, 4, 1, 3, 0]);  permute_244 = None
    view_217: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_246, [16, 512, 64]);  permute_246 = None
    bmm_46: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_216, view_217);  view_216 = view_217 = None
    view_218: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_46, [16, 512, 1, 1, 64]);  bmm_46 = None
    permute_247: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_218, [1, 3, 0, 4, 2]);  view_218 = None
    view_219: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_247, [512, 1, 16, 64]);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_150: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_219, 4);  view_219 = None
    permute_248: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_150, [0, 1, 4, 3, 2]);  unsqueeze_150 = None
    unsqueeze_151: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, 3);  arg41_1 = None
    unsqueeze_152: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 4);  unsqueeze_151 = None
    permute_249: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_152, [3, 4, 0, 2, 1]);  unsqueeze_152 = None
    permute_250: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_248, [0, 3, 4, 1, 2]);  permute_248 = None
    clone_33: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
    view_220: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_33, [1, 512, 1024]);  clone_33 = None
    permute_251: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_249, [3, 4, 1, 2, 0]);  permute_249 = None
    clone_34: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_251, memory_format = torch.contiguous_format);  permute_251 = None
    view_221: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_34, [1, 1024, 1024]);  clone_34 = None
    bmm_47: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_220, view_221);  view_220 = view_221 = None
    view_222: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_47, [512, 1, 1, 1, 1024]);  bmm_47 = None
    permute_252: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_222, [0, 3, 4, 1, 2]);  view_222 = None
    view_223: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_252, [512, 1, 1024]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_35: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_223);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_61: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_35, add_56);  clone_35 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_20: "f32[512, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[512, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_62: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_16: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_61, getitem_21);  add_61 = getitem_21 = None
    mul_45: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
    mul_46: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_45, arg209_1);  mul_45 = arg209_1 = None
    add_63: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_46, arg210_1);  mul_46 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_224: "f32[512, 1024]" = torch.ops.aten.view.default(add_63, [512, 1024])
    permute_253: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
    addmm_10: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg212_1, view_224, permute_253);  arg212_1 = view_224 = permute_253 = None
    view_225: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_10, [512, 1, 4096]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_225, 0.5)
    mul_48: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_225, 0.7071067811865476);  view_225 = None
    erf_5: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_64: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_49: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_47, add_64);  mul_47 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_36: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_49);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_226: "f32[512, 4096]" = torch.ops.aten.view.default(clone_36, [512, 4096]);  clone_36 = None
    permute_254: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
    addmm_11: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg214_1, view_226, permute_254);  arg214_1 = view_226 = permute_254 = None
    view_227: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_11, [512, 1, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_37: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_227);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_65: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_37, add_63);  clone_37 = add_63 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_22: "f32[512, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[512, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_66: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_17: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_23);  add_65 = getitem_23 = None
    mul_50: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_51: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_50, arg215_1);  mul_50 = arg215_1 = None
    add_67: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_51, arg216_1);  mul_51 = arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_51: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_67, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_12: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_51);  slice_51 = None
    alias_13: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_153: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_67, 3)
    unsqueeze_154: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 4);  unsqueeze_153 = None
    permute_255: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_154, [0, 1, 3, 4, 2]);  unsqueeze_154 = None
    unsqueeze_155: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg42_1, 3);  arg42_1 = None
    unsqueeze_156: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 4);  unsqueeze_155 = None
    permute_256: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_156, [3, 4, 1, 2, 0]);  unsqueeze_156 = None
    permute_257: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_255, [0, 4, 1, 2, 3]);  permute_255 = None
    view_228: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_257, [1, 512, 1024]);  permute_257 = None
    permute_258: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_256, [4, 1, 2, 3, 0]);  permute_256 = None
    view_229: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_258, [1, 1024, 1024]);  permute_258 = None
    bmm_48: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_228, view_229);  view_228 = view_229 = None
    view_230: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_48, [512, 1, 1, 16, 64]);  bmm_48 = None
    permute_259: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_230, [0, 2, 3, 4, 1]);  view_230 = None
    view_231: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_259, [512, 1, 16, 64]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_157: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_67, 3)
    unsqueeze_158: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 4);  unsqueeze_157 = None
    permute_260: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_158, [0, 1, 3, 4, 2]);  unsqueeze_158 = None
    unsqueeze_159: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, 3);  arg43_1 = None
    unsqueeze_160: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 4);  unsqueeze_159 = None
    permute_261: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_160, [3, 4, 1, 2, 0]);  unsqueeze_160 = None
    permute_262: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_260, [0, 4, 1, 2, 3]);  permute_260 = None
    view_232: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_262, [1, 512, 1024]);  permute_262 = None
    permute_263: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_261, [4, 1, 2, 3, 0]);  permute_261 = None
    view_233: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_263, [1, 1024, 1024]);  permute_263 = None
    bmm_49: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = view_233 = None
    view_234: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_49, [512, 1, 1, 16, 64]);  bmm_49 = None
    permute_264: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_234, [0, 2, 3, 4, 1]);  view_234 = None
    view_235: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_264, [512, 1, 16, 64]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_161: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_67, 3)
    unsqueeze_162: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 4);  unsqueeze_161 = None
    permute_265: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_162, [0, 1, 3, 4, 2]);  unsqueeze_162 = None
    unsqueeze_163: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, 3);  arg44_1 = None
    unsqueeze_164: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 4);  unsqueeze_163 = None
    permute_266: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_164, [3, 4, 1, 2, 0]);  unsqueeze_164 = None
    permute_267: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_265, [0, 4, 1, 2, 3]);  permute_265 = None
    view_236: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_267, [1, 512, 1024]);  permute_267 = None
    permute_268: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_266, [4, 1, 2, 3, 0]);  permute_266 = None
    view_237: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_268, [1, 1024, 1024]);  permute_268 = None
    bmm_50: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_236, view_237);  view_236 = view_237 = None
    view_238: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_50, [512, 1, 1, 16, 64]);  bmm_50 = None
    permute_269: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_238, [0, 2, 3, 4, 1]);  view_238 = None
    view_239: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_269, [512, 1, 16, 64]);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_165: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_166: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 4);  unsqueeze_165 = None
    permute_270: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_166, [0, 1, 3, 4, 2]);  unsqueeze_166 = None
    unsqueeze_167: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, 3);  arg45_1 = None
    unsqueeze_168: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 4);  unsqueeze_167 = None
    permute_271: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_168, [3, 4, 1, 2, 0]);  unsqueeze_168 = None
    permute_272: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_270, [0, 4, 1, 2, 3]);  permute_270 = None
    view_240: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_272, [1, 1024, 1024]);  permute_272 = None
    permute_273: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_271, [4, 1, 2, 3, 0]);  permute_271 = None
    view_241: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_273, [1, 1024, 1024]);  permute_273 = None
    bmm_51: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_240, view_241);  view_240 = view_241 = None
    view_242: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_51, [1024, 1, 1, 16, 64]);  bmm_51 = None
    permute_274: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_242, [0, 2, 3, 4, 1]);  view_242 = None
    view_243: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_274, [1024, 1, 16, 64]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_68: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_231, arg46_1);  arg46_1 = None
    unsqueeze_169: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_68, 4);  add_68 = None
    permute_275: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_169, [1, 2, 0, 4, 3]);  unsqueeze_169 = None
    unsqueeze_170: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_235, 4);  view_235 = None
    permute_276: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_170, [1, 2, 4, 0, 3]);  unsqueeze_170 = None
    permute_277: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_275, [1, 2, 4, 0, 3]);  permute_275 = None
    view_244: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_277, [16, 512, 64]);  permute_277 = None
    permute_278: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_276, [1, 4, 0, 3, 2]);  permute_276 = None
    view_245: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_278, [16, 64, 512]);  permute_278 = None
    bmm_52: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
    view_246: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_52, [16, 512, 1, 1, 512]);  bmm_52 = None
    permute_279: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_246, [3, 0, 1, 4, 2]);  view_246 = None
    view_247: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_279, [1, 16, 512, 512]);  permute_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_69: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_231, arg47_1);  view_231 = arg47_1 = None
    unsqueeze_171: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_69, 4);  add_69 = None
    permute_280: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_171, [1, 2, 0, 4, 3]);  unsqueeze_171 = None
    unsqueeze_172: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_243, 4);  view_243 = None
    permute_281: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_172, [1, 2, 4, 0, 3]);  unsqueeze_172 = None
    permute_282: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_280, [1, 2, 4, 0, 3]);  permute_280 = None
    view_248: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_282, [16, 512, 64]);  permute_282 = None
    permute_283: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_281, [1, 4, 0, 3, 2]);  permute_281 = None
    view_249: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_283, [16, 64, 1024]);  permute_283 = None
    bmm_53: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_248, view_249);  view_248 = view_249 = None
    view_250: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_53, [16, 512, 1, 1, 1024]);  bmm_53 = None
    permute_284: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_250, [3, 0, 1, 4, 2]);  view_250 = None
    view_251: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_284, [1, 16, 512, 1024]);  permute_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_252: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_251, [1, 16, 1024, 512]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_52: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_252, 0, 0, 9223372036854775807);  view_252 = None
    slice_53: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_52, 1, 0, 9223372036854775807);  slice_52 = None
    slice_54: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_53, 2, 1, 9223372036854775807);  slice_53 = None
    slice_55: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_54, 3, 0, 9223372036854775807);  slice_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_253: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_55, [1, 16, 512, 1023]);  slice_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_8: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_56: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_253, 0, 0, 9223372036854775807);  view_253 = None
    slice_57: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_56, 1, 0, 9223372036854775807);  slice_56 = None
    slice_58: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_57, 2, 0, 9223372036854775807);  slice_57 = None
    index_6: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_58, [None, None, None, iota_8]);  slice_58 = iota_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_70: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_247, index_6);  view_247 = index_6 = None
    add_71: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_70, 0);  add_70 = None
    mul_52: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_71, 0.125);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_6: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_52, [3], True)
    sub_18: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_52, amax_6);  mul_52 = amax_6 = None
    exp_6: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_7: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [3], True)
    div_7: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_38: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_173: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_38, 4);  clone_38 = None
    permute_285: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_173, [2, 0, 1, 4, 3]);  unsqueeze_173 = None
    unsqueeze_174: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_239, 4);  view_239 = None
    permute_286: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_174, [4, 1, 2, 3, 0]);  unsqueeze_174 = None
    permute_287: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_285, [2, 0, 4, 1, 3]);  permute_285 = None
    view_254: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_287, [16, 512, 512]);  permute_287 = None
    permute_288: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_286, [2, 4, 1, 3, 0]);  permute_286 = None
    view_255: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_288, [16, 512, 64]);  permute_288 = None
    bmm_54: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
    view_256: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_54, [16, 512, 1, 1, 64]);  bmm_54 = None
    permute_289: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_256, [1, 3, 0, 4, 2]);  view_256 = None
    view_257: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_289, [512, 1, 16, 64]);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_175: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_257, 4);  view_257 = None
    permute_290: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_175, [0, 1, 4, 3, 2]);  unsqueeze_175 = None
    unsqueeze_176: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, 3);  arg48_1 = None
    unsqueeze_177: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 4);  unsqueeze_176 = None
    permute_291: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_177, [3, 4, 0, 2, 1]);  unsqueeze_177 = None
    permute_292: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_290, [0, 3, 4, 1, 2]);  permute_290 = None
    clone_39: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_258: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_39, [1, 512, 1024]);  clone_39 = None
    permute_293: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_291, [3, 4, 1, 2, 0]);  permute_291 = None
    clone_40: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_259: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_40, [1, 1024, 1024]);  clone_40 = None
    bmm_55: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_258, view_259);  view_258 = view_259 = None
    view_260: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_55, [512, 1, 1, 1, 1024]);  bmm_55 = None
    permute_294: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_260, [0, 3, 4, 1, 2]);  view_260 = None
    view_261: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_294, [512, 1, 1024]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_41: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_261);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_72: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_41, add_67);  clone_41 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_24: "f32[512, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[512, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_73: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_19: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_72, getitem_25);  add_72 = getitem_25 = None
    mul_53: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
    mul_54: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_53, arg217_1);  mul_53 = arg217_1 = None
    add_74: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_54, arg218_1);  mul_54 = arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_262: "f32[512, 1024]" = torch.ops.aten.view.default(add_74, [512, 1024])
    permute_295: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
    addmm_12: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg220_1, view_262, permute_295);  arg220_1 = view_262 = permute_295 = None
    view_263: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_12, [512, 1, 4096]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_56: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
    erf_6: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_75: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_57: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_55, add_75);  mul_55 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_42: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_264: "f32[512, 4096]" = torch.ops.aten.view.default(clone_42, [512, 4096]);  clone_42 = None
    permute_296: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
    addmm_13: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg222_1, view_264, permute_296);  arg222_1 = view_264 = permute_296 = None
    view_265: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_13, [512, 1, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_43: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_265);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_76: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_43, add_74);  clone_43 = add_74 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
    getitem_26: "f32[512, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[512, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_77: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_20: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_76, getitem_27);  add_76 = getitem_27 = None
    mul_58: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_59: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_58, arg223_1);  mul_58 = arg223_1 = None
    add_78: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_59, arg224_1);  mul_59 = arg224_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_59: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_78, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_14: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_59);  slice_59 = None
    alias_15: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_178: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_78, 3)
    unsqueeze_179: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 4);  unsqueeze_178 = None
    permute_297: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_179, [0, 1, 3, 4, 2]);  unsqueeze_179 = None
    unsqueeze_180: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, 3);  arg49_1 = None
    unsqueeze_181: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 4);  unsqueeze_180 = None
    permute_298: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_181, [3, 4, 1, 2, 0]);  unsqueeze_181 = None
    permute_299: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_297, [0, 4, 1, 2, 3]);  permute_297 = None
    view_266: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_299, [1, 512, 1024]);  permute_299 = None
    permute_300: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_298, [4, 1, 2, 3, 0]);  permute_298 = None
    view_267: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_300, [1, 1024, 1024]);  permute_300 = None
    bmm_56: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_266, view_267);  view_266 = view_267 = None
    view_268: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_56, [512, 1, 1, 16, 64]);  bmm_56 = None
    permute_301: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_268, [0, 2, 3, 4, 1]);  view_268 = None
    view_269: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_301, [512, 1, 16, 64]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_182: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_78, 3)
    unsqueeze_183: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 4);  unsqueeze_182 = None
    permute_302: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_183, [0, 1, 3, 4, 2]);  unsqueeze_183 = None
    unsqueeze_184: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, 3);  arg50_1 = None
    unsqueeze_185: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 4);  unsqueeze_184 = None
    permute_303: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_185, [3, 4, 1, 2, 0]);  unsqueeze_185 = None
    permute_304: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_302, [0, 4, 1, 2, 3]);  permute_302 = None
    view_270: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_304, [1, 512, 1024]);  permute_304 = None
    permute_305: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_303, [4, 1, 2, 3, 0]);  permute_303 = None
    view_271: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_305, [1, 1024, 1024]);  permute_305 = None
    bmm_57: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_270, view_271);  view_270 = view_271 = None
    view_272: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_57, [512, 1, 1, 16, 64]);  bmm_57 = None
    permute_306: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_272, [0, 2, 3, 4, 1]);  view_272 = None
    view_273: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_306, [512, 1, 16, 64]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_186: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_78, 3)
    unsqueeze_187: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 4);  unsqueeze_186 = None
    permute_307: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_187, [0, 1, 3, 4, 2]);  unsqueeze_187 = None
    unsqueeze_188: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, 3);  arg51_1 = None
    unsqueeze_189: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 4);  unsqueeze_188 = None
    permute_308: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_189, [3, 4, 1, 2, 0]);  unsqueeze_189 = None
    permute_309: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_307, [0, 4, 1, 2, 3]);  permute_307 = None
    view_274: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_309, [1, 512, 1024]);  permute_309 = None
    permute_310: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_308, [4, 1, 2, 3, 0]);  permute_308 = None
    view_275: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_310, [1, 1024, 1024]);  permute_310 = None
    bmm_58: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_274, view_275);  view_274 = view_275 = None
    view_276: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_58, [512, 1, 1, 16, 64]);  bmm_58 = None
    permute_311: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_276, [0, 2, 3, 4, 1]);  view_276 = None
    view_277: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_311, [512, 1, 16, 64]);  permute_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_190: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_191: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 4);  unsqueeze_190 = None
    permute_312: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 3, 4, 2]);  unsqueeze_191 = None
    unsqueeze_192: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, 3);  arg52_1 = None
    unsqueeze_193: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 4);  unsqueeze_192 = None
    permute_313: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_193, [3, 4, 1, 2, 0]);  unsqueeze_193 = None
    permute_314: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_312, [0, 4, 1, 2, 3]);  permute_312 = None
    view_278: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_314, [1, 1024, 1024]);  permute_314 = None
    permute_315: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_313, [4, 1, 2, 3, 0]);  permute_313 = None
    view_279: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_315, [1, 1024, 1024]);  permute_315 = None
    bmm_59: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_278, view_279);  view_278 = view_279 = None
    view_280: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_59, [1024, 1, 1, 16, 64]);  bmm_59 = None
    permute_316: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_280, [0, 2, 3, 4, 1]);  view_280 = None
    view_281: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_316, [1024, 1, 16, 64]);  permute_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_79: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_269, arg53_1);  arg53_1 = None
    unsqueeze_194: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_79, 4);  add_79 = None
    permute_317: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_194, [1, 2, 0, 4, 3]);  unsqueeze_194 = None
    unsqueeze_195: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_273, 4);  view_273 = None
    permute_318: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_195, [1, 2, 4, 0, 3]);  unsqueeze_195 = None
    permute_319: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_317, [1, 2, 4, 0, 3]);  permute_317 = None
    view_282: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_319, [16, 512, 64]);  permute_319 = None
    permute_320: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_318, [1, 4, 0, 3, 2]);  permute_318 = None
    view_283: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_320, [16, 64, 512]);  permute_320 = None
    bmm_60: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_282, view_283);  view_282 = view_283 = None
    view_284: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_60, [16, 512, 1, 1, 512]);  bmm_60 = None
    permute_321: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_284, [3, 0, 1, 4, 2]);  view_284 = None
    view_285: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_321, [1, 16, 512, 512]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_80: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_269, arg54_1);  view_269 = arg54_1 = None
    unsqueeze_196: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_80, 4);  add_80 = None
    permute_322: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_196, [1, 2, 0, 4, 3]);  unsqueeze_196 = None
    unsqueeze_197: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_281, 4);  view_281 = None
    permute_323: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_197, [1, 2, 4, 0, 3]);  unsqueeze_197 = None
    permute_324: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_322, [1, 2, 4, 0, 3]);  permute_322 = None
    view_286: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_324, [16, 512, 64]);  permute_324 = None
    permute_325: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_323, [1, 4, 0, 3, 2]);  permute_323 = None
    view_287: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_325, [16, 64, 1024]);  permute_325 = None
    bmm_61: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_286, view_287);  view_286 = view_287 = None
    view_288: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_61, [16, 512, 1, 1, 1024]);  bmm_61 = None
    permute_326: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_288, [3, 0, 1, 4, 2]);  view_288 = None
    view_289: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_326, [1, 16, 512, 1024]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_290: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_289, [1, 16, 1024, 512]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_60: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_290, 0, 0, 9223372036854775807);  view_290 = None
    slice_61: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_60, 1, 0, 9223372036854775807);  slice_60 = None
    slice_62: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_61, 2, 1, 9223372036854775807);  slice_61 = None
    slice_63: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_62, 3, 0, 9223372036854775807);  slice_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_291: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_63, [1, 16, 512, 1023]);  slice_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_9: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_64: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_291, 0, 0, 9223372036854775807);  view_291 = None
    slice_65: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_64, 1, 0, 9223372036854775807);  slice_64 = None
    slice_66: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_65, 2, 0, 9223372036854775807);  slice_65 = None
    index_7: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_66, [None, None, None, iota_9]);  slice_66 = iota_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_81: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_285, index_7);  view_285 = index_7 = None
    add_82: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_81, 0);  add_81 = None
    mul_60: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_82, 0.125);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_7: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_60, [3], True)
    sub_21: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_60, amax_7);  mul_60 = amax_7 = None
    exp_7: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_8: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [3], True)
    div_8: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_44: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_198: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_44, 4);  clone_44 = None
    permute_327: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_198, [2, 0, 1, 4, 3]);  unsqueeze_198 = None
    unsqueeze_199: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_277, 4);  view_277 = None
    permute_328: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_199, [4, 1, 2, 3, 0]);  unsqueeze_199 = None
    permute_329: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_327, [2, 0, 4, 1, 3]);  permute_327 = None
    view_292: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_329, [16, 512, 512]);  permute_329 = None
    permute_330: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_328, [2, 4, 1, 3, 0]);  permute_328 = None
    view_293: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_330, [16, 512, 64]);  permute_330 = None
    bmm_62: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_292, view_293);  view_292 = view_293 = None
    view_294: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_62, [16, 512, 1, 1, 64]);  bmm_62 = None
    permute_331: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_294, [1, 3, 0, 4, 2]);  view_294 = None
    view_295: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_331, [512, 1, 16, 64]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_200: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_295, 4);  view_295 = None
    permute_332: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 4, 3, 2]);  unsqueeze_200 = None
    unsqueeze_201: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, 3);  arg55_1 = None
    unsqueeze_202: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 4);  unsqueeze_201 = None
    permute_333: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_202, [3, 4, 0, 2, 1]);  unsqueeze_202 = None
    permute_334: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_332, [0, 3, 4, 1, 2]);  permute_332 = None
    clone_45: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
    view_296: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_45, [1, 512, 1024]);  clone_45 = None
    permute_335: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_333, [3, 4, 1, 2, 0]);  permute_333 = None
    clone_46: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
    view_297: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_46, [1, 1024, 1024]);  clone_46 = None
    bmm_63: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_296, view_297);  view_296 = view_297 = None
    view_298: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_63, [512, 1, 1, 1, 1024]);  bmm_63 = None
    permute_336: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_298, [0, 3, 4, 1, 2]);  view_298 = None
    view_299: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_336, [512, 1, 1024]);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_47: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_299);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_83: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_47, add_78);  clone_47 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_28: "f32[512, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[512, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_84: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_22: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_83, getitem_29);  add_83 = getitem_29 = None
    mul_61: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
    mul_62: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_61, arg225_1);  mul_61 = arg225_1 = None
    add_85: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_62, arg226_1);  mul_62 = arg226_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_300: "f32[512, 1024]" = torch.ops.aten.view.default(add_85, [512, 1024])
    permute_337: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    addmm_14: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg228_1, view_300, permute_337);  arg228_1 = view_300 = permute_337 = None
    view_301: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_14, [512, 1, 4096]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_63: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_301, 0.5)
    mul_64: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_301, 0.7071067811865476);  view_301 = None
    erf_7: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_86: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_65: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_63, add_86);  mul_63 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_48: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_65);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_302: "f32[512, 4096]" = torch.ops.aten.view.default(clone_48, [512, 4096]);  clone_48 = None
    permute_338: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
    addmm_15: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg230_1, view_302, permute_338);  arg230_1 = view_302 = permute_338 = None
    view_303: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_15, [512, 1, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_49: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_303);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_87: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_49, add_85);  clone_49 = add_85 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_30: "f32[512, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[512, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_88: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_23: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_87, getitem_31);  add_87 = getitem_31 = None
    mul_66: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_67: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_66, arg231_1);  mul_66 = arg231_1 = None
    add_89: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_67, arg232_1);  mul_67 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_67: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_89, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_16: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_67);  slice_67 = None
    alias_17: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_203: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_89, 3)
    unsqueeze_204: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 4);  unsqueeze_203 = None
    permute_339: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_204, [0, 1, 3, 4, 2]);  unsqueeze_204 = None
    unsqueeze_205: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, 3);  arg56_1 = None
    unsqueeze_206: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 4);  unsqueeze_205 = None
    permute_340: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_206, [3, 4, 1, 2, 0]);  unsqueeze_206 = None
    permute_341: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_339, [0, 4, 1, 2, 3]);  permute_339 = None
    view_304: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_341, [1, 512, 1024]);  permute_341 = None
    permute_342: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_340, [4, 1, 2, 3, 0]);  permute_340 = None
    view_305: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_342, [1, 1024, 1024]);  permute_342 = None
    bmm_64: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_304, view_305);  view_304 = view_305 = None
    view_306: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_64, [512, 1, 1, 16, 64]);  bmm_64 = None
    permute_343: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_306, [0, 2, 3, 4, 1]);  view_306 = None
    view_307: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_343, [512, 1, 16, 64]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_207: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_89, 3)
    unsqueeze_208: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 4);  unsqueeze_207 = None
    permute_344: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_208, [0, 1, 3, 4, 2]);  unsqueeze_208 = None
    unsqueeze_209: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg57_1, 3);  arg57_1 = None
    unsqueeze_210: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 4);  unsqueeze_209 = None
    permute_345: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_210, [3, 4, 1, 2, 0]);  unsqueeze_210 = None
    permute_346: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_344, [0, 4, 1, 2, 3]);  permute_344 = None
    view_308: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_346, [1, 512, 1024]);  permute_346 = None
    permute_347: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_345, [4, 1, 2, 3, 0]);  permute_345 = None
    view_309: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_347, [1, 1024, 1024]);  permute_347 = None
    bmm_65: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_308, view_309);  view_308 = view_309 = None
    view_310: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_65, [512, 1, 1, 16, 64]);  bmm_65 = None
    permute_348: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_310, [0, 2, 3, 4, 1]);  view_310 = None
    view_311: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_348, [512, 1, 16, 64]);  permute_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_211: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_89, 3)
    unsqueeze_212: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 4);  unsqueeze_211 = None
    permute_349: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_212, [0, 1, 3, 4, 2]);  unsqueeze_212 = None
    unsqueeze_213: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, 3);  arg58_1 = None
    unsqueeze_214: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 4);  unsqueeze_213 = None
    permute_350: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_214, [3, 4, 1, 2, 0]);  unsqueeze_214 = None
    permute_351: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_349, [0, 4, 1, 2, 3]);  permute_349 = None
    view_312: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_351, [1, 512, 1024]);  permute_351 = None
    permute_352: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_350, [4, 1, 2, 3, 0]);  permute_350 = None
    view_313: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_352, [1, 1024, 1024]);  permute_352 = None
    bmm_66: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_312, view_313);  view_312 = view_313 = None
    view_314: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_66, [512, 1, 1, 16, 64]);  bmm_66 = None
    permute_353: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_314, [0, 2, 3, 4, 1]);  view_314 = None
    view_315: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_353, [512, 1, 16, 64]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_215: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_216: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 4);  unsqueeze_215 = None
    permute_354: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_216, [0, 1, 3, 4, 2]);  unsqueeze_216 = None
    unsqueeze_217: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, 3);  arg59_1 = None
    unsqueeze_218: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 4);  unsqueeze_217 = None
    permute_355: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_218, [3, 4, 1, 2, 0]);  unsqueeze_218 = None
    permute_356: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_354, [0, 4, 1, 2, 3]);  permute_354 = None
    view_316: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_356, [1, 1024, 1024]);  permute_356 = None
    permute_357: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_355, [4, 1, 2, 3, 0]);  permute_355 = None
    view_317: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_357, [1, 1024, 1024]);  permute_357 = None
    bmm_67: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_316, view_317);  view_316 = view_317 = None
    view_318: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_67, [1024, 1, 1, 16, 64]);  bmm_67 = None
    permute_358: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_318, [0, 2, 3, 4, 1]);  view_318 = None
    view_319: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_358, [1024, 1, 16, 64]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_90: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_307, arg60_1);  arg60_1 = None
    unsqueeze_219: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_90, 4);  add_90 = None
    permute_359: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_219, [1, 2, 0, 4, 3]);  unsqueeze_219 = None
    unsqueeze_220: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_311, 4);  view_311 = None
    permute_360: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_220, [1, 2, 4, 0, 3]);  unsqueeze_220 = None
    permute_361: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_359, [1, 2, 4, 0, 3]);  permute_359 = None
    view_320: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_361, [16, 512, 64]);  permute_361 = None
    permute_362: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_360, [1, 4, 0, 3, 2]);  permute_360 = None
    view_321: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_362, [16, 64, 512]);  permute_362 = None
    bmm_68: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_320, view_321);  view_320 = view_321 = None
    view_322: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_68, [16, 512, 1, 1, 512]);  bmm_68 = None
    permute_363: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_322, [3, 0, 1, 4, 2]);  view_322 = None
    view_323: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_363, [1, 16, 512, 512]);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_91: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_307, arg61_1);  view_307 = arg61_1 = None
    unsqueeze_221: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_91, 4);  add_91 = None
    permute_364: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_221, [1, 2, 0, 4, 3]);  unsqueeze_221 = None
    unsqueeze_222: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_319, 4);  view_319 = None
    permute_365: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_222, [1, 2, 4, 0, 3]);  unsqueeze_222 = None
    permute_366: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_364, [1, 2, 4, 0, 3]);  permute_364 = None
    view_324: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_366, [16, 512, 64]);  permute_366 = None
    permute_367: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_365, [1, 4, 0, 3, 2]);  permute_365 = None
    view_325: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_367, [16, 64, 1024]);  permute_367 = None
    bmm_69: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_324, view_325);  view_324 = view_325 = None
    view_326: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_69, [16, 512, 1, 1, 1024]);  bmm_69 = None
    permute_368: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_326, [3, 0, 1, 4, 2]);  view_326 = None
    view_327: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_368, [1, 16, 512, 1024]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_328: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_327, [1, 16, 1024, 512]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_68: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_328, 0, 0, 9223372036854775807);  view_328 = None
    slice_69: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_68, 1, 0, 9223372036854775807);  slice_68 = None
    slice_70: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_69, 2, 1, 9223372036854775807);  slice_69 = None
    slice_71: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_70, 3, 0, 9223372036854775807);  slice_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_329: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_71, [1, 16, 512, 1023]);  slice_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_10: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_72: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_329, 0, 0, 9223372036854775807);  view_329 = None
    slice_73: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_72, 1, 0, 9223372036854775807);  slice_72 = None
    slice_74: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_73, 2, 0, 9223372036854775807);  slice_73 = None
    index_8: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_74, [None, None, None, iota_10]);  slice_74 = iota_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_92: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_323, index_8);  view_323 = index_8 = None
    add_93: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_92, 0);  add_92 = None
    mul_68: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_93, 0.125);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_8: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_68, [3], True)
    sub_24: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_68, amax_8);  mul_68 = amax_8 = None
    exp_8: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_9: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [3], True)
    div_9: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_50: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_223: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_50, 4);  clone_50 = None
    permute_369: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_223, [2, 0, 1, 4, 3]);  unsqueeze_223 = None
    unsqueeze_224: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_315, 4);  view_315 = None
    permute_370: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_224, [4, 1, 2, 3, 0]);  unsqueeze_224 = None
    permute_371: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_369, [2, 0, 4, 1, 3]);  permute_369 = None
    view_330: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_371, [16, 512, 512]);  permute_371 = None
    permute_372: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_370, [2, 4, 1, 3, 0]);  permute_370 = None
    view_331: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_372, [16, 512, 64]);  permute_372 = None
    bmm_70: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_330, view_331);  view_330 = view_331 = None
    view_332: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_70, [16, 512, 1, 1, 64]);  bmm_70 = None
    permute_373: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_332, [1, 3, 0, 4, 2]);  view_332 = None
    view_333: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_373, [512, 1, 16, 64]);  permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_225: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_333, 4);  view_333 = None
    permute_374: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_225, [0, 1, 4, 3, 2]);  unsqueeze_225 = None
    unsqueeze_226: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, 3);  arg62_1 = None
    unsqueeze_227: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 4);  unsqueeze_226 = None
    permute_375: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_227, [3, 4, 0, 2, 1]);  unsqueeze_227 = None
    permute_376: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_374, [0, 3, 4, 1, 2]);  permute_374 = None
    clone_51: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format);  permute_376 = None
    view_334: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_51, [1, 512, 1024]);  clone_51 = None
    permute_377: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_375, [3, 4, 1, 2, 0]);  permute_375 = None
    clone_52: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_377, memory_format = torch.contiguous_format);  permute_377 = None
    view_335: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_52, [1, 1024, 1024]);  clone_52 = None
    bmm_71: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_334, view_335);  view_334 = view_335 = None
    view_336: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_71, [512, 1, 1, 1, 1024]);  bmm_71 = None
    permute_378: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_336, [0, 3, 4, 1, 2]);  view_336 = None
    view_337: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_378, [512, 1, 1024]);  permute_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_53: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_337);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_94: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_53, add_89);  clone_53 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_32: "f32[512, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[512, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_95: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_25: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_94, getitem_33);  add_94 = getitem_33 = None
    mul_69: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
    mul_70: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_69, arg233_1);  mul_69 = arg233_1 = None
    add_96: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_70, arg234_1);  mul_70 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_338: "f32[512, 1024]" = torch.ops.aten.view.default(add_96, [512, 1024])
    permute_379: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
    addmm_16: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg236_1, view_338, permute_379);  arg236_1 = view_338 = permute_379 = None
    view_339: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_16, [512, 1, 4096]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_339, 0.5)
    mul_72: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_339, 0.7071067811865476);  view_339 = None
    erf_8: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_97: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_73: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_71, add_97);  mul_71 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_54: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_73);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_340: "f32[512, 4096]" = torch.ops.aten.view.default(clone_54, [512, 4096]);  clone_54 = None
    permute_380: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
    addmm_17: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg238_1, view_340, permute_380);  arg238_1 = view_340 = permute_380 = None
    view_341: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_17, [512, 1, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_55: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_341);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_98: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_55, add_96);  clone_55 = add_96 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_34: "f32[512, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[512, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_99: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_26: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_98, getitem_35);  add_98 = getitem_35 = None
    mul_74: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_75: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_74, arg239_1);  mul_74 = arg239_1 = None
    add_100: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_75, arg240_1);  mul_75 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_75: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_100, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_18: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_75);  slice_75 = None
    alias_19: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_228: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_100, 3)
    unsqueeze_229: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 4);  unsqueeze_228 = None
    permute_381: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_229, [0, 1, 3, 4, 2]);  unsqueeze_229 = None
    unsqueeze_230: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, 3);  arg63_1 = None
    unsqueeze_231: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 4);  unsqueeze_230 = None
    permute_382: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_231, [3, 4, 1, 2, 0]);  unsqueeze_231 = None
    permute_383: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_381, [0, 4, 1, 2, 3]);  permute_381 = None
    view_342: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_383, [1, 512, 1024]);  permute_383 = None
    permute_384: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_382, [4, 1, 2, 3, 0]);  permute_382 = None
    view_343: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_384, [1, 1024, 1024]);  permute_384 = None
    bmm_72: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_342, view_343);  view_342 = view_343 = None
    view_344: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_72, [512, 1, 1, 16, 64]);  bmm_72 = None
    permute_385: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_344, [0, 2, 3, 4, 1]);  view_344 = None
    view_345: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_385, [512, 1, 16, 64]);  permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_232: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_100, 3)
    unsqueeze_233: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 4);  unsqueeze_232 = None
    permute_386: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_233, [0, 1, 3, 4, 2]);  unsqueeze_233 = None
    unsqueeze_234: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, 3);  arg64_1 = None
    unsqueeze_235: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 4);  unsqueeze_234 = None
    permute_387: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_235, [3, 4, 1, 2, 0]);  unsqueeze_235 = None
    permute_388: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_386, [0, 4, 1, 2, 3]);  permute_386 = None
    view_346: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_388, [1, 512, 1024]);  permute_388 = None
    permute_389: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_387, [4, 1, 2, 3, 0]);  permute_387 = None
    view_347: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_389, [1, 1024, 1024]);  permute_389 = None
    bmm_73: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_346, view_347);  view_346 = view_347 = None
    view_348: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_73, [512, 1, 1, 16, 64]);  bmm_73 = None
    permute_390: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_348, [0, 2, 3, 4, 1]);  view_348 = None
    view_349: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_390, [512, 1, 16, 64]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_236: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_100, 3)
    unsqueeze_237: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 4);  unsqueeze_236 = None
    permute_391: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_237, [0, 1, 3, 4, 2]);  unsqueeze_237 = None
    unsqueeze_238: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, 3);  arg65_1 = None
    unsqueeze_239: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 4);  unsqueeze_238 = None
    permute_392: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_239, [3, 4, 1, 2, 0]);  unsqueeze_239 = None
    permute_393: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_391, [0, 4, 1, 2, 3]);  permute_391 = None
    view_350: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_393, [1, 512, 1024]);  permute_393 = None
    permute_394: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_392, [4, 1, 2, 3, 0]);  permute_392 = None
    view_351: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_394, [1, 1024, 1024]);  permute_394 = None
    bmm_74: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_350, view_351);  view_350 = view_351 = None
    view_352: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_74, [512, 1, 1, 16, 64]);  bmm_74 = None
    permute_395: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_352, [0, 2, 3, 4, 1]);  view_352 = None
    view_353: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_395, [512, 1, 16, 64]);  permute_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_240: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_241: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 4);  unsqueeze_240 = None
    permute_396: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_241, [0, 1, 3, 4, 2]);  unsqueeze_241 = None
    unsqueeze_242: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, 3);  arg66_1 = None
    unsqueeze_243: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 4);  unsqueeze_242 = None
    permute_397: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_243, [3, 4, 1, 2, 0]);  unsqueeze_243 = None
    permute_398: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_396, [0, 4, 1, 2, 3]);  permute_396 = None
    view_354: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_398, [1, 1024, 1024]);  permute_398 = None
    permute_399: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_397, [4, 1, 2, 3, 0]);  permute_397 = None
    view_355: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_399, [1, 1024, 1024]);  permute_399 = None
    bmm_75: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_354, view_355);  view_354 = view_355 = None
    view_356: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_75, [1024, 1, 1, 16, 64]);  bmm_75 = None
    permute_400: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_356, [0, 2, 3, 4, 1]);  view_356 = None
    view_357: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_400, [1024, 1, 16, 64]);  permute_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_101: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_345, arg67_1);  arg67_1 = None
    unsqueeze_244: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_101, 4);  add_101 = None
    permute_401: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_244, [1, 2, 0, 4, 3]);  unsqueeze_244 = None
    unsqueeze_245: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_349, 4);  view_349 = None
    permute_402: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_245, [1, 2, 4, 0, 3]);  unsqueeze_245 = None
    permute_403: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_401, [1, 2, 4, 0, 3]);  permute_401 = None
    view_358: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_403, [16, 512, 64]);  permute_403 = None
    permute_404: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_402, [1, 4, 0, 3, 2]);  permute_402 = None
    view_359: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_404, [16, 64, 512]);  permute_404 = None
    bmm_76: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_358, view_359);  view_358 = view_359 = None
    view_360: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_76, [16, 512, 1, 1, 512]);  bmm_76 = None
    permute_405: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_360, [3, 0, 1, 4, 2]);  view_360 = None
    view_361: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_405, [1, 16, 512, 512]);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_102: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_345, arg68_1);  view_345 = arg68_1 = None
    unsqueeze_246: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_102, 4);  add_102 = None
    permute_406: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_246, [1, 2, 0, 4, 3]);  unsqueeze_246 = None
    unsqueeze_247: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_357, 4);  view_357 = None
    permute_407: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_247, [1, 2, 4, 0, 3]);  unsqueeze_247 = None
    permute_408: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_406, [1, 2, 4, 0, 3]);  permute_406 = None
    view_362: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_408, [16, 512, 64]);  permute_408 = None
    permute_409: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_407, [1, 4, 0, 3, 2]);  permute_407 = None
    view_363: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_409, [16, 64, 1024]);  permute_409 = None
    bmm_77: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_362, view_363);  view_362 = view_363 = None
    view_364: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_77, [16, 512, 1, 1, 1024]);  bmm_77 = None
    permute_410: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_364, [3, 0, 1, 4, 2]);  view_364 = None
    view_365: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_410, [1, 16, 512, 1024]);  permute_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_366: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_365, [1, 16, 1024, 512]);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_76: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_366, 0, 0, 9223372036854775807);  view_366 = None
    slice_77: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_76, 1, 0, 9223372036854775807);  slice_76 = None
    slice_78: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_77, 2, 1, 9223372036854775807);  slice_77 = None
    slice_79: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_78, 3, 0, 9223372036854775807);  slice_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_367: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_79, [1, 16, 512, 1023]);  slice_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_11: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_80: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_367, 0, 0, 9223372036854775807);  view_367 = None
    slice_81: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_80, 1, 0, 9223372036854775807);  slice_80 = None
    slice_82: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_81, 2, 0, 9223372036854775807);  slice_81 = None
    index_9: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_82, [None, None, None, iota_11]);  slice_82 = iota_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_103: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_361, index_9);  view_361 = index_9 = None
    add_104: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_103, 0);  add_103 = None
    mul_76: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_104, 0.125);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_9: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_76, [3], True)
    sub_27: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_76, amax_9);  mul_76 = amax_9 = None
    exp_9: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_10: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [3], True)
    div_10: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_56: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_248: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_56, 4);  clone_56 = None
    permute_411: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_248, [2, 0, 1, 4, 3]);  unsqueeze_248 = None
    unsqueeze_249: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_353, 4);  view_353 = None
    permute_412: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_249, [4, 1, 2, 3, 0]);  unsqueeze_249 = None
    permute_413: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_411, [2, 0, 4, 1, 3]);  permute_411 = None
    view_368: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_413, [16, 512, 512]);  permute_413 = None
    permute_414: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_412, [2, 4, 1, 3, 0]);  permute_412 = None
    view_369: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_414, [16, 512, 64]);  permute_414 = None
    bmm_78: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_368, view_369);  view_368 = view_369 = None
    view_370: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_78, [16, 512, 1, 1, 64]);  bmm_78 = None
    permute_415: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_370, [1, 3, 0, 4, 2]);  view_370 = None
    view_371: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_415, [512, 1, 16, 64]);  permute_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_250: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_371, 4);  view_371 = None
    permute_416: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_250, [0, 1, 4, 3, 2]);  unsqueeze_250 = None
    unsqueeze_251: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, 3);  arg69_1 = None
    unsqueeze_252: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 4);  unsqueeze_251 = None
    permute_417: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_252, [3, 4, 0, 2, 1]);  unsqueeze_252 = None
    permute_418: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_416, [0, 3, 4, 1, 2]);  permute_416 = None
    clone_57: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
    view_372: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_57, [1, 512, 1024]);  clone_57 = None
    permute_419: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_417, [3, 4, 1, 2, 0]);  permute_417 = None
    clone_58: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_373: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_58, [1, 1024, 1024]);  clone_58 = None
    bmm_79: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_372, view_373);  view_372 = view_373 = None
    view_374: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_79, [512, 1, 1, 1, 1024]);  bmm_79 = None
    permute_420: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_374, [0, 3, 4, 1, 2]);  view_374 = None
    view_375: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_420, [512, 1, 1024]);  permute_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_59: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_375);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_105: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_59, add_100);  clone_59 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_36: "f32[512, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[512, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_106: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_28: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_105, getitem_37);  add_105 = getitem_37 = None
    mul_77: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
    mul_78: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_77, arg241_1);  mul_77 = arg241_1 = None
    add_107: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_78, arg242_1);  mul_78 = arg242_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_376: "f32[512, 1024]" = torch.ops.aten.view.default(add_107, [512, 1024])
    permute_421: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
    addmm_18: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg244_1, view_376, permute_421);  arg244_1 = view_376 = permute_421 = None
    view_377: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_18, [512, 1, 4096]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_79: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_377, 0.5)
    mul_80: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_377, 0.7071067811865476);  view_377 = None
    erf_9: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_80);  mul_80 = None
    add_108: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_81: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_79, add_108);  mul_79 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_60: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_81);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_378: "f32[512, 4096]" = torch.ops.aten.view.default(clone_60, [512, 4096]);  clone_60 = None
    permute_422: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
    addmm_19: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg246_1, view_378, permute_422);  arg246_1 = view_378 = permute_422 = None
    view_379: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_19, [512, 1, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_61: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_379);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_109: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_61, add_107);  clone_61 = add_107 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_38: "f32[512, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[512, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_110: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_29: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_109, getitem_39);  add_109 = getitem_39 = None
    mul_82: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_83: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_82, arg247_1);  mul_82 = arg247_1 = None
    add_111: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_83, arg248_1);  mul_83 = arg248_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_83: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_111, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_20: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_83);  slice_83 = None
    alias_21: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_253: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_111, 3)
    unsqueeze_254: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 4);  unsqueeze_253 = None
    permute_423: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_254, [0, 1, 3, 4, 2]);  unsqueeze_254 = None
    unsqueeze_255: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, 3);  arg70_1 = None
    unsqueeze_256: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 4);  unsqueeze_255 = None
    permute_424: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_256, [3, 4, 1, 2, 0]);  unsqueeze_256 = None
    permute_425: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_423, [0, 4, 1, 2, 3]);  permute_423 = None
    view_380: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_425, [1, 512, 1024]);  permute_425 = None
    permute_426: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_424, [4, 1, 2, 3, 0]);  permute_424 = None
    view_381: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_426, [1, 1024, 1024]);  permute_426 = None
    bmm_80: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_380, view_381);  view_380 = view_381 = None
    view_382: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_80, [512, 1, 1, 16, 64]);  bmm_80 = None
    permute_427: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_382, [0, 2, 3, 4, 1]);  view_382 = None
    view_383: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_427, [512, 1, 16, 64]);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_257: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_111, 3)
    unsqueeze_258: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 4);  unsqueeze_257 = None
    permute_428: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_258, [0, 1, 3, 4, 2]);  unsqueeze_258 = None
    unsqueeze_259: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, 3);  arg71_1 = None
    unsqueeze_260: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 4);  unsqueeze_259 = None
    permute_429: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_260, [3, 4, 1, 2, 0]);  unsqueeze_260 = None
    permute_430: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_428, [0, 4, 1, 2, 3]);  permute_428 = None
    view_384: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_430, [1, 512, 1024]);  permute_430 = None
    permute_431: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_429, [4, 1, 2, 3, 0]);  permute_429 = None
    view_385: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_431, [1, 1024, 1024]);  permute_431 = None
    bmm_81: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_384, view_385);  view_384 = view_385 = None
    view_386: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_81, [512, 1, 1, 16, 64]);  bmm_81 = None
    permute_432: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_386, [0, 2, 3, 4, 1]);  view_386 = None
    view_387: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_432, [512, 1, 16, 64]);  permute_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_261: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_111, 3)
    unsqueeze_262: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 4);  unsqueeze_261 = None
    permute_433: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_262, [0, 1, 3, 4, 2]);  unsqueeze_262 = None
    unsqueeze_263: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg72_1, 3);  arg72_1 = None
    unsqueeze_264: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 4);  unsqueeze_263 = None
    permute_434: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_264, [3, 4, 1, 2, 0]);  unsqueeze_264 = None
    permute_435: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_433, [0, 4, 1, 2, 3]);  permute_433 = None
    view_388: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_435, [1, 512, 1024]);  permute_435 = None
    permute_436: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_434, [4, 1, 2, 3, 0]);  permute_434 = None
    view_389: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_436, [1, 1024, 1024]);  permute_436 = None
    bmm_82: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_388, view_389);  view_388 = view_389 = None
    view_390: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_82, [512, 1, 1, 16, 64]);  bmm_82 = None
    permute_437: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_390, [0, 2, 3, 4, 1]);  view_390 = None
    view_391: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_437, [512, 1, 16, 64]);  permute_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_265: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_266: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 4);  unsqueeze_265 = None
    permute_438: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_266, [0, 1, 3, 4, 2]);  unsqueeze_266 = None
    unsqueeze_267: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, 3);  arg73_1 = None
    unsqueeze_268: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 4);  unsqueeze_267 = None
    permute_439: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_268, [3, 4, 1, 2, 0]);  unsqueeze_268 = None
    permute_440: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_438, [0, 4, 1, 2, 3]);  permute_438 = None
    view_392: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_440, [1, 1024, 1024]);  permute_440 = None
    permute_441: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_439, [4, 1, 2, 3, 0]);  permute_439 = None
    view_393: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_441, [1, 1024, 1024]);  permute_441 = None
    bmm_83: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_392, view_393);  view_392 = view_393 = None
    view_394: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_83, [1024, 1, 1, 16, 64]);  bmm_83 = None
    permute_442: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_394, [0, 2, 3, 4, 1]);  view_394 = None
    view_395: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_442, [1024, 1, 16, 64]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_112: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_383, arg74_1);  arg74_1 = None
    unsqueeze_269: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_112, 4);  add_112 = None
    permute_443: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_269, [1, 2, 0, 4, 3]);  unsqueeze_269 = None
    unsqueeze_270: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_387, 4);  view_387 = None
    permute_444: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_270, [1, 2, 4, 0, 3]);  unsqueeze_270 = None
    permute_445: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_443, [1, 2, 4, 0, 3]);  permute_443 = None
    view_396: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_445, [16, 512, 64]);  permute_445 = None
    permute_446: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_444, [1, 4, 0, 3, 2]);  permute_444 = None
    view_397: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_446, [16, 64, 512]);  permute_446 = None
    bmm_84: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_396, view_397);  view_396 = view_397 = None
    view_398: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_84, [16, 512, 1, 1, 512]);  bmm_84 = None
    permute_447: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_398, [3, 0, 1, 4, 2]);  view_398 = None
    view_399: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_447, [1, 16, 512, 512]);  permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_113: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_383, arg75_1);  view_383 = arg75_1 = None
    unsqueeze_271: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_113, 4);  add_113 = None
    permute_448: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_271, [1, 2, 0, 4, 3]);  unsqueeze_271 = None
    unsqueeze_272: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_395, 4);  view_395 = None
    permute_449: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_272, [1, 2, 4, 0, 3]);  unsqueeze_272 = None
    permute_450: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_448, [1, 2, 4, 0, 3]);  permute_448 = None
    view_400: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_450, [16, 512, 64]);  permute_450 = None
    permute_451: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_449, [1, 4, 0, 3, 2]);  permute_449 = None
    view_401: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_451, [16, 64, 1024]);  permute_451 = None
    bmm_85: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_400, view_401);  view_400 = view_401 = None
    view_402: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_85, [16, 512, 1, 1, 1024]);  bmm_85 = None
    permute_452: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_402, [3, 0, 1, 4, 2]);  view_402 = None
    view_403: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_452, [1, 16, 512, 1024]);  permute_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_404: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_403, [1, 16, 1024, 512]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_84: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_404, 0, 0, 9223372036854775807);  view_404 = None
    slice_85: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_84, 1, 0, 9223372036854775807);  slice_84 = None
    slice_86: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_85, 2, 1, 9223372036854775807);  slice_85 = None
    slice_87: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_86, 3, 0, 9223372036854775807);  slice_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_405: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_87, [1, 16, 512, 1023]);  slice_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_12: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_88: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_405, 0, 0, 9223372036854775807);  view_405 = None
    slice_89: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_88, 1, 0, 9223372036854775807);  slice_88 = None
    slice_90: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_89, 2, 0, 9223372036854775807);  slice_89 = None
    index_10: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_90, [None, None, None, iota_12]);  slice_90 = iota_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_114: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_399, index_10);  view_399 = index_10 = None
    add_115: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_114, 0);  add_114 = None
    mul_84: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_115, 0.125);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_10: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_84, [3], True)
    sub_30: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_84, amax_10);  mul_84 = amax_10 = None
    exp_10: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_11: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [3], True)
    div_11: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_62: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_273: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_62, 4);  clone_62 = None
    permute_453: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_273, [2, 0, 1, 4, 3]);  unsqueeze_273 = None
    unsqueeze_274: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_391, 4);  view_391 = None
    permute_454: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_274, [4, 1, 2, 3, 0]);  unsqueeze_274 = None
    permute_455: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_453, [2, 0, 4, 1, 3]);  permute_453 = None
    view_406: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_455, [16, 512, 512]);  permute_455 = None
    permute_456: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_454, [2, 4, 1, 3, 0]);  permute_454 = None
    view_407: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_456, [16, 512, 64]);  permute_456 = None
    bmm_86: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_406, view_407);  view_406 = view_407 = None
    view_408: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_86, [16, 512, 1, 1, 64]);  bmm_86 = None
    permute_457: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_408, [1, 3, 0, 4, 2]);  view_408 = None
    view_409: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_457, [512, 1, 16, 64]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_275: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_409, 4);  view_409 = None
    permute_458: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_275, [0, 1, 4, 3, 2]);  unsqueeze_275 = None
    unsqueeze_276: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, 3);  arg76_1 = None
    unsqueeze_277: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 4);  unsqueeze_276 = None
    permute_459: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_277, [3, 4, 0, 2, 1]);  unsqueeze_277 = None
    permute_460: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_458, [0, 3, 4, 1, 2]);  permute_458 = None
    clone_63: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
    view_410: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_63, [1, 512, 1024]);  clone_63 = None
    permute_461: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_459, [3, 4, 1, 2, 0]);  permute_459 = None
    clone_64: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
    view_411: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_64, [1, 1024, 1024]);  clone_64 = None
    bmm_87: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_410, view_411);  view_410 = view_411 = None
    view_412: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_87, [512, 1, 1, 1, 1024]);  bmm_87 = None
    permute_462: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_412, [0, 3, 4, 1, 2]);  view_412 = None
    view_413: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_462, [512, 1, 1024]);  permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_65: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_413);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_116: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_65, add_111);  clone_65 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
    getitem_40: "f32[512, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[512, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_117: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_31: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_116, getitem_41);  add_116 = getitem_41 = None
    mul_85: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
    mul_86: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_85, arg249_1);  mul_85 = arg249_1 = None
    add_118: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_86, arg250_1);  mul_86 = arg250_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_414: "f32[512, 1024]" = torch.ops.aten.view.default(add_118, [512, 1024])
    permute_463: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
    addmm_20: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg252_1, view_414, permute_463);  arg252_1 = view_414 = permute_463 = None
    view_415: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_20, [512, 1, 4096]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
    mul_88: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
    erf_10: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_119: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_89: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_87, add_119);  mul_87 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_66: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_89);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_416: "f32[512, 4096]" = torch.ops.aten.view.default(clone_66, [512, 4096]);  clone_66 = None
    permute_464: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    addmm_21: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg254_1, view_416, permute_464);  arg254_1 = view_416 = permute_464 = None
    view_417: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_21, [512, 1, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_67: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_417);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_120: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_67, add_118);  clone_67 = add_118 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_120, [2], correction = 0, keepdim = True)
    getitem_42: "f32[512, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[512, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_121: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_32: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_120, getitem_43);  add_120 = getitem_43 = None
    mul_90: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_91: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_90, arg255_1);  mul_90 = arg255_1 = None
    add_122: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_91, arg256_1);  mul_91 = arg256_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_91: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_122, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_22: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_91);  slice_91 = None
    alias_23: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_278: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_122, 3)
    unsqueeze_279: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 4);  unsqueeze_278 = None
    permute_465: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_279, [0, 1, 3, 4, 2]);  unsqueeze_279 = None
    unsqueeze_280: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, 3);  arg77_1 = None
    unsqueeze_281: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 4);  unsqueeze_280 = None
    permute_466: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_281, [3, 4, 1, 2, 0]);  unsqueeze_281 = None
    permute_467: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_465, [0, 4, 1, 2, 3]);  permute_465 = None
    view_418: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_467, [1, 512, 1024]);  permute_467 = None
    permute_468: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_466, [4, 1, 2, 3, 0]);  permute_466 = None
    view_419: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_468, [1, 1024, 1024]);  permute_468 = None
    bmm_88: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_418, view_419);  view_418 = view_419 = None
    view_420: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_88, [512, 1, 1, 16, 64]);  bmm_88 = None
    permute_469: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_420, [0, 2, 3, 4, 1]);  view_420 = None
    view_421: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_469, [512, 1, 16, 64]);  permute_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_282: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_122, 3)
    unsqueeze_283: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 4);  unsqueeze_282 = None
    permute_470: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_283, [0, 1, 3, 4, 2]);  unsqueeze_283 = None
    unsqueeze_284: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, 3);  arg78_1 = None
    unsqueeze_285: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 4);  unsqueeze_284 = None
    permute_471: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_285, [3, 4, 1, 2, 0]);  unsqueeze_285 = None
    permute_472: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_470, [0, 4, 1, 2, 3]);  permute_470 = None
    view_422: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_472, [1, 512, 1024]);  permute_472 = None
    permute_473: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_471, [4, 1, 2, 3, 0]);  permute_471 = None
    view_423: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_473, [1, 1024, 1024]);  permute_473 = None
    bmm_89: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_422, view_423);  view_422 = view_423 = None
    view_424: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_89, [512, 1, 1, 16, 64]);  bmm_89 = None
    permute_474: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_424, [0, 2, 3, 4, 1]);  view_424 = None
    view_425: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_474, [512, 1, 16, 64]);  permute_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_286: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_122, 3)
    unsqueeze_287: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 4);  unsqueeze_286 = None
    permute_475: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_287, [0, 1, 3, 4, 2]);  unsqueeze_287 = None
    unsqueeze_288: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, 3);  arg79_1 = None
    unsqueeze_289: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 4);  unsqueeze_288 = None
    permute_476: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_289, [3, 4, 1, 2, 0]);  unsqueeze_289 = None
    permute_477: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_475, [0, 4, 1, 2, 3]);  permute_475 = None
    view_426: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_477, [1, 512, 1024]);  permute_477 = None
    permute_478: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_476, [4, 1, 2, 3, 0]);  permute_476 = None
    view_427: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_478, [1, 1024, 1024]);  permute_478 = None
    bmm_90: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_426, view_427);  view_426 = view_427 = None
    view_428: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_90, [512, 1, 1, 16, 64]);  bmm_90 = None
    permute_479: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_428, [0, 2, 3, 4, 1]);  view_428 = None
    view_429: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_479, [512, 1, 16, 64]);  permute_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_290: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_291: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 4);  unsqueeze_290 = None
    permute_480: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_291, [0, 1, 3, 4, 2]);  unsqueeze_291 = None
    unsqueeze_292: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, 3);  arg80_1 = None
    unsqueeze_293: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 4);  unsqueeze_292 = None
    permute_481: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_293, [3, 4, 1, 2, 0]);  unsqueeze_293 = None
    permute_482: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_480, [0, 4, 1, 2, 3]);  permute_480 = None
    view_430: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_482, [1, 1024, 1024]);  permute_482 = None
    permute_483: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_481, [4, 1, 2, 3, 0]);  permute_481 = None
    view_431: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_483, [1, 1024, 1024]);  permute_483 = None
    bmm_91: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_430, view_431);  view_430 = view_431 = None
    view_432: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_91, [1024, 1, 1, 16, 64]);  bmm_91 = None
    permute_484: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_432, [0, 2, 3, 4, 1]);  view_432 = None
    view_433: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_484, [1024, 1, 16, 64]);  permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_123: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_421, arg81_1);  arg81_1 = None
    unsqueeze_294: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_123, 4);  add_123 = None
    permute_485: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_294, [1, 2, 0, 4, 3]);  unsqueeze_294 = None
    unsqueeze_295: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_425, 4);  view_425 = None
    permute_486: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_295, [1, 2, 4, 0, 3]);  unsqueeze_295 = None
    permute_487: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_485, [1, 2, 4, 0, 3]);  permute_485 = None
    view_434: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_487, [16, 512, 64]);  permute_487 = None
    permute_488: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_486, [1, 4, 0, 3, 2]);  permute_486 = None
    view_435: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_488, [16, 64, 512]);  permute_488 = None
    bmm_92: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_434, view_435);  view_434 = view_435 = None
    view_436: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_92, [16, 512, 1, 1, 512]);  bmm_92 = None
    permute_489: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_436, [3, 0, 1, 4, 2]);  view_436 = None
    view_437: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_489, [1, 16, 512, 512]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_124: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_421, arg82_1);  view_421 = arg82_1 = None
    unsqueeze_296: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_124, 4);  add_124 = None
    permute_490: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_296, [1, 2, 0, 4, 3]);  unsqueeze_296 = None
    unsqueeze_297: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_433, 4);  view_433 = None
    permute_491: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_297, [1, 2, 4, 0, 3]);  unsqueeze_297 = None
    permute_492: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_490, [1, 2, 4, 0, 3]);  permute_490 = None
    view_438: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_492, [16, 512, 64]);  permute_492 = None
    permute_493: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_491, [1, 4, 0, 3, 2]);  permute_491 = None
    view_439: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_493, [16, 64, 1024]);  permute_493 = None
    bmm_93: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_438, view_439);  view_438 = view_439 = None
    view_440: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_93, [16, 512, 1, 1, 1024]);  bmm_93 = None
    permute_494: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_440, [3, 0, 1, 4, 2]);  view_440 = None
    view_441: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_494, [1, 16, 512, 1024]);  permute_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_442: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_441, [1, 16, 1024, 512]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_92: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_442, 0, 0, 9223372036854775807);  view_442 = None
    slice_93: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_92, 1, 0, 9223372036854775807);  slice_92 = None
    slice_94: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_93, 2, 1, 9223372036854775807);  slice_93 = None
    slice_95: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_94, 3, 0, 9223372036854775807);  slice_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_443: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_95, [1, 16, 512, 1023]);  slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_13: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_96: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_443, 0, 0, 9223372036854775807);  view_443 = None
    slice_97: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_96, 1, 0, 9223372036854775807);  slice_96 = None
    slice_98: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_97, 2, 0, 9223372036854775807);  slice_97 = None
    index_11: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_98, [None, None, None, iota_13]);  slice_98 = iota_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_125: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_437, index_11);  view_437 = index_11 = None
    add_126: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_125, 0);  add_125 = None
    mul_92: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_126, 0.125);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_11: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_92, [3], True)
    sub_33: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_92, amax_11);  mul_92 = amax_11 = None
    exp_11: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_12: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [3], True)
    div_12: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_68: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_298: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_68, 4);  clone_68 = None
    permute_495: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_298, [2, 0, 1, 4, 3]);  unsqueeze_298 = None
    unsqueeze_299: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_429, 4);  view_429 = None
    permute_496: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_299, [4, 1, 2, 3, 0]);  unsqueeze_299 = None
    permute_497: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_495, [2, 0, 4, 1, 3]);  permute_495 = None
    view_444: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_497, [16, 512, 512]);  permute_497 = None
    permute_498: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_496, [2, 4, 1, 3, 0]);  permute_496 = None
    view_445: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_498, [16, 512, 64]);  permute_498 = None
    bmm_94: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_444, view_445);  view_444 = view_445 = None
    view_446: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_94, [16, 512, 1, 1, 64]);  bmm_94 = None
    permute_499: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_446, [1, 3, 0, 4, 2]);  view_446 = None
    view_447: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_499, [512, 1, 16, 64]);  permute_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_300: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_447, 4);  view_447 = None
    permute_500: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_300, [0, 1, 4, 3, 2]);  unsqueeze_300 = None
    unsqueeze_301: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, 3);  arg83_1 = None
    unsqueeze_302: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 4);  unsqueeze_301 = None
    permute_501: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_302, [3, 4, 0, 2, 1]);  unsqueeze_302 = None
    permute_502: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_500, [0, 3, 4, 1, 2]);  permute_500 = None
    clone_69: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
    view_448: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_69, [1, 512, 1024]);  clone_69 = None
    permute_503: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_501, [3, 4, 1, 2, 0]);  permute_501 = None
    clone_70: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_503, memory_format = torch.contiguous_format);  permute_503 = None
    view_449: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_70, [1, 1024, 1024]);  clone_70 = None
    bmm_95: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_448, view_449);  view_448 = view_449 = None
    view_450: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_95, [512, 1, 1, 1, 1024]);  bmm_95 = None
    permute_504: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_450, [0, 3, 4, 1, 2]);  view_450 = None
    view_451: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_504, [512, 1, 1024]);  permute_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_71: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_451);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_127: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_71, add_122);  clone_71 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_127, [2], correction = 0, keepdim = True)
    getitem_44: "f32[512, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[512, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_128: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_34: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_127, getitem_45);  add_127 = getitem_45 = None
    mul_93: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
    mul_94: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_93, arg257_1);  mul_93 = arg257_1 = None
    add_129: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_94, arg258_1);  mul_94 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_452: "f32[512, 1024]" = torch.ops.aten.view.default(add_129, [512, 1024])
    permute_505: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
    addmm_22: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg260_1, view_452, permute_505);  arg260_1 = view_452 = permute_505 = None
    view_453: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_22, [512, 1, 4096]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_95: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_453, 0.5)
    mul_96: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_453, 0.7071067811865476);  view_453 = None
    erf_11: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_130: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_97: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_95, add_130);  mul_95 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_72: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_454: "f32[512, 4096]" = torch.ops.aten.view.default(clone_72, [512, 4096]);  clone_72 = None
    permute_506: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
    addmm_23: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg262_1, view_454, permute_506);  arg262_1 = view_454 = permute_506 = None
    view_455: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_23, [512, 1, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_73: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_455);  view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_131: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_73, add_129);  clone_73 = add_129 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
    getitem_46: "f32[512, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[512, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_132: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_35: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_131, getitem_47);  add_131 = getitem_47 = None
    mul_98: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_99: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_98, arg263_1);  mul_98 = arg263_1 = None
    add_133: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_99, arg264_1);  mul_99 = arg264_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_99: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_133, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_24: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_99);  slice_99 = None
    alias_25: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_303: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_133, 3)
    unsqueeze_304: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 4);  unsqueeze_303 = None
    permute_507: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_304, [0, 1, 3, 4, 2]);  unsqueeze_304 = None
    unsqueeze_305: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, 3);  arg84_1 = None
    unsqueeze_306: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 4);  unsqueeze_305 = None
    permute_508: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_306, [3, 4, 1, 2, 0]);  unsqueeze_306 = None
    permute_509: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_507, [0, 4, 1, 2, 3]);  permute_507 = None
    view_456: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_509, [1, 512, 1024]);  permute_509 = None
    permute_510: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_508, [4, 1, 2, 3, 0]);  permute_508 = None
    view_457: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_510, [1, 1024, 1024]);  permute_510 = None
    bmm_96: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_456, view_457);  view_456 = view_457 = None
    view_458: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_96, [512, 1, 1, 16, 64]);  bmm_96 = None
    permute_511: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_458, [0, 2, 3, 4, 1]);  view_458 = None
    view_459: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_511, [512, 1, 16, 64]);  permute_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_307: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_133, 3)
    unsqueeze_308: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 4);  unsqueeze_307 = None
    permute_512: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_308, [0, 1, 3, 4, 2]);  unsqueeze_308 = None
    unsqueeze_309: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, 3);  arg85_1 = None
    unsqueeze_310: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 4);  unsqueeze_309 = None
    permute_513: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_310, [3, 4, 1, 2, 0]);  unsqueeze_310 = None
    permute_514: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_512, [0, 4, 1, 2, 3]);  permute_512 = None
    view_460: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_514, [1, 512, 1024]);  permute_514 = None
    permute_515: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_513, [4, 1, 2, 3, 0]);  permute_513 = None
    view_461: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_515, [1, 1024, 1024]);  permute_515 = None
    bmm_97: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_460, view_461);  view_460 = view_461 = None
    view_462: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_97, [512, 1, 1, 16, 64]);  bmm_97 = None
    permute_516: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_462, [0, 2, 3, 4, 1]);  view_462 = None
    view_463: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_516, [512, 1, 16, 64]);  permute_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_311: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_133, 3)
    unsqueeze_312: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 4);  unsqueeze_311 = None
    permute_517: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_312, [0, 1, 3, 4, 2]);  unsqueeze_312 = None
    unsqueeze_313: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, 3);  arg86_1 = None
    unsqueeze_314: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 4);  unsqueeze_313 = None
    permute_518: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_314, [3, 4, 1, 2, 0]);  unsqueeze_314 = None
    permute_519: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_517, [0, 4, 1, 2, 3]);  permute_517 = None
    view_464: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_519, [1, 512, 1024]);  permute_519 = None
    permute_520: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_518, [4, 1, 2, 3, 0]);  permute_518 = None
    view_465: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_520, [1, 1024, 1024]);  permute_520 = None
    bmm_98: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_464, view_465);  view_464 = view_465 = None
    view_466: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_98, [512, 1, 1, 16, 64]);  bmm_98 = None
    permute_521: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_466, [0, 2, 3, 4, 1]);  view_466 = None
    view_467: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_521, [512, 1, 16, 64]);  permute_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_315: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_316: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 4);  unsqueeze_315 = None
    permute_522: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_316, [0, 1, 3, 4, 2]);  unsqueeze_316 = None
    unsqueeze_317: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, 3);  arg87_1 = None
    unsqueeze_318: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 4);  unsqueeze_317 = None
    permute_523: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_318, [3, 4, 1, 2, 0]);  unsqueeze_318 = None
    permute_524: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_522, [0, 4, 1, 2, 3]);  permute_522 = None
    view_468: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_524, [1, 1024, 1024]);  permute_524 = None
    permute_525: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_523, [4, 1, 2, 3, 0]);  permute_523 = None
    view_469: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_525, [1, 1024, 1024]);  permute_525 = None
    bmm_99: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_468, view_469);  view_468 = view_469 = None
    view_470: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_99, [1024, 1, 1, 16, 64]);  bmm_99 = None
    permute_526: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_470, [0, 2, 3, 4, 1]);  view_470 = None
    view_471: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_526, [1024, 1, 16, 64]);  permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_134: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_459, arg88_1);  arg88_1 = None
    unsqueeze_319: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_134, 4);  add_134 = None
    permute_527: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_319, [1, 2, 0, 4, 3]);  unsqueeze_319 = None
    unsqueeze_320: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_463, 4);  view_463 = None
    permute_528: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_320, [1, 2, 4, 0, 3]);  unsqueeze_320 = None
    permute_529: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_527, [1, 2, 4, 0, 3]);  permute_527 = None
    view_472: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_529, [16, 512, 64]);  permute_529 = None
    permute_530: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_528, [1, 4, 0, 3, 2]);  permute_528 = None
    view_473: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_530, [16, 64, 512]);  permute_530 = None
    bmm_100: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_472, view_473);  view_472 = view_473 = None
    view_474: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_100, [16, 512, 1, 1, 512]);  bmm_100 = None
    permute_531: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_474, [3, 0, 1, 4, 2]);  view_474 = None
    view_475: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_531, [1, 16, 512, 512]);  permute_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_135: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_459, arg89_1);  view_459 = arg89_1 = None
    unsqueeze_321: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_135, 4);  add_135 = None
    permute_532: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_321, [1, 2, 0, 4, 3]);  unsqueeze_321 = None
    unsqueeze_322: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_471, 4);  view_471 = None
    permute_533: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_322, [1, 2, 4, 0, 3]);  unsqueeze_322 = None
    permute_534: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_532, [1, 2, 4, 0, 3]);  permute_532 = None
    view_476: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_534, [16, 512, 64]);  permute_534 = None
    permute_535: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_533, [1, 4, 0, 3, 2]);  permute_533 = None
    view_477: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_535, [16, 64, 1024]);  permute_535 = None
    bmm_101: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_476, view_477);  view_476 = view_477 = None
    view_478: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_101, [16, 512, 1, 1, 1024]);  bmm_101 = None
    permute_536: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_478, [3, 0, 1, 4, 2]);  view_478 = None
    view_479: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_536, [1, 16, 512, 1024]);  permute_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_480: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_479, [1, 16, 1024, 512]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_100: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_480, 0, 0, 9223372036854775807);  view_480 = None
    slice_101: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_100, 1, 0, 9223372036854775807);  slice_100 = None
    slice_102: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_101, 2, 1, 9223372036854775807);  slice_101 = None
    slice_103: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_102, 3, 0, 9223372036854775807);  slice_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_481: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_103, [1, 16, 512, 1023]);  slice_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_14: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_104: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_481, 0, 0, 9223372036854775807);  view_481 = None
    slice_105: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_104, 1, 0, 9223372036854775807);  slice_104 = None
    slice_106: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_105, 2, 0, 9223372036854775807);  slice_105 = None
    index_12: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_106, [None, None, None, iota_14]);  slice_106 = iota_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_136: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_475, index_12);  view_475 = index_12 = None
    add_137: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_136, 0);  add_136 = None
    mul_100: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_137, 0.125);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_12: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_100, [3], True)
    sub_36: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_100, amax_12);  mul_100 = amax_12 = None
    exp_12: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_13: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [3], True)
    div_13: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_74: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_323: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_74, 4);  clone_74 = None
    permute_537: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_323, [2, 0, 1, 4, 3]);  unsqueeze_323 = None
    unsqueeze_324: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_467, 4);  view_467 = None
    permute_538: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_324, [4, 1, 2, 3, 0]);  unsqueeze_324 = None
    permute_539: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_537, [2, 0, 4, 1, 3]);  permute_537 = None
    view_482: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_539, [16, 512, 512]);  permute_539 = None
    permute_540: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_538, [2, 4, 1, 3, 0]);  permute_538 = None
    view_483: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_540, [16, 512, 64]);  permute_540 = None
    bmm_102: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_482, view_483);  view_482 = view_483 = None
    view_484: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_102, [16, 512, 1, 1, 64]);  bmm_102 = None
    permute_541: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_484, [1, 3, 0, 4, 2]);  view_484 = None
    view_485: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_541, [512, 1, 16, 64]);  permute_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_325: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_485, 4);  view_485 = None
    permute_542: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_325, [0, 1, 4, 3, 2]);  unsqueeze_325 = None
    unsqueeze_326: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, 3);  arg90_1 = None
    unsqueeze_327: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 4);  unsqueeze_326 = None
    permute_543: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_327, [3, 4, 0, 2, 1]);  unsqueeze_327 = None
    permute_544: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_542, [0, 3, 4, 1, 2]);  permute_542 = None
    clone_75: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_544, memory_format = torch.contiguous_format);  permute_544 = None
    view_486: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_75, [1, 512, 1024]);  clone_75 = None
    permute_545: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_543, [3, 4, 1, 2, 0]);  permute_543 = None
    clone_76: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_545, memory_format = torch.contiguous_format);  permute_545 = None
    view_487: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_76, [1, 1024, 1024]);  clone_76 = None
    bmm_103: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_486, view_487);  view_486 = view_487 = None
    view_488: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_103, [512, 1, 1, 1, 1024]);  bmm_103 = None
    permute_546: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_488, [0, 3, 4, 1, 2]);  view_488 = None
    view_489: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_546, [512, 1, 1024]);  permute_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_77: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_489);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_138: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_77, add_133);  clone_77 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_138, [2], correction = 0, keepdim = True)
    getitem_48: "f32[512, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[512, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_139: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    sub_37: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_138, getitem_49);  add_138 = getitem_49 = None
    mul_101: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
    mul_102: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_101, arg265_1);  mul_101 = arg265_1 = None
    add_140: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_102, arg266_1);  mul_102 = arg266_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_490: "f32[512, 1024]" = torch.ops.aten.view.default(add_140, [512, 1024])
    permute_547: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
    addmm_24: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg268_1, view_490, permute_547);  arg268_1 = view_490 = permute_547 = None
    view_491: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_24, [512, 1, 4096]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_103: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_491, 0.5)
    mul_104: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_491, 0.7071067811865476);  view_491 = None
    erf_12: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
    add_141: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_105: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_103, add_141);  mul_103 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_78: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_105);  mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_492: "f32[512, 4096]" = torch.ops.aten.view.default(clone_78, [512, 4096]);  clone_78 = None
    permute_548: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
    addmm_25: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg270_1, view_492, permute_548);  arg270_1 = view_492 = permute_548 = None
    view_493: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_25, [512, 1, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_79: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_493);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_142: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_79, add_140);  clone_79 = add_140 = None
    var_mean_25 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
    getitem_50: "f32[512, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[512, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_143: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_38: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_142, getitem_51);  add_142 = getitem_51 = None
    mul_106: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
    mul_107: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_106, arg271_1);  mul_106 = arg271_1 = None
    add_144: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_107, arg272_1);  mul_107 = arg272_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_107: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_144, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_26: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_107);  slice_107 = None
    alias_27: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_328: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_144, 3)
    unsqueeze_329: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 4);  unsqueeze_328 = None
    permute_549: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_329, [0, 1, 3, 4, 2]);  unsqueeze_329 = None
    unsqueeze_330: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, 3);  arg91_1 = None
    unsqueeze_331: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 4);  unsqueeze_330 = None
    permute_550: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_331, [3, 4, 1, 2, 0]);  unsqueeze_331 = None
    permute_551: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_549, [0, 4, 1, 2, 3]);  permute_549 = None
    view_494: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_551, [1, 512, 1024]);  permute_551 = None
    permute_552: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_550, [4, 1, 2, 3, 0]);  permute_550 = None
    view_495: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_552, [1, 1024, 1024]);  permute_552 = None
    bmm_104: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_494, view_495);  view_494 = view_495 = None
    view_496: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_104, [512, 1, 1, 16, 64]);  bmm_104 = None
    permute_553: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_496, [0, 2, 3, 4, 1]);  view_496 = None
    view_497: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_553, [512, 1, 16, 64]);  permute_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_332: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_144, 3)
    unsqueeze_333: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 4);  unsqueeze_332 = None
    permute_554: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_333, [0, 1, 3, 4, 2]);  unsqueeze_333 = None
    unsqueeze_334: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, 3);  arg92_1 = None
    unsqueeze_335: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 4);  unsqueeze_334 = None
    permute_555: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_335, [3, 4, 1, 2, 0]);  unsqueeze_335 = None
    permute_556: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_554, [0, 4, 1, 2, 3]);  permute_554 = None
    view_498: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_556, [1, 512, 1024]);  permute_556 = None
    permute_557: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_555, [4, 1, 2, 3, 0]);  permute_555 = None
    view_499: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_557, [1, 1024, 1024]);  permute_557 = None
    bmm_105: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_498, view_499);  view_498 = view_499 = None
    view_500: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_105, [512, 1, 1, 16, 64]);  bmm_105 = None
    permute_558: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_500, [0, 2, 3, 4, 1]);  view_500 = None
    view_501: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_558, [512, 1, 16, 64]);  permute_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_336: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_144, 3)
    unsqueeze_337: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 4);  unsqueeze_336 = None
    permute_559: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_337, [0, 1, 3, 4, 2]);  unsqueeze_337 = None
    unsqueeze_338: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, 3);  arg93_1 = None
    unsqueeze_339: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 4);  unsqueeze_338 = None
    permute_560: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_339, [3, 4, 1, 2, 0]);  unsqueeze_339 = None
    permute_561: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_559, [0, 4, 1, 2, 3]);  permute_559 = None
    view_502: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_561, [1, 512, 1024]);  permute_561 = None
    permute_562: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_560, [4, 1, 2, 3, 0]);  permute_560 = None
    view_503: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_562, [1, 1024, 1024]);  permute_562 = None
    bmm_106: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_502, view_503);  view_502 = view_503 = None
    view_504: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_106, [512, 1, 1, 16, 64]);  bmm_106 = None
    permute_563: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_504, [0, 2, 3, 4, 1]);  view_504 = None
    view_505: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_563, [512, 1, 16, 64]);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_340: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_341: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 4);  unsqueeze_340 = None
    permute_564: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_341, [0, 1, 3, 4, 2]);  unsqueeze_341 = None
    unsqueeze_342: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, 3);  arg94_1 = None
    unsqueeze_343: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 4);  unsqueeze_342 = None
    permute_565: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_343, [3, 4, 1, 2, 0]);  unsqueeze_343 = None
    permute_566: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_564, [0, 4, 1, 2, 3]);  permute_564 = None
    view_506: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_566, [1, 1024, 1024]);  permute_566 = None
    permute_567: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_565, [4, 1, 2, 3, 0]);  permute_565 = None
    view_507: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_567, [1, 1024, 1024]);  permute_567 = None
    bmm_107: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_506, view_507);  view_506 = view_507 = None
    view_508: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_107, [1024, 1, 1, 16, 64]);  bmm_107 = None
    permute_568: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_508, [0, 2, 3, 4, 1]);  view_508 = None
    view_509: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_568, [1024, 1, 16, 64]);  permute_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_145: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_497, arg95_1);  arg95_1 = None
    unsqueeze_344: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_145, 4);  add_145 = None
    permute_569: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_344, [1, 2, 0, 4, 3]);  unsqueeze_344 = None
    unsqueeze_345: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_501, 4);  view_501 = None
    permute_570: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_345, [1, 2, 4, 0, 3]);  unsqueeze_345 = None
    permute_571: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_569, [1, 2, 4, 0, 3]);  permute_569 = None
    view_510: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_571, [16, 512, 64]);  permute_571 = None
    permute_572: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_570, [1, 4, 0, 3, 2]);  permute_570 = None
    view_511: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_572, [16, 64, 512]);  permute_572 = None
    bmm_108: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_510, view_511);  view_510 = view_511 = None
    view_512: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_108, [16, 512, 1, 1, 512]);  bmm_108 = None
    permute_573: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_512, [3, 0, 1, 4, 2]);  view_512 = None
    view_513: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_573, [1, 16, 512, 512]);  permute_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_146: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_497, arg96_1);  view_497 = arg96_1 = None
    unsqueeze_346: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_146, 4);  add_146 = None
    permute_574: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_346, [1, 2, 0, 4, 3]);  unsqueeze_346 = None
    unsqueeze_347: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_509, 4);  view_509 = None
    permute_575: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_347, [1, 2, 4, 0, 3]);  unsqueeze_347 = None
    permute_576: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_574, [1, 2, 4, 0, 3]);  permute_574 = None
    view_514: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_576, [16, 512, 64]);  permute_576 = None
    permute_577: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_575, [1, 4, 0, 3, 2]);  permute_575 = None
    view_515: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_577, [16, 64, 1024]);  permute_577 = None
    bmm_109: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_514, view_515);  view_514 = view_515 = None
    view_516: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_109, [16, 512, 1, 1, 1024]);  bmm_109 = None
    permute_578: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_516, [3, 0, 1, 4, 2]);  view_516 = None
    view_517: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_578, [1, 16, 512, 1024]);  permute_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_518: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_517, [1, 16, 1024, 512]);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_108: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_518, 0, 0, 9223372036854775807);  view_518 = None
    slice_109: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_108, 1, 0, 9223372036854775807);  slice_108 = None
    slice_110: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_109, 2, 1, 9223372036854775807);  slice_109 = None
    slice_111: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_110, 3, 0, 9223372036854775807);  slice_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_519: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_111, [1, 16, 512, 1023]);  slice_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_15: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_112: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_519, 0, 0, 9223372036854775807);  view_519 = None
    slice_113: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_112, 1, 0, 9223372036854775807);  slice_112 = None
    slice_114: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_113, 2, 0, 9223372036854775807);  slice_113 = None
    index_13: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_114, [None, None, None, iota_15]);  slice_114 = iota_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_147: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_513, index_13);  view_513 = index_13 = None
    add_148: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_147, 0);  add_147 = None
    mul_108: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_148, 0.125);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_13: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_108, [3], True)
    sub_39: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_108, amax_13);  mul_108 = amax_13 = None
    exp_13: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_14: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [3], True)
    div_14: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_80: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_348: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_80, 4);  clone_80 = None
    permute_579: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_348, [2, 0, 1, 4, 3]);  unsqueeze_348 = None
    unsqueeze_349: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_505, 4);  view_505 = None
    permute_580: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_349, [4, 1, 2, 3, 0]);  unsqueeze_349 = None
    permute_581: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_579, [2, 0, 4, 1, 3]);  permute_579 = None
    view_520: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_581, [16, 512, 512]);  permute_581 = None
    permute_582: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_580, [2, 4, 1, 3, 0]);  permute_580 = None
    view_521: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_582, [16, 512, 64]);  permute_582 = None
    bmm_110: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_520, view_521);  view_520 = view_521 = None
    view_522: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_110, [16, 512, 1, 1, 64]);  bmm_110 = None
    permute_583: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_522, [1, 3, 0, 4, 2]);  view_522 = None
    view_523: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_583, [512, 1, 16, 64]);  permute_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_350: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_523, 4);  view_523 = None
    permute_584: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_350, [0, 1, 4, 3, 2]);  unsqueeze_350 = None
    unsqueeze_351: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, 3);  arg97_1 = None
    unsqueeze_352: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 4);  unsqueeze_351 = None
    permute_585: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_352, [3, 4, 0, 2, 1]);  unsqueeze_352 = None
    permute_586: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_584, [0, 3, 4, 1, 2]);  permute_584 = None
    clone_81: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
    view_524: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_81, [1, 512, 1024]);  clone_81 = None
    permute_587: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_585, [3, 4, 1, 2, 0]);  permute_585 = None
    clone_82: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
    view_525: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_82, [1, 1024, 1024]);  clone_82 = None
    bmm_111: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_524, view_525);  view_524 = view_525 = None
    view_526: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_111, [512, 1, 1, 1, 1024]);  bmm_111 = None
    permute_588: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_526, [0, 3, 4, 1, 2]);  view_526 = None
    view_527: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_588, [512, 1, 1024]);  permute_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_83: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_527);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_149: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_83, add_144);  clone_83 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
    getitem_52: "f32[512, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[512, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_150: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_26: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_40: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_149, getitem_53);  add_149 = getitem_53 = None
    mul_109: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = rsqrt_26 = None
    mul_110: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_109, arg273_1);  mul_109 = arg273_1 = None
    add_151: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_110, arg274_1);  mul_110 = arg274_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_528: "f32[512, 1024]" = torch.ops.aten.view.default(add_151, [512, 1024])
    permute_589: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
    addmm_26: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg276_1, view_528, permute_589);  arg276_1 = view_528 = permute_589 = None
    view_529: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_26, [512, 1, 4096]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_111: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
    mul_112: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476);  view_529 = None
    erf_13: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
    add_152: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_113: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_111, add_152);  mul_111 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_84: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_113);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_530: "f32[512, 4096]" = torch.ops.aten.view.default(clone_84, [512, 4096]);  clone_84 = None
    permute_590: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    addmm_27: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg278_1, view_530, permute_590);  arg278_1 = view_530 = permute_590 = None
    view_531: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_27, [512, 1, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_85: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_531);  view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_153: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_85, add_151);  clone_85 = add_151 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
    getitem_54: "f32[512, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[512, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_154: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
    rsqrt_27: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_41: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_153, getitem_55);  add_153 = getitem_55 = None
    mul_114: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
    mul_115: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_114, arg279_1);  mul_114 = arg279_1 = None
    add_155: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_115, arg280_1);  mul_115 = arg280_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_115: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_155, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_28: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_115);  slice_115 = None
    alias_29: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_353: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_155, 3)
    unsqueeze_354: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 4);  unsqueeze_353 = None
    permute_591: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_354, [0, 1, 3, 4, 2]);  unsqueeze_354 = None
    unsqueeze_355: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, 3);  arg98_1 = None
    unsqueeze_356: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 4);  unsqueeze_355 = None
    permute_592: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_356, [3, 4, 1, 2, 0]);  unsqueeze_356 = None
    permute_593: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_591, [0, 4, 1, 2, 3]);  permute_591 = None
    view_532: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_593, [1, 512, 1024]);  permute_593 = None
    permute_594: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_592, [4, 1, 2, 3, 0]);  permute_592 = None
    view_533: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_594, [1, 1024, 1024]);  permute_594 = None
    bmm_112: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_532, view_533);  view_532 = view_533 = None
    view_534: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_112, [512, 1, 1, 16, 64]);  bmm_112 = None
    permute_595: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_534, [0, 2, 3, 4, 1]);  view_534 = None
    view_535: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_595, [512, 1, 16, 64]);  permute_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_357: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_155, 3)
    unsqueeze_358: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 4);  unsqueeze_357 = None
    permute_596: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_358, [0, 1, 3, 4, 2]);  unsqueeze_358 = None
    unsqueeze_359: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, 3);  arg99_1 = None
    unsqueeze_360: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 4);  unsqueeze_359 = None
    permute_597: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_360, [3, 4, 1, 2, 0]);  unsqueeze_360 = None
    permute_598: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_596, [0, 4, 1, 2, 3]);  permute_596 = None
    view_536: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_598, [1, 512, 1024]);  permute_598 = None
    permute_599: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_597, [4, 1, 2, 3, 0]);  permute_597 = None
    view_537: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_599, [1, 1024, 1024]);  permute_599 = None
    bmm_113: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_536, view_537);  view_536 = view_537 = None
    view_538: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_113, [512, 1, 1, 16, 64]);  bmm_113 = None
    permute_600: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_538, [0, 2, 3, 4, 1]);  view_538 = None
    view_539: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_600, [512, 1, 16, 64]);  permute_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_361: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_155, 3)
    unsqueeze_362: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 4);  unsqueeze_361 = None
    permute_601: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_362, [0, 1, 3, 4, 2]);  unsqueeze_362 = None
    unsqueeze_363: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, 3);  arg100_1 = None
    unsqueeze_364: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 4);  unsqueeze_363 = None
    permute_602: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_364, [3, 4, 1, 2, 0]);  unsqueeze_364 = None
    permute_603: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_601, [0, 4, 1, 2, 3]);  permute_601 = None
    view_540: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_603, [1, 512, 1024]);  permute_603 = None
    permute_604: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_602, [4, 1, 2, 3, 0]);  permute_602 = None
    view_541: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_604, [1, 1024, 1024]);  permute_604 = None
    bmm_114: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_540, view_541);  view_540 = view_541 = None
    view_542: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_114, [512, 1, 1, 16, 64]);  bmm_114 = None
    permute_605: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_542, [0, 2, 3, 4, 1]);  view_542 = None
    view_543: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_605, [512, 1, 16, 64]);  permute_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_365: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_366: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 4);  unsqueeze_365 = None
    permute_606: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_366, [0, 1, 3, 4, 2]);  unsqueeze_366 = None
    unsqueeze_367: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, 3);  arg101_1 = None
    unsqueeze_368: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 4);  unsqueeze_367 = None
    permute_607: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_368, [3, 4, 1, 2, 0]);  unsqueeze_368 = None
    permute_608: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_606, [0, 4, 1, 2, 3]);  permute_606 = None
    view_544: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_608, [1, 1024, 1024]);  permute_608 = None
    permute_609: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_607, [4, 1, 2, 3, 0]);  permute_607 = None
    view_545: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_609, [1, 1024, 1024]);  permute_609 = None
    bmm_115: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_544, view_545);  view_544 = view_545 = None
    view_546: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_115, [1024, 1, 1, 16, 64]);  bmm_115 = None
    permute_610: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_546, [0, 2, 3, 4, 1]);  view_546 = None
    view_547: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_610, [1024, 1, 16, 64]);  permute_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_156: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_535, arg102_1);  arg102_1 = None
    unsqueeze_369: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_156, 4);  add_156 = None
    permute_611: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_369, [1, 2, 0, 4, 3]);  unsqueeze_369 = None
    unsqueeze_370: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_539, 4);  view_539 = None
    permute_612: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_370, [1, 2, 4, 0, 3]);  unsqueeze_370 = None
    permute_613: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_611, [1, 2, 4, 0, 3]);  permute_611 = None
    view_548: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_613, [16, 512, 64]);  permute_613 = None
    permute_614: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_612, [1, 4, 0, 3, 2]);  permute_612 = None
    view_549: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_614, [16, 64, 512]);  permute_614 = None
    bmm_116: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_548, view_549);  view_548 = view_549 = None
    view_550: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_116, [16, 512, 1, 1, 512]);  bmm_116 = None
    permute_615: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_550, [3, 0, 1, 4, 2]);  view_550 = None
    view_551: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_615, [1, 16, 512, 512]);  permute_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_157: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_535, arg103_1);  view_535 = arg103_1 = None
    unsqueeze_371: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_157, 4);  add_157 = None
    permute_616: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_371, [1, 2, 0, 4, 3]);  unsqueeze_371 = None
    unsqueeze_372: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_547, 4);  view_547 = None
    permute_617: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_372, [1, 2, 4, 0, 3]);  unsqueeze_372 = None
    permute_618: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_616, [1, 2, 4, 0, 3]);  permute_616 = None
    view_552: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_618, [16, 512, 64]);  permute_618 = None
    permute_619: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_617, [1, 4, 0, 3, 2]);  permute_617 = None
    view_553: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_619, [16, 64, 1024]);  permute_619 = None
    bmm_117: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_552, view_553);  view_552 = view_553 = None
    view_554: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_117, [16, 512, 1, 1, 1024]);  bmm_117 = None
    permute_620: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_554, [3, 0, 1, 4, 2]);  view_554 = None
    view_555: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_620, [1, 16, 512, 1024]);  permute_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_556: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_555, [1, 16, 1024, 512]);  view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_116: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_556, 0, 0, 9223372036854775807);  view_556 = None
    slice_117: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_116, 1, 0, 9223372036854775807);  slice_116 = None
    slice_118: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_117, 2, 1, 9223372036854775807);  slice_117 = None
    slice_119: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_118, 3, 0, 9223372036854775807);  slice_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_557: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_119, [1, 16, 512, 1023]);  slice_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_16: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_120: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_557, 0, 0, 9223372036854775807);  view_557 = None
    slice_121: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_120, 1, 0, 9223372036854775807);  slice_120 = None
    slice_122: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_121, 2, 0, 9223372036854775807);  slice_121 = None
    index_14: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_122, [None, None, None, iota_16]);  slice_122 = iota_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_158: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_551, index_14);  view_551 = index_14 = None
    add_159: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_158, 0);  add_158 = None
    mul_116: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_159, 0.125);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_14: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_116, [3], True)
    sub_42: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_116, amax_14);  mul_116 = amax_14 = None
    exp_14: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_15: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [3], True)
    div_15: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_86: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_373: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_86, 4);  clone_86 = None
    permute_621: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_373, [2, 0, 1, 4, 3]);  unsqueeze_373 = None
    unsqueeze_374: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_543, 4);  view_543 = None
    permute_622: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_374, [4, 1, 2, 3, 0]);  unsqueeze_374 = None
    permute_623: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_621, [2, 0, 4, 1, 3]);  permute_621 = None
    view_558: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_623, [16, 512, 512]);  permute_623 = None
    permute_624: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_622, [2, 4, 1, 3, 0]);  permute_622 = None
    view_559: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_624, [16, 512, 64]);  permute_624 = None
    bmm_118: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_558, view_559);  view_558 = view_559 = None
    view_560: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_118, [16, 512, 1, 1, 64]);  bmm_118 = None
    permute_625: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_560, [1, 3, 0, 4, 2]);  view_560 = None
    view_561: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_625, [512, 1, 16, 64]);  permute_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_375: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_561, 4);  view_561 = None
    permute_626: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_375, [0, 1, 4, 3, 2]);  unsqueeze_375 = None
    unsqueeze_376: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, 3);  arg104_1 = None
    unsqueeze_377: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 4);  unsqueeze_376 = None
    permute_627: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_377, [3, 4, 0, 2, 1]);  unsqueeze_377 = None
    permute_628: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_626, [0, 3, 4, 1, 2]);  permute_626 = None
    clone_87: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_628, memory_format = torch.contiguous_format);  permute_628 = None
    view_562: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_87, [1, 512, 1024]);  clone_87 = None
    permute_629: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_627, [3, 4, 1, 2, 0]);  permute_627 = None
    clone_88: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
    view_563: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_88, [1, 1024, 1024]);  clone_88 = None
    bmm_119: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_562, view_563);  view_562 = view_563 = None
    view_564: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_119, [512, 1, 1, 1, 1024]);  bmm_119 = None
    permute_630: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_564, [0, 3, 4, 1, 2]);  view_564 = None
    view_565: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_630, [512, 1, 1024]);  permute_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_89: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_565);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_160: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_89, add_155);  clone_89 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_56: "f32[512, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[512, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_161: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
    rsqrt_28: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_43: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_160, getitem_57);  add_160 = getitem_57 = None
    mul_117: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = rsqrt_28 = None
    mul_118: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_117, arg281_1);  mul_117 = arg281_1 = None
    add_162: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_118, arg282_1);  mul_118 = arg282_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_566: "f32[512, 1024]" = torch.ops.aten.view.default(add_162, [512, 1024])
    permute_631: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
    addmm_28: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg284_1, view_566, permute_631);  arg284_1 = view_566 = permute_631 = None
    view_567: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_28, [512, 1, 4096]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_119: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_567, 0.5)
    mul_120: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_567, 0.7071067811865476);  view_567 = None
    erf_14: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
    add_163: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_121: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_119, add_163);  mul_119 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_90: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_121);  mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_568: "f32[512, 4096]" = torch.ops.aten.view.default(clone_90, [512, 4096]);  clone_90 = None
    permute_632: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
    addmm_29: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg286_1, view_568, permute_632);  arg286_1 = view_568 = permute_632 = None
    view_569: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_29, [512, 1, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_91: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_569);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_164: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_91, add_162);  clone_91 = add_162 = None
    var_mean_29 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_58: "f32[512, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[512, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_165: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_29: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_44: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_164, getitem_59);  add_164 = getitem_59 = None
    mul_122: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
    mul_123: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_122, arg287_1);  mul_122 = arg287_1 = None
    add_166: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_123, arg288_1);  mul_123 = arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_123: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_166, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_30: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_123);  slice_123 = None
    alias_31: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_378: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_166, 3)
    unsqueeze_379: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 4);  unsqueeze_378 = None
    permute_633: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_379, [0, 1, 3, 4, 2]);  unsqueeze_379 = None
    unsqueeze_380: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, 3);  arg105_1 = None
    unsqueeze_381: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 4);  unsqueeze_380 = None
    permute_634: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_381, [3, 4, 1, 2, 0]);  unsqueeze_381 = None
    permute_635: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_633, [0, 4, 1, 2, 3]);  permute_633 = None
    view_570: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_635, [1, 512, 1024]);  permute_635 = None
    permute_636: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_634, [4, 1, 2, 3, 0]);  permute_634 = None
    view_571: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_636, [1, 1024, 1024]);  permute_636 = None
    bmm_120: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_570, view_571);  view_570 = view_571 = None
    view_572: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_120, [512, 1, 1, 16, 64]);  bmm_120 = None
    permute_637: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_572, [0, 2, 3, 4, 1]);  view_572 = None
    view_573: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_637, [512, 1, 16, 64]);  permute_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_382: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_166, 3)
    unsqueeze_383: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 4);  unsqueeze_382 = None
    permute_638: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_383, [0, 1, 3, 4, 2]);  unsqueeze_383 = None
    unsqueeze_384: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, 3);  arg106_1 = None
    unsqueeze_385: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 4);  unsqueeze_384 = None
    permute_639: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_385, [3, 4, 1, 2, 0]);  unsqueeze_385 = None
    permute_640: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_638, [0, 4, 1, 2, 3]);  permute_638 = None
    view_574: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_640, [1, 512, 1024]);  permute_640 = None
    permute_641: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_639, [4, 1, 2, 3, 0]);  permute_639 = None
    view_575: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_641, [1, 1024, 1024]);  permute_641 = None
    bmm_121: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_574, view_575);  view_574 = view_575 = None
    view_576: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_121, [512, 1, 1, 16, 64]);  bmm_121 = None
    permute_642: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_576, [0, 2, 3, 4, 1]);  view_576 = None
    view_577: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_642, [512, 1, 16, 64]);  permute_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_386: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_166, 3)
    unsqueeze_387: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 4);  unsqueeze_386 = None
    permute_643: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_387, [0, 1, 3, 4, 2]);  unsqueeze_387 = None
    unsqueeze_388: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, 3);  arg107_1 = None
    unsqueeze_389: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 4);  unsqueeze_388 = None
    permute_644: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_389, [3, 4, 1, 2, 0]);  unsqueeze_389 = None
    permute_645: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_643, [0, 4, 1, 2, 3]);  permute_643 = None
    view_578: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_645, [1, 512, 1024]);  permute_645 = None
    permute_646: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_644, [4, 1, 2, 3, 0]);  permute_644 = None
    view_579: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_646, [1, 1024, 1024]);  permute_646 = None
    bmm_122: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_578, view_579);  view_578 = view_579 = None
    view_580: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_122, [512, 1, 1, 16, 64]);  bmm_122 = None
    permute_647: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_580, [0, 2, 3, 4, 1]);  view_580 = None
    view_581: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_647, [512, 1, 16, 64]);  permute_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_390: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_391: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 4);  unsqueeze_390 = None
    permute_648: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_391, [0, 1, 3, 4, 2]);  unsqueeze_391 = None
    unsqueeze_392: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, 3);  arg108_1 = None
    unsqueeze_393: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 4);  unsqueeze_392 = None
    permute_649: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_393, [3, 4, 1, 2, 0]);  unsqueeze_393 = None
    permute_650: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_648, [0, 4, 1, 2, 3]);  permute_648 = None
    view_582: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_650, [1, 1024, 1024]);  permute_650 = None
    permute_651: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_649, [4, 1, 2, 3, 0]);  permute_649 = None
    view_583: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_651, [1, 1024, 1024]);  permute_651 = None
    bmm_123: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_582, view_583);  view_582 = view_583 = None
    view_584: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_123, [1024, 1, 1, 16, 64]);  bmm_123 = None
    permute_652: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_584, [0, 2, 3, 4, 1]);  view_584 = None
    view_585: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_652, [1024, 1, 16, 64]);  permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_167: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_573, arg109_1);  arg109_1 = None
    unsqueeze_394: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_167, 4);  add_167 = None
    permute_653: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_394, [1, 2, 0, 4, 3]);  unsqueeze_394 = None
    unsqueeze_395: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_577, 4);  view_577 = None
    permute_654: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_395, [1, 2, 4, 0, 3]);  unsqueeze_395 = None
    permute_655: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_653, [1, 2, 4, 0, 3]);  permute_653 = None
    view_586: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_655, [16, 512, 64]);  permute_655 = None
    permute_656: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_654, [1, 4, 0, 3, 2]);  permute_654 = None
    view_587: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_656, [16, 64, 512]);  permute_656 = None
    bmm_124: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_586, view_587);  view_586 = view_587 = None
    view_588: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_124, [16, 512, 1, 1, 512]);  bmm_124 = None
    permute_657: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_588, [3, 0, 1, 4, 2]);  view_588 = None
    view_589: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_657, [1, 16, 512, 512]);  permute_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_168: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_573, arg110_1);  view_573 = arg110_1 = None
    unsqueeze_396: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_168, 4);  add_168 = None
    permute_658: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_396, [1, 2, 0, 4, 3]);  unsqueeze_396 = None
    unsqueeze_397: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_585, 4);  view_585 = None
    permute_659: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_397, [1, 2, 4, 0, 3]);  unsqueeze_397 = None
    permute_660: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_658, [1, 2, 4, 0, 3]);  permute_658 = None
    view_590: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_660, [16, 512, 64]);  permute_660 = None
    permute_661: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_659, [1, 4, 0, 3, 2]);  permute_659 = None
    view_591: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_661, [16, 64, 1024]);  permute_661 = None
    bmm_125: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_590, view_591);  view_590 = view_591 = None
    view_592: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_125, [16, 512, 1, 1, 1024]);  bmm_125 = None
    permute_662: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_592, [3, 0, 1, 4, 2]);  view_592 = None
    view_593: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_662, [1, 16, 512, 1024]);  permute_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_594: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_593, [1, 16, 1024, 512]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_124: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_594, 0, 0, 9223372036854775807);  view_594 = None
    slice_125: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_124, 1, 0, 9223372036854775807);  slice_124 = None
    slice_126: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_125, 2, 1, 9223372036854775807);  slice_125 = None
    slice_127: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_126, 3, 0, 9223372036854775807);  slice_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_595: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_127, [1, 16, 512, 1023]);  slice_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_17: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_128: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_595, 0, 0, 9223372036854775807);  view_595 = None
    slice_129: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_128, 1, 0, 9223372036854775807);  slice_128 = None
    slice_130: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_129, 2, 0, 9223372036854775807);  slice_129 = None
    index_15: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_130, [None, None, None, iota_17]);  slice_130 = iota_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_169: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_589, index_15);  view_589 = index_15 = None
    add_170: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_169, 0);  add_169 = None
    mul_124: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_170, 0.125);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_15: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_124, [3], True)
    sub_45: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_124, amax_15);  mul_124 = amax_15 = None
    exp_15: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_16: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [3], True)
    div_16: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_92: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_398: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_92, 4);  clone_92 = None
    permute_663: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_398, [2, 0, 1, 4, 3]);  unsqueeze_398 = None
    unsqueeze_399: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_581, 4);  view_581 = None
    permute_664: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_399, [4, 1, 2, 3, 0]);  unsqueeze_399 = None
    permute_665: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_663, [2, 0, 4, 1, 3]);  permute_663 = None
    view_596: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_665, [16, 512, 512]);  permute_665 = None
    permute_666: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_664, [2, 4, 1, 3, 0]);  permute_664 = None
    view_597: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_666, [16, 512, 64]);  permute_666 = None
    bmm_126: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_596, view_597);  view_596 = view_597 = None
    view_598: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_126, [16, 512, 1, 1, 64]);  bmm_126 = None
    permute_667: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_598, [1, 3, 0, 4, 2]);  view_598 = None
    view_599: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_667, [512, 1, 16, 64]);  permute_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_400: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_599, 4);  view_599 = None
    permute_668: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_400, [0, 1, 4, 3, 2]);  unsqueeze_400 = None
    unsqueeze_401: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, 3);  arg111_1 = None
    unsqueeze_402: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 4);  unsqueeze_401 = None
    permute_669: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_402, [3, 4, 0, 2, 1]);  unsqueeze_402 = None
    permute_670: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_668, [0, 3, 4, 1, 2]);  permute_668 = None
    clone_93: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_670, memory_format = torch.contiguous_format);  permute_670 = None
    view_600: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_93, [1, 512, 1024]);  clone_93 = None
    permute_671: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_669, [3, 4, 1, 2, 0]);  permute_669 = None
    clone_94: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_671, memory_format = torch.contiguous_format);  permute_671 = None
    view_601: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_94, [1, 1024, 1024]);  clone_94 = None
    bmm_127: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_600, view_601);  view_600 = view_601 = None
    view_602: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_127, [512, 1, 1, 1, 1024]);  bmm_127 = None
    permute_672: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_602, [0, 3, 4, 1, 2]);  view_602 = None
    view_603: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_672, [512, 1, 1024]);  permute_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_95: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_603);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_171: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_95, add_166);  clone_95 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
    getitem_60: "f32[512, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[512, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_172: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
    rsqrt_30: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_46: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_171, getitem_61);  add_171 = getitem_61 = None
    mul_125: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = rsqrt_30 = None
    mul_126: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_125, arg289_1);  mul_125 = arg289_1 = None
    add_173: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_126, arg290_1);  mul_126 = arg290_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_604: "f32[512, 1024]" = torch.ops.aten.view.default(add_173, [512, 1024])
    permute_673: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
    addmm_30: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg292_1, view_604, permute_673);  arg292_1 = view_604 = permute_673 = None
    view_605: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_30, [512, 1, 4096]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_127: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_605, 0.5)
    mul_128: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_605, 0.7071067811865476);  view_605 = None
    erf_15: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_174: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_129: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_127, add_174);  mul_127 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_96: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_129);  mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_606: "f32[512, 4096]" = torch.ops.aten.view.default(clone_96, [512, 4096]);  clone_96 = None
    permute_674: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
    addmm_31: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg294_1, view_606, permute_674);  arg294_1 = view_606 = permute_674 = None
    view_607: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_31, [512, 1, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_97: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_607);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_175: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_97, add_173);  clone_97 = add_173 = None
    var_mean_31 = torch.ops.aten.var_mean.correction(add_175, [2], correction = 0, keepdim = True)
    getitem_62: "f32[512, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[512, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_176: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_31: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_47: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_175, getitem_63);  add_175 = getitem_63 = None
    mul_130: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
    mul_131: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_130, arg295_1);  mul_130 = arg295_1 = None
    add_177: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_131, arg296_1);  mul_131 = arg296_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_131: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_177, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_32: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_131);  slice_131 = None
    alias_33: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_403: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_177, 3)
    unsqueeze_404: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 4);  unsqueeze_403 = None
    permute_675: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_404, [0, 1, 3, 4, 2]);  unsqueeze_404 = None
    unsqueeze_405: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, 3);  arg112_1 = None
    unsqueeze_406: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 4);  unsqueeze_405 = None
    permute_676: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_406, [3, 4, 1, 2, 0]);  unsqueeze_406 = None
    permute_677: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_675, [0, 4, 1, 2, 3]);  permute_675 = None
    view_608: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_677, [1, 512, 1024]);  permute_677 = None
    permute_678: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_676, [4, 1, 2, 3, 0]);  permute_676 = None
    view_609: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_678, [1, 1024, 1024]);  permute_678 = None
    bmm_128: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_608, view_609);  view_608 = view_609 = None
    view_610: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_128, [512, 1, 1, 16, 64]);  bmm_128 = None
    permute_679: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_610, [0, 2, 3, 4, 1]);  view_610 = None
    view_611: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_679, [512, 1, 16, 64]);  permute_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_407: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_177, 3)
    unsqueeze_408: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 4);  unsqueeze_407 = None
    permute_680: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_408, [0, 1, 3, 4, 2]);  unsqueeze_408 = None
    unsqueeze_409: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, 3);  arg113_1 = None
    unsqueeze_410: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 4);  unsqueeze_409 = None
    permute_681: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_410, [3, 4, 1, 2, 0]);  unsqueeze_410 = None
    permute_682: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_680, [0, 4, 1, 2, 3]);  permute_680 = None
    view_612: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_682, [1, 512, 1024]);  permute_682 = None
    permute_683: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_681, [4, 1, 2, 3, 0]);  permute_681 = None
    view_613: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_683, [1, 1024, 1024]);  permute_683 = None
    bmm_129: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_612, view_613);  view_612 = view_613 = None
    view_614: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_129, [512, 1, 1, 16, 64]);  bmm_129 = None
    permute_684: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_614, [0, 2, 3, 4, 1]);  view_614 = None
    view_615: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_684, [512, 1, 16, 64]);  permute_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_411: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_177, 3)
    unsqueeze_412: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 4);  unsqueeze_411 = None
    permute_685: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_412, [0, 1, 3, 4, 2]);  unsqueeze_412 = None
    unsqueeze_413: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, 3);  arg114_1 = None
    unsqueeze_414: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 4);  unsqueeze_413 = None
    permute_686: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_414, [3, 4, 1, 2, 0]);  unsqueeze_414 = None
    permute_687: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_685, [0, 4, 1, 2, 3]);  permute_685 = None
    view_616: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_687, [1, 512, 1024]);  permute_687 = None
    permute_688: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_686, [4, 1, 2, 3, 0]);  permute_686 = None
    view_617: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_688, [1, 1024, 1024]);  permute_688 = None
    bmm_130: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_616, view_617);  view_616 = view_617 = None
    view_618: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_130, [512, 1, 1, 16, 64]);  bmm_130 = None
    permute_689: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_618, [0, 2, 3, 4, 1]);  view_618 = None
    view_619: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_689, [512, 1, 16, 64]);  permute_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_415: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_416: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 4);  unsqueeze_415 = None
    permute_690: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_416, [0, 1, 3, 4, 2]);  unsqueeze_416 = None
    unsqueeze_417: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, 3);  arg115_1 = None
    unsqueeze_418: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 4);  unsqueeze_417 = None
    permute_691: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_418, [3, 4, 1, 2, 0]);  unsqueeze_418 = None
    permute_692: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_690, [0, 4, 1, 2, 3]);  permute_690 = None
    view_620: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_692, [1, 1024, 1024]);  permute_692 = None
    permute_693: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_691, [4, 1, 2, 3, 0]);  permute_691 = None
    view_621: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_693, [1, 1024, 1024]);  permute_693 = None
    bmm_131: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_620, view_621);  view_620 = view_621 = None
    view_622: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_131, [1024, 1, 1, 16, 64]);  bmm_131 = None
    permute_694: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_622, [0, 2, 3, 4, 1]);  view_622 = None
    view_623: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_694, [1024, 1, 16, 64]);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_178: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_611, arg116_1);  arg116_1 = None
    unsqueeze_419: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_178, 4);  add_178 = None
    permute_695: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_419, [1, 2, 0, 4, 3]);  unsqueeze_419 = None
    unsqueeze_420: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_615, 4);  view_615 = None
    permute_696: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_420, [1, 2, 4, 0, 3]);  unsqueeze_420 = None
    permute_697: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_695, [1, 2, 4, 0, 3]);  permute_695 = None
    view_624: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_697, [16, 512, 64]);  permute_697 = None
    permute_698: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_696, [1, 4, 0, 3, 2]);  permute_696 = None
    view_625: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_698, [16, 64, 512]);  permute_698 = None
    bmm_132: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_624, view_625);  view_624 = view_625 = None
    view_626: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_132, [16, 512, 1, 1, 512]);  bmm_132 = None
    permute_699: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_626, [3, 0, 1, 4, 2]);  view_626 = None
    view_627: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_699, [1, 16, 512, 512]);  permute_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_179: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_611, arg117_1);  view_611 = arg117_1 = None
    unsqueeze_421: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_179, 4);  add_179 = None
    permute_700: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_421, [1, 2, 0, 4, 3]);  unsqueeze_421 = None
    unsqueeze_422: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_623, 4);  view_623 = None
    permute_701: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_422, [1, 2, 4, 0, 3]);  unsqueeze_422 = None
    permute_702: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_700, [1, 2, 4, 0, 3]);  permute_700 = None
    view_628: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_702, [16, 512, 64]);  permute_702 = None
    permute_703: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_701, [1, 4, 0, 3, 2]);  permute_701 = None
    view_629: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_703, [16, 64, 1024]);  permute_703 = None
    bmm_133: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_628, view_629);  view_628 = view_629 = None
    view_630: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_133, [16, 512, 1, 1, 1024]);  bmm_133 = None
    permute_704: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_630, [3, 0, 1, 4, 2]);  view_630 = None
    view_631: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_704, [1, 16, 512, 1024]);  permute_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_632: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_631, [1, 16, 1024, 512]);  view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_132: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_632, 0, 0, 9223372036854775807);  view_632 = None
    slice_133: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_132, 1, 0, 9223372036854775807);  slice_132 = None
    slice_134: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_133, 2, 1, 9223372036854775807);  slice_133 = None
    slice_135: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_134, 3, 0, 9223372036854775807);  slice_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_633: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_135, [1, 16, 512, 1023]);  slice_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_18: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_136: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_633, 0, 0, 9223372036854775807);  view_633 = None
    slice_137: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_136, 1, 0, 9223372036854775807);  slice_136 = None
    slice_138: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_137, 2, 0, 9223372036854775807);  slice_137 = None
    index_16: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_138, [None, None, None, iota_18]);  slice_138 = iota_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_180: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_627, index_16);  view_627 = index_16 = None
    add_181: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_180, 0);  add_180 = None
    mul_132: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_181, 0.125);  add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_16: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_132, [3], True)
    sub_48: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_132, amax_16);  mul_132 = amax_16 = None
    exp_16: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_48);  sub_48 = None
    sum_17: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [3], True)
    div_17: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_98: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_423: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_98, 4);  clone_98 = None
    permute_705: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_423, [2, 0, 1, 4, 3]);  unsqueeze_423 = None
    unsqueeze_424: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_619, 4);  view_619 = None
    permute_706: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_424, [4, 1, 2, 3, 0]);  unsqueeze_424 = None
    permute_707: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_705, [2, 0, 4, 1, 3]);  permute_705 = None
    view_634: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_707, [16, 512, 512]);  permute_707 = None
    permute_708: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_706, [2, 4, 1, 3, 0]);  permute_706 = None
    view_635: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_708, [16, 512, 64]);  permute_708 = None
    bmm_134: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_634, view_635);  view_634 = view_635 = None
    view_636: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_134, [16, 512, 1, 1, 64]);  bmm_134 = None
    permute_709: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_636, [1, 3, 0, 4, 2]);  view_636 = None
    view_637: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_709, [512, 1, 16, 64]);  permute_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_425: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_637, 4);  view_637 = None
    permute_710: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_425, [0, 1, 4, 3, 2]);  unsqueeze_425 = None
    unsqueeze_426: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, 3);  arg118_1 = None
    unsqueeze_427: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 4);  unsqueeze_426 = None
    permute_711: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_427, [3, 4, 0, 2, 1]);  unsqueeze_427 = None
    permute_712: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_710, [0, 3, 4, 1, 2]);  permute_710 = None
    clone_99: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
    view_638: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_99, [1, 512, 1024]);  clone_99 = None
    permute_713: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_711, [3, 4, 1, 2, 0]);  permute_711 = None
    clone_100: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_713, memory_format = torch.contiguous_format);  permute_713 = None
    view_639: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_100, [1, 1024, 1024]);  clone_100 = None
    bmm_135: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_638, view_639);  view_638 = view_639 = None
    view_640: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_135, [512, 1, 1, 1, 1024]);  bmm_135 = None
    permute_714: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_640, [0, 3, 4, 1, 2]);  view_640 = None
    view_641: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_714, [512, 1, 1024]);  permute_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_101: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_641);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_182: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_101, add_177);  clone_101 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
    getitem_64: "f32[512, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[512, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_183: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
    rsqrt_32: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    sub_49: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_182, getitem_65);  add_182 = getitem_65 = None
    mul_133: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = rsqrt_32 = None
    mul_134: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_133, arg297_1);  mul_133 = arg297_1 = None
    add_184: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_134, arg298_1);  mul_134 = arg298_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_642: "f32[512, 1024]" = torch.ops.aten.view.default(add_184, [512, 1024])
    permute_715: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
    addmm_32: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg300_1, view_642, permute_715);  arg300_1 = view_642 = permute_715 = None
    view_643: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_32, [512, 1, 4096]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_135: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_643, 0.5)
    mul_136: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_643, 0.7071067811865476);  view_643 = None
    erf_16: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_136);  mul_136 = None
    add_185: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_137: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_135, add_185);  mul_135 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_102: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_137);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_644: "f32[512, 4096]" = torch.ops.aten.view.default(clone_102, [512, 4096]);  clone_102 = None
    permute_716: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
    addmm_33: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg302_1, view_644, permute_716);  arg302_1 = view_644 = permute_716 = None
    view_645: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_33, [512, 1, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_103: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_645);  view_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_186: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_103, add_184);  clone_103 = add_184 = None
    var_mean_33 = torch.ops.aten.var_mean.correction(add_186, [2], correction = 0, keepdim = True)
    getitem_66: "f32[512, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[512, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_187: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
    rsqrt_33: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_50: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_186, getitem_67);  add_186 = getitem_67 = None
    mul_138: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
    mul_139: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_138, arg303_1);  mul_138 = arg303_1 = None
    add_188: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_139, arg304_1);  mul_139 = arg304_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_139: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_188, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_34: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_139);  slice_139 = None
    alias_35: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_428: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_188, 3)
    unsqueeze_429: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 4);  unsqueeze_428 = None
    permute_717: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_429, [0, 1, 3, 4, 2]);  unsqueeze_429 = None
    unsqueeze_430: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, 3);  arg119_1 = None
    unsqueeze_431: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 4);  unsqueeze_430 = None
    permute_718: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_431, [3, 4, 1, 2, 0]);  unsqueeze_431 = None
    permute_719: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_717, [0, 4, 1, 2, 3]);  permute_717 = None
    view_646: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_719, [1, 512, 1024]);  permute_719 = None
    permute_720: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_718, [4, 1, 2, 3, 0]);  permute_718 = None
    view_647: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_720, [1, 1024, 1024]);  permute_720 = None
    bmm_136: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_646, view_647);  view_646 = view_647 = None
    view_648: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_136, [512, 1, 1, 16, 64]);  bmm_136 = None
    permute_721: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_648, [0, 2, 3, 4, 1]);  view_648 = None
    view_649: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_721, [512, 1, 16, 64]);  permute_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_432: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_188, 3)
    unsqueeze_433: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 4);  unsqueeze_432 = None
    permute_722: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_433, [0, 1, 3, 4, 2]);  unsqueeze_433 = None
    unsqueeze_434: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, 3);  arg120_1 = None
    unsqueeze_435: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 4);  unsqueeze_434 = None
    permute_723: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_435, [3, 4, 1, 2, 0]);  unsqueeze_435 = None
    permute_724: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_722, [0, 4, 1, 2, 3]);  permute_722 = None
    view_650: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_724, [1, 512, 1024]);  permute_724 = None
    permute_725: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_723, [4, 1, 2, 3, 0]);  permute_723 = None
    view_651: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_725, [1, 1024, 1024]);  permute_725 = None
    bmm_137: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_650, view_651);  view_650 = view_651 = None
    view_652: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_137, [512, 1, 1, 16, 64]);  bmm_137 = None
    permute_726: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_652, [0, 2, 3, 4, 1]);  view_652 = None
    view_653: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_726, [512, 1, 16, 64]);  permute_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_436: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_188, 3)
    unsqueeze_437: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 4);  unsqueeze_436 = None
    permute_727: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_437, [0, 1, 3, 4, 2]);  unsqueeze_437 = None
    unsqueeze_438: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, 3);  arg121_1 = None
    unsqueeze_439: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 4);  unsqueeze_438 = None
    permute_728: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_439, [3, 4, 1, 2, 0]);  unsqueeze_439 = None
    permute_729: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_727, [0, 4, 1, 2, 3]);  permute_727 = None
    view_654: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_729, [1, 512, 1024]);  permute_729 = None
    permute_730: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_728, [4, 1, 2, 3, 0]);  permute_728 = None
    view_655: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_730, [1, 1024, 1024]);  permute_730 = None
    bmm_138: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_654, view_655);  view_654 = view_655 = None
    view_656: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_138, [512, 1, 1, 16, 64]);  bmm_138 = None
    permute_731: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_656, [0, 2, 3, 4, 1]);  view_656 = None
    view_657: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_731, [512, 1, 16, 64]);  permute_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_440: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_441: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 4);  unsqueeze_440 = None
    permute_732: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_441, [0, 1, 3, 4, 2]);  unsqueeze_441 = None
    unsqueeze_442: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, 3);  arg122_1 = None
    unsqueeze_443: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 4);  unsqueeze_442 = None
    permute_733: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_443, [3, 4, 1, 2, 0]);  unsqueeze_443 = None
    permute_734: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_732, [0, 4, 1, 2, 3]);  permute_732 = None
    view_658: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_734, [1, 1024, 1024]);  permute_734 = None
    permute_735: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_733, [4, 1, 2, 3, 0]);  permute_733 = None
    view_659: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_735, [1, 1024, 1024]);  permute_735 = None
    bmm_139: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_658, view_659);  view_658 = view_659 = None
    view_660: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_139, [1024, 1, 1, 16, 64]);  bmm_139 = None
    permute_736: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_660, [0, 2, 3, 4, 1]);  view_660 = None
    view_661: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_736, [1024, 1, 16, 64]);  permute_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_189: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_649, arg123_1);  arg123_1 = None
    unsqueeze_444: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_189, 4);  add_189 = None
    permute_737: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_444, [1, 2, 0, 4, 3]);  unsqueeze_444 = None
    unsqueeze_445: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_653, 4);  view_653 = None
    permute_738: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_445, [1, 2, 4, 0, 3]);  unsqueeze_445 = None
    permute_739: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_737, [1, 2, 4, 0, 3]);  permute_737 = None
    view_662: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_739, [16, 512, 64]);  permute_739 = None
    permute_740: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_738, [1, 4, 0, 3, 2]);  permute_738 = None
    view_663: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_740, [16, 64, 512]);  permute_740 = None
    bmm_140: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_662, view_663);  view_662 = view_663 = None
    view_664: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_140, [16, 512, 1, 1, 512]);  bmm_140 = None
    permute_741: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_664, [3, 0, 1, 4, 2]);  view_664 = None
    view_665: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_741, [1, 16, 512, 512]);  permute_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_190: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_649, arg124_1);  view_649 = arg124_1 = None
    unsqueeze_446: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_190, 4);  add_190 = None
    permute_742: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_446, [1, 2, 0, 4, 3]);  unsqueeze_446 = None
    unsqueeze_447: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_661, 4);  view_661 = None
    permute_743: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_447, [1, 2, 4, 0, 3]);  unsqueeze_447 = None
    permute_744: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_742, [1, 2, 4, 0, 3]);  permute_742 = None
    view_666: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_744, [16, 512, 64]);  permute_744 = None
    permute_745: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_743, [1, 4, 0, 3, 2]);  permute_743 = None
    view_667: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_745, [16, 64, 1024]);  permute_745 = None
    bmm_141: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_666, view_667);  view_666 = view_667 = None
    view_668: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_141, [16, 512, 1, 1, 1024]);  bmm_141 = None
    permute_746: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_668, [3, 0, 1, 4, 2]);  view_668 = None
    view_669: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_746, [1, 16, 512, 1024]);  permute_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_670: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_669, [1, 16, 1024, 512]);  view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_140: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_670, 0, 0, 9223372036854775807);  view_670 = None
    slice_141: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_140, 1, 0, 9223372036854775807);  slice_140 = None
    slice_142: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_141, 2, 1, 9223372036854775807);  slice_141 = None
    slice_143: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_142, 3, 0, 9223372036854775807);  slice_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_671: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_143, [1, 16, 512, 1023]);  slice_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_19: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_144: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_671, 0, 0, 9223372036854775807);  view_671 = None
    slice_145: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_144, 1, 0, 9223372036854775807);  slice_144 = None
    slice_146: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_145, 2, 0, 9223372036854775807);  slice_145 = None
    index_17: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_146, [None, None, None, iota_19]);  slice_146 = iota_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_191: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_665, index_17);  view_665 = index_17 = None
    add_192: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_191, 0);  add_191 = None
    mul_140: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_192, 0.125);  add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_17: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_140, [3], True)
    sub_51: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_140, amax_17);  mul_140 = amax_17 = None
    exp_17: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_18: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [3], True)
    div_18: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_104: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_448: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_104, 4);  clone_104 = None
    permute_747: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_448, [2, 0, 1, 4, 3]);  unsqueeze_448 = None
    unsqueeze_449: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_657, 4);  view_657 = None
    permute_748: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_449, [4, 1, 2, 3, 0]);  unsqueeze_449 = None
    permute_749: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_747, [2, 0, 4, 1, 3]);  permute_747 = None
    view_672: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_749, [16, 512, 512]);  permute_749 = None
    permute_750: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_748, [2, 4, 1, 3, 0]);  permute_748 = None
    view_673: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_750, [16, 512, 64]);  permute_750 = None
    bmm_142: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_672, view_673);  view_672 = view_673 = None
    view_674: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_142, [16, 512, 1, 1, 64]);  bmm_142 = None
    permute_751: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_674, [1, 3, 0, 4, 2]);  view_674 = None
    view_675: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_751, [512, 1, 16, 64]);  permute_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_450: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_675, 4);  view_675 = None
    permute_752: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_450, [0, 1, 4, 3, 2]);  unsqueeze_450 = None
    unsqueeze_451: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, 3);  arg125_1 = None
    unsqueeze_452: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 4);  unsqueeze_451 = None
    permute_753: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_452, [3, 4, 0, 2, 1]);  unsqueeze_452 = None
    permute_754: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_752, [0, 3, 4, 1, 2]);  permute_752 = None
    clone_105: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
    view_676: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_105, [1, 512, 1024]);  clone_105 = None
    permute_755: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_753, [3, 4, 1, 2, 0]);  permute_753 = None
    clone_106: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_755, memory_format = torch.contiguous_format);  permute_755 = None
    view_677: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_106, [1, 1024, 1024]);  clone_106 = None
    bmm_143: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_676, view_677);  view_676 = view_677 = None
    view_678: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_143, [512, 1, 1, 1, 1024]);  bmm_143 = None
    permute_756: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_678, [0, 3, 4, 1, 2]);  view_678 = None
    view_679: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_756, [512, 1, 1024]);  permute_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_107: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_679);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_193: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_107, add_188);  clone_107 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
    getitem_68: "f32[512, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[512, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_194: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_34: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_52: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_193, getitem_69);  add_193 = getitem_69 = None
    mul_141: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = rsqrt_34 = None
    mul_142: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_141, arg305_1);  mul_141 = arg305_1 = None
    add_195: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_142, arg306_1);  mul_142 = arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_680: "f32[512, 1024]" = torch.ops.aten.view.default(add_195, [512, 1024])
    permute_757: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
    addmm_34: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg308_1, view_680, permute_757);  arg308_1 = view_680 = permute_757 = None
    view_681: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_34, [512, 1, 4096]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_143: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_681, 0.5)
    mul_144: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_681, 0.7071067811865476);  view_681 = None
    erf_17: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
    add_196: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_145: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_143, add_196);  mul_143 = add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_108: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_145);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_682: "f32[512, 4096]" = torch.ops.aten.view.default(clone_108, [512, 4096]);  clone_108 = None
    permute_758: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
    addmm_35: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg310_1, view_682, permute_758);  arg310_1 = view_682 = permute_758 = None
    view_683: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_35, [512, 1, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_109: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_683);  view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_197: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_109, add_195);  clone_109 = add_195 = None
    var_mean_35 = torch.ops.aten.var_mean.correction(add_197, [2], correction = 0, keepdim = True)
    getitem_70: "f32[512, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[512, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_198: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
    rsqrt_35: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_53: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_197, getitem_71);  add_197 = getitem_71 = None
    mul_146: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
    mul_147: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_146, arg311_1);  mul_146 = arg311_1 = None
    add_199: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_147, arg312_1);  mul_147 = arg312_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_147: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_199, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_36: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_147);  slice_147 = None
    alias_37: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_453: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_199, 3)
    unsqueeze_454: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 4);  unsqueeze_453 = None
    permute_759: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_454, [0, 1, 3, 4, 2]);  unsqueeze_454 = None
    unsqueeze_455: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, 3);  arg126_1 = None
    unsqueeze_456: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 4);  unsqueeze_455 = None
    permute_760: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_456, [3, 4, 1, 2, 0]);  unsqueeze_456 = None
    permute_761: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_759, [0, 4, 1, 2, 3]);  permute_759 = None
    view_684: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_761, [1, 512, 1024]);  permute_761 = None
    permute_762: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_760, [4, 1, 2, 3, 0]);  permute_760 = None
    view_685: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_762, [1, 1024, 1024]);  permute_762 = None
    bmm_144: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_684, view_685);  view_684 = view_685 = None
    view_686: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_144, [512, 1, 1, 16, 64]);  bmm_144 = None
    permute_763: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_686, [0, 2, 3, 4, 1]);  view_686 = None
    view_687: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_763, [512, 1, 16, 64]);  permute_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_457: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_199, 3)
    unsqueeze_458: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 4);  unsqueeze_457 = None
    permute_764: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_458, [0, 1, 3, 4, 2]);  unsqueeze_458 = None
    unsqueeze_459: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, 3);  arg127_1 = None
    unsqueeze_460: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 4);  unsqueeze_459 = None
    permute_765: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_460, [3, 4, 1, 2, 0]);  unsqueeze_460 = None
    permute_766: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_764, [0, 4, 1, 2, 3]);  permute_764 = None
    view_688: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_766, [1, 512, 1024]);  permute_766 = None
    permute_767: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_765, [4, 1, 2, 3, 0]);  permute_765 = None
    view_689: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_767, [1, 1024, 1024]);  permute_767 = None
    bmm_145: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_688, view_689);  view_688 = view_689 = None
    view_690: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_145, [512, 1, 1, 16, 64]);  bmm_145 = None
    permute_768: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_690, [0, 2, 3, 4, 1]);  view_690 = None
    view_691: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_768, [512, 1, 16, 64]);  permute_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_461: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_199, 3)
    unsqueeze_462: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 4);  unsqueeze_461 = None
    permute_769: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_462, [0, 1, 3, 4, 2]);  unsqueeze_462 = None
    unsqueeze_463: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, 3);  arg128_1 = None
    unsqueeze_464: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 4);  unsqueeze_463 = None
    permute_770: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_464, [3, 4, 1, 2, 0]);  unsqueeze_464 = None
    permute_771: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_769, [0, 4, 1, 2, 3]);  permute_769 = None
    view_692: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_771, [1, 512, 1024]);  permute_771 = None
    permute_772: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_770, [4, 1, 2, 3, 0]);  permute_770 = None
    view_693: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_772, [1, 1024, 1024]);  permute_772 = None
    bmm_146: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_692, view_693);  view_692 = view_693 = None
    view_694: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_146, [512, 1, 1, 16, 64]);  bmm_146 = None
    permute_773: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_694, [0, 2, 3, 4, 1]);  view_694 = None
    view_695: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_773, [512, 1, 16, 64]);  permute_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_465: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_466: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 4);  unsqueeze_465 = None
    permute_774: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_466, [0, 1, 3, 4, 2]);  unsqueeze_466 = None
    unsqueeze_467: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, 3);  arg129_1 = None
    unsqueeze_468: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 4);  unsqueeze_467 = None
    permute_775: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_468, [3, 4, 1, 2, 0]);  unsqueeze_468 = None
    permute_776: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_774, [0, 4, 1, 2, 3]);  permute_774 = None
    view_696: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_776, [1, 1024, 1024]);  permute_776 = None
    permute_777: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_775, [4, 1, 2, 3, 0]);  permute_775 = None
    view_697: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_777, [1, 1024, 1024]);  permute_777 = None
    bmm_147: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_696, view_697);  view_696 = view_697 = None
    view_698: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_147, [1024, 1, 1, 16, 64]);  bmm_147 = None
    permute_778: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_698, [0, 2, 3, 4, 1]);  view_698 = None
    view_699: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_778, [1024, 1, 16, 64]);  permute_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_200: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_687, arg130_1);  arg130_1 = None
    unsqueeze_469: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_200, 4);  add_200 = None
    permute_779: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_469, [1, 2, 0, 4, 3]);  unsqueeze_469 = None
    unsqueeze_470: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_691, 4);  view_691 = None
    permute_780: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_470, [1, 2, 4, 0, 3]);  unsqueeze_470 = None
    permute_781: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_779, [1, 2, 4, 0, 3]);  permute_779 = None
    view_700: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_781, [16, 512, 64]);  permute_781 = None
    permute_782: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_780, [1, 4, 0, 3, 2]);  permute_780 = None
    view_701: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_782, [16, 64, 512]);  permute_782 = None
    bmm_148: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_700, view_701);  view_700 = view_701 = None
    view_702: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_148, [16, 512, 1, 1, 512]);  bmm_148 = None
    permute_783: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_702, [3, 0, 1, 4, 2]);  view_702 = None
    view_703: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_783, [1, 16, 512, 512]);  permute_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_201: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_687, arg131_1);  view_687 = arg131_1 = None
    unsqueeze_471: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_201, 4);  add_201 = None
    permute_784: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_471, [1, 2, 0, 4, 3]);  unsqueeze_471 = None
    unsqueeze_472: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_699, 4);  view_699 = None
    permute_785: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_472, [1, 2, 4, 0, 3]);  unsqueeze_472 = None
    permute_786: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_784, [1, 2, 4, 0, 3]);  permute_784 = None
    view_704: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_786, [16, 512, 64]);  permute_786 = None
    permute_787: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_785, [1, 4, 0, 3, 2]);  permute_785 = None
    view_705: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_787, [16, 64, 1024]);  permute_787 = None
    bmm_149: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_704, view_705);  view_704 = view_705 = None
    view_706: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_149, [16, 512, 1, 1, 1024]);  bmm_149 = None
    permute_788: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_706, [3, 0, 1, 4, 2]);  view_706 = None
    view_707: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_788, [1, 16, 512, 1024]);  permute_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_708: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_707, [1, 16, 1024, 512]);  view_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_148: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_708, 0, 0, 9223372036854775807);  view_708 = None
    slice_149: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_148, 1, 0, 9223372036854775807);  slice_148 = None
    slice_150: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_149, 2, 1, 9223372036854775807);  slice_149 = None
    slice_151: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_150, 3, 0, 9223372036854775807);  slice_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_709: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_151, [1, 16, 512, 1023]);  slice_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_20: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_152: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_709, 0, 0, 9223372036854775807);  view_709 = None
    slice_153: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_152, 1, 0, 9223372036854775807);  slice_152 = None
    slice_154: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_153, 2, 0, 9223372036854775807);  slice_153 = None
    index_18: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_154, [None, None, None, iota_20]);  slice_154 = iota_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_202: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_703, index_18);  view_703 = index_18 = None
    add_203: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_202, 0);  add_202 = None
    mul_148: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_203, 0.125);  add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_18: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_148, [3], True)
    sub_54: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_148, amax_18);  mul_148 = amax_18 = None
    exp_18: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
    sum_19: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [3], True)
    div_19: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_110: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_473: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_110, 4);  clone_110 = None
    permute_789: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_473, [2, 0, 1, 4, 3]);  unsqueeze_473 = None
    unsqueeze_474: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_695, 4);  view_695 = None
    permute_790: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_474, [4, 1, 2, 3, 0]);  unsqueeze_474 = None
    permute_791: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_789, [2, 0, 4, 1, 3]);  permute_789 = None
    view_710: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_791, [16, 512, 512]);  permute_791 = None
    permute_792: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_790, [2, 4, 1, 3, 0]);  permute_790 = None
    view_711: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_792, [16, 512, 64]);  permute_792 = None
    bmm_150: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_710, view_711);  view_710 = view_711 = None
    view_712: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_150, [16, 512, 1, 1, 64]);  bmm_150 = None
    permute_793: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_712, [1, 3, 0, 4, 2]);  view_712 = None
    view_713: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_793, [512, 1, 16, 64]);  permute_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_475: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_713, 4);  view_713 = None
    permute_794: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_475, [0, 1, 4, 3, 2]);  unsqueeze_475 = None
    unsqueeze_476: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg132_1, 3);  arg132_1 = None
    unsqueeze_477: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 4);  unsqueeze_476 = None
    permute_795: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_477, [3, 4, 0, 2, 1]);  unsqueeze_477 = None
    permute_796: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_794, [0, 3, 4, 1, 2]);  permute_794 = None
    clone_111: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_796, memory_format = torch.contiguous_format);  permute_796 = None
    view_714: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_111, [1, 512, 1024]);  clone_111 = None
    permute_797: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_795, [3, 4, 1, 2, 0]);  permute_795 = None
    clone_112: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format);  permute_797 = None
    view_715: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_112, [1, 1024, 1024]);  clone_112 = None
    bmm_151: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_714, view_715);  view_714 = view_715 = None
    view_716: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_151, [512, 1, 1, 1, 1024]);  bmm_151 = None
    permute_798: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_716, [0, 3, 4, 1, 2]);  view_716 = None
    view_717: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_798, [512, 1, 1024]);  permute_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_113: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_717);  view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_204: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_113, add_199);  clone_113 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_204, [2], correction = 0, keepdim = True)
    getitem_72: "f32[512, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[512, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_205: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
    rsqrt_36: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    sub_55: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_204, getitem_73);  add_204 = getitem_73 = None
    mul_149: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = rsqrt_36 = None
    mul_150: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_149, arg313_1);  mul_149 = arg313_1 = None
    add_206: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_150, arg314_1);  mul_150 = arg314_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_718: "f32[512, 1024]" = torch.ops.aten.view.default(add_206, [512, 1024])
    permute_799: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
    addmm_36: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg316_1, view_718, permute_799);  arg316_1 = view_718 = permute_799 = None
    view_719: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_36, [512, 1, 4096]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_151: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_719, 0.5)
    mul_152: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_719, 0.7071067811865476);  view_719 = None
    erf_18: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
    add_207: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_153: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_151, add_207);  mul_151 = add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_114: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_153);  mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_720: "f32[512, 4096]" = torch.ops.aten.view.default(clone_114, [512, 4096]);  clone_114 = None
    permute_800: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
    addmm_37: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg318_1, view_720, permute_800);  arg318_1 = view_720 = permute_800 = None
    view_721: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_37, [512, 1, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_115: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_721);  view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_208: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_115, add_206);  clone_115 = add_206 = None
    var_mean_37 = torch.ops.aten.var_mean.correction(add_208, [2], correction = 0, keepdim = True)
    getitem_74: "f32[512, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[512, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_209: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
    rsqrt_37: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    sub_56: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_208, getitem_75);  add_208 = getitem_75 = None
    mul_154: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
    mul_155: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_154, arg319_1);  mul_154 = arg319_1 = None
    add_210: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_155, arg320_1);  mul_155 = arg320_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_155: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_210, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_38: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_155);  slice_155 = None
    alias_39: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_478: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_210, 3)
    unsqueeze_479: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 4);  unsqueeze_478 = None
    permute_801: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_479, [0, 1, 3, 4, 2]);  unsqueeze_479 = None
    unsqueeze_480: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, 3);  arg133_1 = None
    unsqueeze_481: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 4);  unsqueeze_480 = None
    permute_802: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_481, [3, 4, 1, 2, 0]);  unsqueeze_481 = None
    permute_803: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_801, [0, 4, 1, 2, 3]);  permute_801 = None
    view_722: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_803, [1, 512, 1024]);  permute_803 = None
    permute_804: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_802, [4, 1, 2, 3, 0]);  permute_802 = None
    view_723: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_804, [1, 1024, 1024]);  permute_804 = None
    bmm_152: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_722, view_723);  view_722 = view_723 = None
    view_724: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_152, [512, 1, 1, 16, 64]);  bmm_152 = None
    permute_805: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_724, [0, 2, 3, 4, 1]);  view_724 = None
    view_725: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_805, [512, 1, 16, 64]);  permute_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_482: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_210, 3)
    unsqueeze_483: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 4);  unsqueeze_482 = None
    permute_806: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_483, [0, 1, 3, 4, 2]);  unsqueeze_483 = None
    unsqueeze_484: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, 3);  arg134_1 = None
    unsqueeze_485: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 4);  unsqueeze_484 = None
    permute_807: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_485, [3, 4, 1, 2, 0]);  unsqueeze_485 = None
    permute_808: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_806, [0, 4, 1, 2, 3]);  permute_806 = None
    view_726: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_808, [1, 512, 1024]);  permute_808 = None
    permute_809: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_807, [4, 1, 2, 3, 0]);  permute_807 = None
    view_727: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_809, [1, 1024, 1024]);  permute_809 = None
    bmm_153: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_726, view_727);  view_726 = view_727 = None
    view_728: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_153, [512, 1, 1, 16, 64]);  bmm_153 = None
    permute_810: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_728, [0, 2, 3, 4, 1]);  view_728 = None
    view_729: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_810, [512, 1, 16, 64]);  permute_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_486: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_210, 3)
    unsqueeze_487: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 4);  unsqueeze_486 = None
    permute_811: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_487, [0, 1, 3, 4, 2]);  unsqueeze_487 = None
    unsqueeze_488: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, 3);  arg135_1 = None
    unsqueeze_489: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 4);  unsqueeze_488 = None
    permute_812: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_489, [3, 4, 1, 2, 0]);  unsqueeze_489 = None
    permute_813: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_811, [0, 4, 1, 2, 3]);  permute_811 = None
    view_730: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_813, [1, 512, 1024]);  permute_813 = None
    permute_814: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_812, [4, 1, 2, 3, 0]);  permute_812 = None
    view_731: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_814, [1, 1024, 1024]);  permute_814 = None
    bmm_154: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_730, view_731);  view_730 = view_731 = None
    view_732: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_154, [512, 1, 1, 16, 64]);  bmm_154 = None
    permute_815: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_732, [0, 2, 3, 4, 1]);  view_732 = None
    view_733: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_815, [512, 1, 16, 64]);  permute_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_490: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_491: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 4);  unsqueeze_490 = None
    permute_816: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_491, [0, 1, 3, 4, 2]);  unsqueeze_491 = None
    unsqueeze_492: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, 3);  arg136_1 = None
    unsqueeze_493: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 4);  unsqueeze_492 = None
    permute_817: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_493, [3, 4, 1, 2, 0]);  unsqueeze_493 = None
    permute_818: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_816, [0, 4, 1, 2, 3]);  permute_816 = None
    view_734: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_818, [1, 1024, 1024]);  permute_818 = None
    permute_819: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_817, [4, 1, 2, 3, 0]);  permute_817 = None
    view_735: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_819, [1, 1024, 1024]);  permute_819 = None
    bmm_155: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_734, view_735);  view_734 = view_735 = None
    view_736: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_155, [1024, 1, 1, 16, 64]);  bmm_155 = None
    permute_820: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_736, [0, 2, 3, 4, 1]);  view_736 = None
    view_737: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_820, [1024, 1, 16, 64]);  permute_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_211: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_725, arg137_1);  arg137_1 = None
    unsqueeze_494: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_211, 4);  add_211 = None
    permute_821: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_494, [1, 2, 0, 4, 3]);  unsqueeze_494 = None
    unsqueeze_495: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_729, 4);  view_729 = None
    permute_822: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_495, [1, 2, 4, 0, 3]);  unsqueeze_495 = None
    permute_823: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_821, [1, 2, 4, 0, 3]);  permute_821 = None
    view_738: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_823, [16, 512, 64]);  permute_823 = None
    permute_824: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_822, [1, 4, 0, 3, 2]);  permute_822 = None
    view_739: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_824, [16, 64, 512]);  permute_824 = None
    bmm_156: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_738, view_739);  view_738 = view_739 = None
    view_740: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_156, [16, 512, 1, 1, 512]);  bmm_156 = None
    permute_825: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_740, [3, 0, 1, 4, 2]);  view_740 = None
    view_741: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_825, [1, 16, 512, 512]);  permute_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_212: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_725, arg138_1);  view_725 = arg138_1 = None
    unsqueeze_496: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_212, 4);  add_212 = None
    permute_826: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_496, [1, 2, 0, 4, 3]);  unsqueeze_496 = None
    unsqueeze_497: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_737, 4);  view_737 = None
    permute_827: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_497, [1, 2, 4, 0, 3]);  unsqueeze_497 = None
    permute_828: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_826, [1, 2, 4, 0, 3]);  permute_826 = None
    view_742: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_828, [16, 512, 64]);  permute_828 = None
    permute_829: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_827, [1, 4, 0, 3, 2]);  permute_827 = None
    view_743: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_829, [16, 64, 1024]);  permute_829 = None
    bmm_157: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_742, view_743);  view_742 = view_743 = None
    view_744: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_157, [16, 512, 1, 1, 1024]);  bmm_157 = None
    permute_830: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_744, [3, 0, 1, 4, 2]);  view_744 = None
    view_745: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_830, [1, 16, 512, 1024]);  permute_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_746: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_745, [1, 16, 1024, 512]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_156: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_746, 0, 0, 9223372036854775807);  view_746 = None
    slice_157: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_156, 1, 0, 9223372036854775807);  slice_156 = None
    slice_158: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_157, 2, 1, 9223372036854775807);  slice_157 = None
    slice_159: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_158, 3, 0, 9223372036854775807);  slice_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_747: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_159, [1, 16, 512, 1023]);  slice_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_21: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_160: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_747, 0, 0, 9223372036854775807);  view_747 = None
    slice_161: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_160, 1, 0, 9223372036854775807);  slice_160 = None
    slice_162: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_161, 2, 0, 9223372036854775807);  slice_161 = None
    index_19: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_162, [None, None, None, iota_21]);  slice_162 = iota_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_213: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_741, index_19);  view_741 = index_19 = None
    add_214: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_213, 0);  add_213 = None
    mul_156: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_214, 0.125);  add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_19: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_156, [3], True)
    sub_57: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_156, amax_19);  mul_156 = amax_19 = None
    exp_19: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_20: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [3], True)
    div_20: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_116: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_498: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_116, 4);  clone_116 = None
    permute_831: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_498, [2, 0, 1, 4, 3]);  unsqueeze_498 = None
    unsqueeze_499: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_733, 4);  view_733 = None
    permute_832: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_499, [4, 1, 2, 3, 0]);  unsqueeze_499 = None
    permute_833: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_831, [2, 0, 4, 1, 3]);  permute_831 = None
    view_748: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_833, [16, 512, 512]);  permute_833 = None
    permute_834: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_832, [2, 4, 1, 3, 0]);  permute_832 = None
    view_749: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_834, [16, 512, 64]);  permute_834 = None
    bmm_158: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_748, view_749);  view_748 = view_749 = None
    view_750: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_158, [16, 512, 1, 1, 64]);  bmm_158 = None
    permute_835: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_750, [1, 3, 0, 4, 2]);  view_750 = None
    view_751: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_835, [512, 1, 16, 64]);  permute_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_500: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_751, 4);  view_751 = None
    permute_836: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_500, [0, 1, 4, 3, 2]);  unsqueeze_500 = None
    unsqueeze_501: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, 3);  arg139_1 = None
    unsqueeze_502: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 4);  unsqueeze_501 = None
    permute_837: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_502, [3, 4, 0, 2, 1]);  unsqueeze_502 = None
    permute_838: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_836, [0, 3, 4, 1, 2]);  permute_836 = None
    clone_117: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
    view_752: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_117, [1, 512, 1024]);  clone_117 = None
    permute_839: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_837, [3, 4, 1, 2, 0]);  permute_837 = None
    clone_118: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_839, memory_format = torch.contiguous_format);  permute_839 = None
    view_753: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_118, [1, 1024, 1024]);  clone_118 = None
    bmm_159: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_752, view_753);  view_752 = view_753 = None
    view_754: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_159, [512, 1, 1, 1, 1024]);  bmm_159 = None
    permute_840: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_754, [0, 3, 4, 1, 2]);  view_754 = None
    view_755: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_840, [512, 1, 1024]);  permute_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_119: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_755);  view_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_215: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_119, add_210);  clone_119 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_215, [2], correction = 0, keepdim = True)
    getitem_76: "f32[512, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[512, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_216: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
    rsqrt_38: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    sub_58: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_215, getitem_77);  add_215 = getitem_77 = None
    mul_157: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = rsqrt_38 = None
    mul_158: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_157, arg321_1);  mul_157 = arg321_1 = None
    add_217: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_158, arg322_1);  mul_158 = arg322_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_756: "f32[512, 1024]" = torch.ops.aten.view.default(add_217, [512, 1024])
    permute_841: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
    addmm_38: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg324_1, view_756, permute_841);  arg324_1 = view_756 = permute_841 = None
    view_757: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_38, [512, 1, 4096]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_159: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_757, 0.5)
    mul_160: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_757, 0.7071067811865476);  view_757 = None
    erf_19: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
    add_218: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_161: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_159, add_218);  mul_159 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_120: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_161);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_758: "f32[512, 4096]" = torch.ops.aten.view.default(clone_120, [512, 4096]);  clone_120 = None
    permute_842: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
    addmm_39: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg326_1, view_758, permute_842);  arg326_1 = view_758 = permute_842 = None
    view_759: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_39, [512, 1, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_121: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_759);  view_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_219: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_121, add_217);  clone_121 = add_217 = None
    var_mean_39 = torch.ops.aten.var_mean.correction(add_219, [2], correction = 0, keepdim = True)
    getitem_78: "f32[512, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[512, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_220: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
    rsqrt_39: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    sub_59: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_219, getitem_79);  add_219 = getitem_79 = None
    mul_162: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
    mul_163: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_162, arg327_1);  mul_162 = arg327_1 = None
    add_221: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_163, arg328_1);  mul_163 = arg328_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_163: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_221, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_40: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_163);  slice_163 = None
    alias_41: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_503: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_221, 3)
    unsqueeze_504: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 4);  unsqueeze_503 = None
    permute_843: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_504, [0, 1, 3, 4, 2]);  unsqueeze_504 = None
    unsqueeze_505: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, 3);  arg140_1 = None
    unsqueeze_506: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 4);  unsqueeze_505 = None
    permute_844: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_506, [3, 4, 1, 2, 0]);  unsqueeze_506 = None
    permute_845: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_843, [0, 4, 1, 2, 3]);  permute_843 = None
    view_760: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_845, [1, 512, 1024]);  permute_845 = None
    permute_846: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_844, [4, 1, 2, 3, 0]);  permute_844 = None
    view_761: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_846, [1, 1024, 1024]);  permute_846 = None
    bmm_160: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_760, view_761);  view_760 = view_761 = None
    view_762: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_160, [512, 1, 1, 16, 64]);  bmm_160 = None
    permute_847: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_762, [0, 2, 3, 4, 1]);  view_762 = None
    view_763: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_847, [512, 1, 16, 64]);  permute_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_507: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_221, 3)
    unsqueeze_508: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 4);  unsqueeze_507 = None
    permute_848: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_508, [0, 1, 3, 4, 2]);  unsqueeze_508 = None
    unsqueeze_509: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, 3);  arg141_1 = None
    unsqueeze_510: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 4);  unsqueeze_509 = None
    permute_849: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_510, [3, 4, 1, 2, 0]);  unsqueeze_510 = None
    permute_850: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_848, [0, 4, 1, 2, 3]);  permute_848 = None
    view_764: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_850, [1, 512, 1024]);  permute_850 = None
    permute_851: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_849, [4, 1, 2, 3, 0]);  permute_849 = None
    view_765: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_851, [1, 1024, 1024]);  permute_851 = None
    bmm_161: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_764, view_765);  view_764 = view_765 = None
    view_766: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_161, [512, 1, 1, 16, 64]);  bmm_161 = None
    permute_852: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_766, [0, 2, 3, 4, 1]);  view_766 = None
    view_767: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_852, [512, 1, 16, 64]);  permute_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_511: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_221, 3)
    unsqueeze_512: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 4);  unsqueeze_511 = None
    permute_853: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_512, [0, 1, 3, 4, 2]);  unsqueeze_512 = None
    unsqueeze_513: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, 3);  arg142_1 = None
    unsqueeze_514: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 4);  unsqueeze_513 = None
    permute_854: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_514, [3, 4, 1, 2, 0]);  unsqueeze_514 = None
    permute_855: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_853, [0, 4, 1, 2, 3]);  permute_853 = None
    view_768: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_855, [1, 512, 1024]);  permute_855 = None
    permute_856: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_854, [4, 1, 2, 3, 0]);  permute_854 = None
    view_769: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_856, [1, 1024, 1024]);  permute_856 = None
    bmm_162: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_768, view_769);  view_768 = view_769 = None
    view_770: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_162, [512, 1, 1, 16, 64]);  bmm_162 = None
    permute_857: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_770, [0, 2, 3, 4, 1]);  view_770 = None
    view_771: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_857, [512, 1, 16, 64]);  permute_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_515: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_516: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 4);  unsqueeze_515 = None
    permute_858: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_516, [0, 1, 3, 4, 2]);  unsqueeze_516 = None
    unsqueeze_517: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, 3);  arg143_1 = None
    unsqueeze_518: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 4);  unsqueeze_517 = None
    permute_859: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_518, [3, 4, 1, 2, 0]);  unsqueeze_518 = None
    permute_860: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_858, [0, 4, 1, 2, 3]);  permute_858 = None
    view_772: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_860, [1, 1024, 1024]);  permute_860 = None
    permute_861: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_859, [4, 1, 2, 3, 0]);  permute_859 = None
    view_773: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_861, [1, 1024, 1024]);  permute_861 = None
    bmm_163: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_772, view_773);  view_772 = view_773 = None
    view_774: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_163, [1024, 1, 1, 16, 64]);  bmm_163 = None
    permute_862: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_774, [0, 2, 3, 4, 1]);  view_774 = None
    view_775: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_862, [1024, 1, 16, 64]);  permute_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_222: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_763, arg144_1);  arg144_1 = None
    unsqueeze_519: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_222, 4);  add_222 = None
    permute_863: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_519, [1, 2, 0, 4, 3]);  unsqueeze_519 = None
    unsqueeze_520: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_767, 4);  view_767 = None
    permute_864: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_520, [1, 2, 4, 0, 3]);  unsqueeze_520 = None
    permute_865: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_863, [1, 2, 4, 0, 3]);  permute_863 = None
    view_776: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_865, [16, 512, 64]);  permute_865 = None
    permute_866: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_864, [1, 4, 0, 3, 2]);  permute_864 = None
    view_777: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_866, [16, 64, 512]);  permute_866 = None
    bmm_164: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_776, view_777);  view_776 = view_777 = None
    view_778: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_164, [16, 512, 1, 1, 512]);  bmm_164 = None
    permute_867: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_778, [3, 0, 1, 4, 2]);  view_778 = None
    view_779: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_867, [1, 16, 512, 512]);  permute_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_223: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_763, arg145_1);  view_763 = arg145_1 = None
    unsqueeze_521: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_223, 4);  add_223 = None
    permute_868: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_521, [1, 2, 0, 4, 3]);  unsqueeze_521 = None
    unsqueeze_522: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_775, 4);  view_775 = None
    permute_869: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_522, [1, 2, 4, 0, 3]);  unsqueeze_522 = None
    permute_870: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_868, [1, 2, 4, 0, 3]);  permute_868 = None
    view_780: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_870, [16, 512, 64]);  permute_870 = None
    permute_871: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_869, [1, 4, 0, 3, 2]);  permute_869 = None
    view_781: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_871, [16, 64, 1024]);  permute_871 = None
    bmm_165: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
    view_782: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_165, [16, 512, 1, 1, 1024]);  bmm_165 = None
    permute_872: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_782, [3, 0, 1, 4, 2]);  view_782 = None
    view_783: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_872, [1, 16, 512, 1024]);  permute_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_784: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_783, [1, 16, 1024, 512]);  view_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_164: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_784, 0, 0, 9223372036854775807);  view_784 = None
    slice_165: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_164, 1, 0, 9223372036854775807);  slice_164 = None
    slice_166: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_165, 2, 1, 9223372036854775807);  slice_165 = None
    slice_167: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_166, 3, 0, 9223372036854775807);  slice_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_785: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_167, [1, 16, 512, 1023]);  slice_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_22: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_168: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_785, 0, 0, 9223372036854775807);  view_785 = None
    slice_169: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_168, 1, 0, 9223372036854775807);  slice_168 = None
    slice_170: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_169, 2, 0, 9223372036854775807);  slice_169 = None
    index_20: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_170, [None, None, None, iota_22]);  slice_170 = iota_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_224: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_779, index_20);  view_779 = index_20 = None
    add_225: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_224, 0);  add_224 = None
    mul_164: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_225, 0.125);  add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_20: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_164, [3], True)
    sub_60: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_164, amax_20);  mul_164 = amax_20 = None
    exp_20: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_21: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [3], True)
    div_21: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_122: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_523: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_122, 4);  clone_122 = None
    permute_873: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_523, [2, 0, 1, 4, 3]);  unsqueeze_523 = None
    unsqueeze_524: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_771, 4);  view_771 = None
    permute_874: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_524, [4, 1, 2, 3, 0]);  unsqueeze_524 = None
    permute_875: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_873, [2, 0, 4, 1, 3]);  permute_873 = None
    view_786: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_875, [16, 512, 512]);  permute_875 = None
    permute_876: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_874, [2, 4, 1, 3, 0]);  permute_874 = None
    view_787: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_876, [16, 512, 64]);  permute_876 = None
    bmm_166: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_786, view_787);  view_786 = view_787 = None
    view_788: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_166, [16, 512, 1, 1, 64]);  bmm_166 = None
    permute_877: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_788, [1, 3, 0, 4, 2]);  view_788 = None
    view_789: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_877, [512, 1, 16, 64]);  permute_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_525: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_789, 4);  view_789 = None
    permute_878: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_525, [0, 1, 4, 3, 2]);  unsqueeze_525 = None
    unsqueeze_526: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, 3);  arg146_1 = None
    unsqueeze_527: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 4);  unsqueeze_526 = None
    permute_879: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_527, [3, 4, 0, 2, 1]);  unsqueeze_527 = None
    permute_880: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_878, [0, 3, 4, 1, 2]);  permute_878 = None
    clone_123: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
    view_790: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_123, [1, 512, 1024]);  clone_123 = None
    permute_881: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_879, [3, 4, 1, 2, 0]);  permute_879 = None
    clone_124: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
    view_791: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_124, [1, 1024, 1024]);  clone_124 = None
    bmm_167: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_790, view_791);  view_790 = view_791 = None
    view_792: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_167, [512, 1, 1, 1, 1024]);  bmm_167 = None
    permute_882: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_792, [0, 3, 4, 1, 2]);  view_792 = None
    view_793: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_882, [512, 1, 1024]);  permute_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_125: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_793);  view_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_226: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_125, add_221);  clone_125 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_226, [2], correction = 0, keepdim = True)
    getitem_80: "f32[512, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[512, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_227: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
    rsqrt_40: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    sub_61: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_226, getitem_81);  add_226 = getitem_81 = None
    mul_165: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = rsqrt_40 = None
    mul_166: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_165, arg329_1);  mul_165 = arg329_1 = None
    add_228: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_166, arg330_1);  mul_166 = arg330_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_794: "f32[512, 1024]" = torch.ops.aten.view.default(add_228, [512, 1024])
    permute_883: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
    addmm_40: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg332_1, view_794, permute_883);  arg332_1 = view_794 = permute_883 = None
    view_795: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_40, [512, 1, 4096]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_167: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_795, 0.5)
    mul_168: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_795, 0.7071067811865476);  view_795 = None
    erf_20: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
    add_229: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_169: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_167, add_229);  mul_167 = add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_126: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_169);  mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_796: "f32[512, 4096]" = torch.ops.aten.view.default(clone_126, [512, 4096]);  clone_126 = None
    permute_884: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
    addmm_41: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg334_1, view_796, permute_884);  arg334_1 = view_796 = permute_884 = None
    view_797: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_41, [512, 1, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_127: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_797);  view_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_230: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_127, add_228);  clone_127 = add_228 = None
    var_mean_41 = torch.ops.aten.var_mean.correction(add_230, [2], correction = 0, keepdim = True)
    getitem_82: "f32[512, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[512, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_231: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
    rsqrt_41: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_62: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_230, getitem_83);  add_230 = getitem_83 = None
    mul_170: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
    mul_171: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_170, arg335_1);  mul_170 = arg335_1 = None
    add_232: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_171, arg336_1);  mul_171 = arg336_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_171: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_232, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_42: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_171);  slice_171 = None
    alias_43: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_528: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_232, 3)
    unsqueeze_529: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 4);  unsqueeze_528 = None
    permute_885: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_529, [0, 1, 3, 4, 2]);  unsqueeze_529 = None
    unsqueeze_530: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, 3);  arg147_1 = None
    unsqueeze_531: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 4);  unsqueeze_530 = None
    permute_886: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_531, [3, 4, 1, 2, 0]);  unsqueeze_531 = None
    permute_887: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_885, [0, 4, 1, 2, 3]);  permute_885 = None
    view_798: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_887, [1, 512, 1024]);  permute_887 = None
    permute_888: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_886, [4, 1, 2, 3, 0]);  permute_886 = None
    view_799: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_888, [1, 1024, 1024]);  permute_888 = None
    bmm_168: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_798, view_799);  view_798 = view_799 = None
    view_800: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_168, [512, 1, 1, 16, 64]);  bmm_168 = None
    permute_889: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_800, [0, 2, 3, 4, 1]);  view_800 = None
    view_801: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_889, [512, 1, 16, 64]);  permute_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_532: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_232, 3)
    unsqueeze_533: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 4);  unsqueeze_532 = None
    permute_890: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_533, [0, 1, 3, 4, 2]);  unsqueeze_533 = None
    unsqueeze_534: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, 3);  arg148_1 = None
    unsqueeze_535: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 4);  unsqueeze_534 = None
    permute_891: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_535, [3, 4, 1, 2, 0]);  unsqueeze_535 = None
    permute_892: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_890, [0, 4, 1, 2, 3]);  permute_890 = None
    view_802: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_892, [1, 512, 1024]);  permute_892 = None
    permute_893: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_891, [4, 1, 2, 3, 0]);  permute_891 = None
    view_803: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_893, [1, 1024, 1024]);  permute_893 = None
    bmm_169: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_802, view_803);  view_802 = view_803 = None
    view_804: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_169, [512, 1, 1, 16, 64]);  bmm_169 = None
    permute_894: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_804, [0, 2, 3, 4, 1]);  view_804 = None
    view_805: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_894, [512, 1, 16, 64]);  permute_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_536: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_232, 3)
    unsqueeze_537: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 4);  unsqueeze_536 = None
    permute_895: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_537, [0, 1, 3, 4, 2]);  unsqueeze_537 = None
    unsqueeze_538: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, 3);  arg149_1 = None
    unsqueeze_539: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 4);  unsqueeze_538 = None
    permute_896: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_539, [3, 4, 1, 2, 0]);  unsqueeze_539 = None
    permute_897: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_895, [0, 4, 1, 2, 3]);  permute_895 = None
    view_806: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_897, [1, 512, 1024]);  permute_897 = None
    permute_898: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_896, [4, 1, 2, 3, 0]);  permute_896 = None
    view_807: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_898, [1, 1024, 1024]);  permute_898 = None
    bmm_170: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_806, view_807);  view_806 = view_807 = None
    view_808: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_170, [512, 1, 1, 16, 64]);  bmm_170 = None
    permute_899: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_808, [0, 2, 3, 4, 1]);  view_808 = None
    view_809: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_899, [512, 1, 16, 64]);  permute_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_540: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_541: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 4);  unsqueeze_540 = None
    permute_900: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_541, [0, 1, 3, 4, 2]);  unsqueeze_541 = None
    unsqueeze_542: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, 3);  arg150_1 = None
    unsqueeze_543: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 4);  unsqueeze_542 = None
    permute_901: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_543, [3, 4, 1, 2, 0]);  unsqueeze_543 = None
    permute_902: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_900, [0, 4, 1, 2, 3]);  permute_900 = None
    view_810: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_902, [1, 1024, 1024]);  permute_902 = None
    permute_903: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_901, [4, 1, 2, 3, 0]);  permute_901 = None
    view_811: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_903, [1, 1024, 1024]);  permute_903 = None
    bmm_171: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_810, view_811);  view_810 = view_811 = None
    view_812: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_171, [1024, 1, 1, 16, 64]);  bmm_171 = None
    permute_904: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_812, [0, 2, 3, 4, 1]);  view_812 = None
    view_813: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_904, [1024, 1, 16, 64]);  permute_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_233: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_801, arg151_1);  arg151_1 = None
    unsqueeze_544: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_233, 4);  add_233 = None
    permute_905: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_544, [1, 2, 0, 4, 3]);  unsqueeze_544 = None
    unsqueeze_545: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_805, 4);  view_805 = None
    permute_906: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_545, [1, 2, 4, 0, 3]);  unsqueeze_545 = None
    permute_907: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_905, [1, 2, 4, 0, 3]);  permute_905 = None
    view_814: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_907, [16, 512, 64]);  permute_907 = None
    permute_908: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_906, [1, 4, 0, 3, 2]);  permute_906 = None
    view_815: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_908, [16, 64, 512]);  permute_908 = None
    bmm_172: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_814, view_815);  view_814 = view_815 = None
    view_816: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_172, [16, 512, 1, 1, 512]);  bmm_172 = None
    permute_909: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_816, [3, 0, 1, 4, 2]);  view_816 = None
    view_817: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_909, [1, 16, 512, 512]);  permute_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_234: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_801, arg152_1);  view_801 = arg152_1 = None
    unsqueeze_546: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_234, 4);  add_234 = None
    permute_910: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_546, [1, 2, 0, 4, 3]);  unsqueeze_546 = None
    unsqueeze_547: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_813, 4);  view_813 = None
    permute_911: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_547, [1, 2, 4, 0, 3]);  unsqueeze_547 = None
    permute_912: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_910, [1, 2, 4, 0, 3]);  permute_910 = None
    view_818: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_912, [16, 512, 64]);  permute_912 = None
    permute_913: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_911, [1, 4, 0, 3, 2]);  permute_911 = None
    view_819: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_913, [16, 64, 1024]);  permute_913 = None
    bmm_173: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_818, view_819);  view_818 = view_819 = None
    view_820: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_173, [16, 512, 1, 1, 1024]);  bmm_173 = None
    permute_914: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_820, [3, 0, 1, 4, 2]);  view_820 = None
    view_821: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_914, [1, 16, 512, 1024]);  permute_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_822: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_821, [1, 16, 1024, 512]);  view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_172: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_822, 0, 0, 9223372036854775807);  view_822 = None
    slice_173: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_172, 1, 0, 9223372036854775807);  slice_172 = None
    slice_174: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_173, 2, 1, 9223372036854775807);  slice_173 = None
    slice_175: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_174, 3, 0, 9223372036854775807);  slice_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_823: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_175, [1, 16, 512, 1023]);  slice_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_23: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_176: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_823, 0, 0, 9223372036854775807);  view_823 = None
    slice_177: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_176, 1, 0, 9223372036854775807);  slice_176 = None
    slice_178: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_177, 2, 0, 9223372036854775807);  slice_177 = None
    index_21: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_178, [None, None, None, iota_23]);  slice_178 = iota_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_235: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_817, index_21);  view_817 = index_21 = None
    add_236: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_235, 0);  add_235 = None
    mul_172: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_236, 0.125);  add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_21: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_172, [3], True)
    sub_63: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_172, amax_21);  mul_172 = amax_21 = None
    exp_21: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_22: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [3], True)
    div_22: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_128: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_22);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_548: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_128, 4);  clone_128 = None
    permute_915: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_548, [2, 0, 1, 4, 3]);  unsqueeze_548 = None
    unsqueeze_549: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_809, 4);  view_809 = None
    permute_916: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_549, [4, 1, 2, 3, 0]);  unsqueeze_549 = None
    permute_917: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_915, [2, 0, 4, 1, 3]);  permute_915 = None
    view_824: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_917, [16, 512, 512]);  permute_917 = None
    permute_918: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_916, [2, 4, 1, 3, 0]);  permute_916 = None
    view_825: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_918, [16, 512, 64]);  permute_918 = None
    bmm_174: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_824, view_825);  view_824 = view_825 = None
    view_826: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_174, [16, 512, 1, 1, 64]);  bmm_174 = None
    permute_919: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_826, [1, 3, 0, 4, 2]);  view_826 = None
    view_827: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_919, [512, 1, 16, 64]);  permute_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_550: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_827, 4);  view_827 = None
    permute_920: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_550, [0, 1, 4, 3, 2]);  unsqueeze_550 = None
    unsqueeze_551: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, 3);  arg153_1 = None
    unsqueeze_552: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 4);  unsqueeze_551 = None
    permute_921: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_552, [3, 4, 0, 2, 1]);  unsqueeze_552 = None
    permute_922: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_920, [0, 3, 4, 1, 2]);  permute_920 = None
    clone_129: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_922, memory_format = torch.contiguous_format);  permute_922 = None
    view_828: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_129, [1, 512, 1024]);  clone_129 = None
    permute_923: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_921, [3, 4, 1, 2, 0]);  permute_921 = None
    clone_130: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_923, memory_format = torch.contiguous_format);  permute_923 = None
    view_829: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_130, [1, 1024, 1024]);  clone_130 = None
    bmm_175: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_828, view_829);  view_828 = view_829 = None
    view_830: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_175, [512, 1, 1, 1, 1024]);  bmm_175 = None
    permute_924: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_830, [0, 3, 4, 1, 2]);  view_830 = None
    view_831: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_924, [512, 1, 1024]);  permute_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_131: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_831);  view_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_237: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_131, add_232);  clone_131 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_237, [2], correction = 0, keepdim = True)
    getitem_84: "f32[512, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[512, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_238: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
    rsqrt_42: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
    sub_64: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_237, getitem_85);  add_237 = getitem_85 = None
    mul_173: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = rsqrt_42 = None
    mul_174: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_173, arg337_1);  mul_173 = arg337_1 = None
    add_239: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_174, arg338_1);  mul_174 = arg338_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_832: "f32[512, 1024]" = torch.ops.aten.view.default(add_239, [512, 1024])
    permute_925: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
    addmm_42: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg340_1, view_832, permute_925);  arg340_1 = view_832 = permute_925 = None
    view_833: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_42, [512, 1, 4096]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_175: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_833, 0.5)
    mul_176: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_833, 0.7071067811865476);  view_833 = None
    erf_21: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_176);  mul_176 = None
    add_240: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_177: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_175, add_240);  mul_175 = add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_132: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_177);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_834: "f32[512, 4096]" = torch.ops.aten.view.default(clone_132, [512, 4096]);  clone_132 = None
    permute_926: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
    addmm_43: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg342_1, view_834, permute_926);  arg342_1 = view_834 = permute_926 = None
    view_835: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_43, [512, 1, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_133: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_835);  view_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_241: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_133, add_239);  clone_133 = add_239 = None
    var_mean_43 = torch.ops.aten.var_mean.correction(add_241, [2], correction = 0, keepdim = True)
    getitem_86: "f32[512, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[512, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_242: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
    rsqrt_43: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    sub_65: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_241, getitem_87);  add_241 = getitem_87 = None
    mul_178: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
    mul_179: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_178, arg343_1);  mul_178 = arg343_1 = None
    add_243: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_179, arg344_1);  mul_179 = arg344_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_179: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_243, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_44: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_179);  slice_179 = None
    alias_45: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_553: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_243, 3)
    unsqueeze_554: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 4);  unsqueeze_553 = None
    permute_927: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_554, [0, 1, 3, 4, 2]);  unsqueeze_554 = None
    unsqueeze_555: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, 3);  arg154_1 = None
    unsqueeze_556: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 4);  unsqueeze_555 = None
    permute_928: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_556, [3, 4, 1, 2, 0]);  unsqueeze_556 = None
    permute_929: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_927, [0, 4, 1, 2, 3]);  permute_927 = None
    view_836: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_929, [1, 512, 1024]);  permute_929 = None
    permute_930: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_928, [4, 1, 2, 3, 0]);  permute_928 = None
    view_837: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_930, [1, 1024, 1024]);  permute_930 = None
    bmm_176: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_836, view_837);  view_836 = view_837 = None
    view_838: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_176, [512, 1, 1, 16, 64]);  bmm_176 = None
    permute_931: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_838, [0, 2, 3, 4, 1]);  view_838 = None
    view_839: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_931, [512, 1, 16, 64]);  permute_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_557: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_243, 3)
    unsqueeze_558: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 4);  unsqueeze_557 = None
    permute_932: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_558, [0, 1, 3, 4, 2]);  unsqueeze_558 = None
    unsqueeze_559: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, 3);  arg155_1 = None
    unsqueeze_560: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 4);  unsqueeze_559 = None
    permute_933: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_560, [3, 4, 1, 2, 0]);  unsqueeze_560 = None
    permute_934: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_932, [0, 4, 1, 2, 3]);  permute_932 = None
    view_840: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_934, [1, 512, 1024]);  permute_934 = None
    permute_935: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_933, [4, 1, 2, 3, 0]);  permute_933 = None
    view_841: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_935, [1, 1024, 1024]);  permute_935 = None
    bmm_177: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_840, view_841);  view_840 = view_841 = None
    view_842: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_177, [512, 1, 1, 16, 64]);  bmm_177 = None
    permute_936: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_842, [0, 2, 3, 4, 1]);  view_842 = None
    view_843: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_936, [512, 1, 16, 64]);  permute_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_561: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_243, 3)
    unsqueeze_562: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 4);  unsqueeze_561 = None
    permute_937: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_562, [0, 1, 3, 4, 2]);  unsqueeze_562 = None
    unsqueeze_563: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, 3);  arg156_1 = None
    unsqueeze_564: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 4);  unsqueeze_563 = None
    permute_938: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_564, [3, 4, 1, 2, 0]);  unsqueeze_564 = None
    permute_939: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_937, [0, 4, 1, 2, 3]);  permute_937 = None
    view_844: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_939, [1, 512, 1024]);  permute_939 = None
    permute_940: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_938, [4, 1, 2, 3, 0]);  permute_938 = None
    view_845: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_940, [1, 1024, 1024]);  permute_940 = None
    bmm_178: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_844, view_845);  view_844 = view_845 = None
    view_846: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_178, [512, 1, 1, 16, 64]);  bmm_178 = None
    permute_941: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_846, [0, 2, 3, 4, 1]);  view_846 = None
    view_847: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_941, [512, 1, 16, 64]);  permute_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_565: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3)
    unsqueeze_566: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 4);  unsqueeze_565 = None
    permute_942: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_566, [0, 1, 3, 4, 2]);  unsqueeze_566 = None
    unsqueeze_567: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, 3);  arg157_1 = None
    unsqueeze_568: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 4);  unsqueeze_567 = None
    permute_943: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_568, [3, 4, 1, 2, 0]);  unsqueeze_568 = None
    permute_944: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_942, [0, 4, 1, 2, 3]);  permute_942 = None
    view_848: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_944, [1, 1024, 1024]);  permute_944 = None
    permute_945: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_943, [4, 1, 2, 3, 0]);  permute_943 = None
    view_849: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_945, [1, 1024, 1024]);  permute_945 = None
    bmm_179: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_848, view_849);  view_848 = view_849 = None
    view_850: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_179, [1024, 1, 1, 16, 64]);  bmm_179 = None
    permute_946: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_850, [0, 2, 3, 4, 1]);  view_850 = None
    view_851: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_946, [1024, 1, 16, 64]);  permute_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_244: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_839, arg158_1);  arg158_1 = None
    unsqueeze_569: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_244, 4);  add_244 = None
    permute_947: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_569, [1, 2, 0, 4, 3]);  unsqueeze_569 = None
    unsqueeze_570: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_843, 4);  view_843 = None
    permute_948: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_570, [1, 2, 4, 0, 3]);  unsqueeze_570 = None
    permute_949: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_947, [1, 2, 4, 0, 3]);  permute_947 = None
    view_852: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_949, [16, 512, 64]);  permute_949 = None
    permute_950: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_948, [1, 4, 0, 3, 2]);  permute_948 = None
    view_853: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_950, [16, 64, 512]);  permute_950 = None
    bmm_180: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_852, view_853);  view_852 = view_853 = None
    view_854: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_180, [16, 512, 1, 1, 512]);  bmm_180 = None
    permute_951: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_854, [3, 0, 1, 4, 2]);  view_854 = None
    view_855: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_951, [1, 16, 512, 512]);  permute_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_245: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_839, arg159_1);  view_839 = arg159_1 = None
    unsqueeze_571: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_245, 4);  add_245 = None
    permute_952: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_571, [1, 2, 0, 4, 3]);  unsqueeze_571 = None
    unsqueeze_572: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_851, 4);  view_851 = None
    permute_953: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_572, [1, 2, 4, 0, 3]);  unsqueeze_572 = None
    permute_954: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_952, [1, 2, 4, 0, 3]);  permute_952 = None
    view_856: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_954, [16, 512, 64]);  permute_954 = None
    permute_955: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_953, [1, 4, 0, 3, 2]);  permute_953 = None
    view_857: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_955, [16, 64, 1024]);  permute_955 = None
    bmm_181: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_856, view_857);  view_856 = view_857 = None
    view_858: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_181, [16, 512, 1, 1, 1024]);  bmm_181 = None
    permute_956: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_858, [3, 0, 1, 4, 2]);  view_858 = None
    view_859: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_956, [1, 16, 512, 1024]);  permute_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_860: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_859, [1, 16, 1024, 512]);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_180: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_860, 0, 0, 9223372036854775807);  view_860 = None
    slice_181: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_180, 1, 0, 9223372036854775807);  slice_180 = None
    slice_182: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_181, 2, 1, 9223372036854775807);  slice_181 = None
    slice_183: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_182, 3, 0, 9223372036854775807);  slice_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_861: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_183, [1, 16, 512, 1023]);  slice_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_24: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_184: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_861, 0, 0, 9223372036854775807);  view_861 = None
    slice_185: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_184, 1, 0, 9223372036854775807);  slice_184 = None
    slice_186: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_185, 2, 0, 9223372036854775807);  slice_185 = None
    index_22: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_186, [None, None, None, iota_24]);  slice_186 = iota_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_246: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_855, index_22);  view_855 = index_22 = None
    add_247: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_246, 0);  add_246 = None
    mul_180: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_247, 0.125);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_22: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_180, [3], True)
    sub_66: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_180, amax_22);  mul_180 = amax_22 = None
    exp_22: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
    sum_23: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [3], True)
    div_23: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_134: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_573: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_134, 4);  clone_134 = None
    permute_957: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_573, [2, 0, 1, 4, 3]);  unsqueeze_573 = None
    unsqueeze_574: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_847, 4);  view_847 = None
    permute_958: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_574, [4, 1, 2, 3, 0]);  unsqueeze_574 = None
    permute_959: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_957, [2, 0, 4, 1, 3]);  permute_957 = None
    view_862: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_959, [16, 512, 512]);  permute_959 = None
    permute_960: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_958, [2, 4, 1, 3, 0]);  permute_958 = None
    view_863: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_960, [16, 512, 64]);  permute_960 = None
    bmm_182: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_862, view_863);  view_862 = view_863 = None
    view_864: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_182, [16, 512, 1, 1, 64]);  bmm_182 = None
    permute_961: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_864, [1, 3, 0, 4, 2]);  view_864 = None
    view_865: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_961, [512, 1, 16, 64]);  permute_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_575: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_865, 4);  view_865 = None
    permute_962: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_575, [0, 1, 4, 3, 2]);  unsqueeze_575 = None
    unsqueeze_576: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, 3);  arg160_1 = None
    unsqueeze_577: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 4);  unsqueeze_576 = None
    permute_963: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_577, [3, 4, 0, 2, 1]);  unsqueeze_577 = None
    permute_964: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_962, [0, 3, 4, 1, 2]);  permute_962 = None
    clone_135: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_964, memory_format = torch.contiguous_format);  permute_964 = None
    view_866: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_135, [1, 512, 1024]);  clone_135 = None
    permute_965: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_963, [3, 4, 1, 2, 0]);  permute_963 = None
    clone_136: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_965, memory_format = torch.contiguous_format);  permute_965 = None
    view_867: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_136, [1, 1024, 1024]);  clone_136 = None
    bmm_183: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_866, view_867);  view_866 = view_867 = None
    view_868: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_183, [512, 1, 1, 1, 1024]);  bmm_183 = None
    permute_966: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_868, [0, 3, 4, 1, 2]);  view_868 = None
    view_869: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_966, [512, 1, 1024]);  permute_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_137: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_869);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_248: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_137, add_243);  clone_137 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_248, [2], correction = 0, keepdim = True)
    getitem_88: "f32[512, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[512, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_249: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
    rsqrt_44: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
    sub_67: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_248, getitem_89);  add_248 = getitem_89 = None
    mul_181: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = rsqrt_44 = None
    mul_182: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_181, arg345_1);  mul_181 = arg345_1 = None
    add_250: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_182, arg346_1);  mul_182 = arg346_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_870: "f32[512, 1024]" = torch.ops.aten.view.default(add_250, [512, 1024])
    permute_967: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
    addmm_44: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg348_1, view_870, permute_967);  arg348_1 = view_870 = permute_967 = None
    view_871: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_44, [512, 1, 4096]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_183: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_871, 0.5)
    mul_184: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_871, 0.7071067811865476);  view_871 = None
    erf_22: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_184);  mul_184 = None
    add_251: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_185: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_183, add_251);  mul_183 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_138: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_185);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_872: "f32[512, 4096]" = torch.ops.aten.view.default(clone_138, [512, 4096]);  clone_138 = None
    permute_968: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
    addmm_45: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg350_1, view_872, permute_968);  arg350_1 = view_872 = permute_968 = None
    view_873: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_45, [512, 1, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_139: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_873);  view_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_252: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_139, add_250);  clone_139 = add_250 = None
    var_mean_45 = torch.ops.aten.var_mean.correction(add_252, [2], correction = 0, keepdim = True)
    getitem_90: "f32[512, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[512, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_253: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
    rsqrt_45: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_253);  add_253 = None
    sub_68: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_252, getitem_91);  add_252 = getitem_91 = None
    mul_186: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
    mul_187: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_186, arg351_1);  mul_186 = arg351_1 = None
    add_254: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_187, arg352_1);  mul_187 = arg352_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    slice_187: "f32[512, 1, 1024]" = torch.ops.aten.slice.Tensor(add_254, 0, -512, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    alias_46: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(slice_187);  slice_187 = None
    alias_47: "f32[512, 1, 1024]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    unsqueeze_578: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_254, 3)
    unsqueeze_579: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 4);  unsqueeze_578 = None
    permute_969: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_579, [0, 1, 3, 4, 2]);  unsqueeze_579 = None
    unsqueeze_580: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, 3);  arg161_1 = None
    unsqueeze_581: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 4);  unsqueeze_580 = None
    permute_970: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_581, [3, 4, 1, 2, 0]);  unsqueeze_581 = None
    permute_971: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_969, [0, 4, 1, 2, 3]);  permute_969 = None
    view_874: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_971, [1, 512, 1024]);  permute_971 = None
    permute_972: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_970, [4, 1, 2, 3, 0]);  permute_970 = None
    view_875: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_972, [1, 1024, 1024]);  permute_972 = None
    bmm_184: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_874, view_875);  view_874 = view_875 = None
    view_876: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_184, [512, 1, 1, 16, 64]);  bmm_184 = None
    permute_973: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_876, [0, 2, 3, 4, 1]);  view_876 = None
    view_877: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_973, [512, 1, 16, 64]);  permute_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    unsqueeze_582: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_254, 3)
    unsqueeze_583: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 4);  unsqueeze_582 = None
    permute_974: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_583, [0, 1, 3, 4, 2]);  unsqueeze_583 = None
    unsqueeze_584: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, 3);  arg162_1 = None
    unsqueeze_585: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 4);  unsqueeze_584 = None
    permute_975: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_585, [3, 4, 1, 2, 0]);  unsqueeze_585 = None
    permute_976: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_974, [0, 4, 1, 2, 3]);  permute_974 = None
    view_878: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_976, [1, 512, 1024]);  permute_976 = None
    permute_977: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_975, [4, 1, 2, 3, 0]);  permute_975 = None
    view_879: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_977, [1, 1024, 1024]);  permute_977 = None
    bmm_185: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_878, view_879);  view_878 = view_879 = None
    view_880: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_185, [512, 1, 1, 16, 64]);  bmm_185 = None
    permute_978: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_880, [0, 2, 3, 4, 1]);  view_880 = None
    view_881: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_978, [512, 1, 16, 64]);  permute_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    unsqueeze_586: "f32[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_254, 3)
    unsqueeze_587: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 4);  unsqueeze_586 = None
    permute_979: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_587, [0, 1, 3, 4, 2]);  unsqueeze_587 = None
    unsqueeze_588: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, 3);  arg163_1 = None
    unsqueeze_589: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 4);  unsqueeze_588 = None
    permute_980: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_589, [3, 4, 1, 2, 0]);  unsqueeze_589 = None
    permute_981: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_979, [0, 4, 1, 2, 3]);  permute_979 = None
    view_882: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_981, [1, 512, 1024]);  permute_981 = None
    permute_982: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_980, [4, 1, 2, 3, 0]);  permute_980 = None
    view_883: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_982, [1, 1024, 1024]);  permute_982 = None
    bmm_186: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_882, view_883);  view_882 = view_883 = None
    view_884: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_186, [512, 1, 1, 16, 64]);  bmm_186 = None
    permute_983: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_884, [0, 2, 3, 4, 1]);  view_884 = None
    view_885: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_983, [512, 1, 16, 64]);  permute_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    unsqueeze_590: "f32[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_1, 3);  clone_1 = None
    unsqueeze_591: "f32[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 4);  unsqueeze_590 = None
    permute_984: "f32[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_591, [0, 1, 3, 4, 2]);  unsqueeze_591 = None
    unsqueeze_592: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, 3);  arg164_1 = None
    unsqueeze_593: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 4);  unsqueeze_592 = None
    permute_985: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(unsqueeze_593, [3, 4, 1, 2, 0]);  unsqueeze_593 = None
    permute_986: "f32[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_984, [0, 4, 1, 2, 3]);  permute_984 = None
    view_886: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_986, [1, 1024, 1024]);  permute_986 = None
    permute_987: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_985, [4, 1, 2, 3, 0]);  permute_985 = None
    view_887: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(permute_987, [1, 1024, 1024]);  permute_987 = None
    bmm_187: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(view_886, view_887);  view_886 = view_887 = None
    view_888: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(bmm_187, [1024, 1, 1, 16, 64]);  bmm_187 = None
    permute_988: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_888, [0, 2, 3, 4, 1]);  view_888 = None
    view_889: "f32[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_988, [1024, 1, 16, 64]);  permute_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_255: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_877, arg165_1);  arg165_1 = None
    unsqueeze_594: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_255, 4);  add_255 = None
    permute_989: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_594, [1, 2, 0, 4, 3]);  unsqueeze_594 = None
    unsqueeze_595: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_881, 4);  view_881 = None
    permute_990: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_595, [1, 2, 4, 0, 3]);  unsqueeze_595 = None
    permute_991: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_989, [1, 2, 4, 0, 3]);  permute_989 = None
    view_890: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_991, [16, 512, 64]);  permute_991 = None
    permute_992: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_990, [1, 4, 0, 3, 2]);  permute_990 = None
    view_891: "f32[16, 64, 512]" = torch.ops.aten.view.default(permute_992, [16, 64, 512]);  permute_992 = None
    bmm_188: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_890, view_891);  view_890 = view_891 = None
    view_892: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_188, [16, 512, 1, 1, 512]);  bmm_188 = None
    permute_993: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_892, [3, 0, 1, 4, 2]);  view_892 = None
    view_893: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_993, [1, 16, 512, 512]);  permute_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    add_256: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_877, arg166_1);  view_877 = arg166_1 = None
    unsqueeze_596: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_256, 4);  add_256 = None
    permute_994: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_596, [1, 2, 0, 4, 3]);  unsqueeze_596 = None
    unsqueeze_597: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_889, 4);  view_889 = None
    permute_995: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_597, [1, 2, 4, 0, 3]);  unsqueeze_597 = None
    permute_996: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_994, [1, 2, 4, 0, 3]);  permute_994 = None
    view_894: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_996, [16, 512, 64]);  permute_996 = None
    permute_997: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_995, [1, 4, 0, 3, 2]);  permute_995 = None
    view_895: "f32[16, 64, 1024]" = torch.ops.aten.view.default(permute_997, [16, 64, 1024]);  permute_997 = None
    bmm_189: "f32[16, 512, 1024]" = torch.ops.aten.bmm.default(view_894, view_895);  view_894 = view_895 = None
    view_896: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_189, [16, 512, 1, 1, 1024]);  bmm_189 = None
    permute_998: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_896, [3, 0, 1, 4, 2]);  view_896 = None
    view_897: "f32[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_998, [1, 16, 512, 1024]);  permute_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_898: "f32[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_897, [1, 16, 1024, 512]);  view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_188: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(view_898, 0, 0, 9223372036854775807);  view_898 = None
    slice_189: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice.Tensor(slice_188, 1, 0, 9223372036854775807);  slice_188 = None
    slice_190: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_189, 2, 1, 9223372036854775807);  slice_189 = None
    slice_191: "f32[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(slice_190, 3, 0, 9223372036854775807);  slice_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_899: "f32[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_191, [1, 16, 512, 1023]);  slice_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    iota_25: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_192: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(view_899, 0, 0, 9223372036854775807);  view_899 = None
    slice_193: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_192, 1, 0, 9223372036854775807);  slice_192 = None
    slice_194: "f32[1, 16, 512, 1023]" = torch.ops.aten.slice.Tensor(slice_193, 2, 0, 9223372036854775807);  slice_193 = None
    index_23: "f32[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(slice_194, [None, None, None, iota_25]);  slice_194 = iota_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_257: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_893, index_23);  view_893 = index_23 = None
    add_258: "f32[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_257, 0);  add_257 = None
    mul_188: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_258, 0.125);  add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    amax_23: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_188, [3], True)
    sub_69: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_188, amax_23);  mul_188 = amax_23 = None
    exp_23: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
    sum_24: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [3], True)
    div_24: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    clone_140: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(div_24);  div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    unsqueeze_598: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_140, 4);  clone_140 = None
    permute_999: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_598, [2, 0, 1, 4, 3]);  unsqueeze_598 = None
    unsqueeze_599: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_885, 4);  view_885 = None
    permute_1000: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_599, [4, 1, 2, 3, 0]);  unsqueeze_599 = None
    permute_1001: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_999, [2, 0, 4, 1, 3]);  permute_999 = None
    view_900: "f32[16, 512, 512]" = torch.ops.aten.view.default(permute_1001, [16, 512, 512]);  permute_1001 = None
    permute_1002: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_1000, [2, 4, 1, 3, 0]);  permute_1000 = None
    view_901: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_1002, [16, 512, 64]);  permute_1002 = None
    bmm_190: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_900, view_901);  view_900 = view_901 = None
    view_902: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_190, [16, 512, 1, 1, 64]);  bmm_190 = None
    permute_1003: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_902, [1, 3, 0, 4, 2]);  view_902 = None
    view_903: "f32[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_1003, [512, 1, 16, 64]);  permute_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    unsqueeze_600: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_903, 4);  view_903 = None
    permute_1004: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_600, [0, 1, 4, 3, 2]);  unsqueeze_600 = None
    unsqueeze_601: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, 3);  arg167_1 = None
    unsqueeze_602: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 4);  unsqueeze_601 = None
    permute_1005: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_602, [3, 4, 0, 2, 1]);  unsqueeze_602 = None
    permute_1006: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_1004, [0, 3, 4, 1, 2]);  permute_1004 = None
    clone_141: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_1006, memory_format = torch.contiguous_format);  permute_1006 = None
    view_904: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_141, [1, 512, 1024]);  clone_141 = None
    permute_1007: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_1005, [3, 4, 1, 2, 0]);  permute_1005 = None
    clone_142: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.clone.default(permute_1007, memory_format = torch.contiguous_format);  permute_1007 = None
    view_905: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_142, [1, 1024, 1024]);  clone_142 = None
    bmm_191: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_904, view_905);  view_904 = view_905 = None
    view_906: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_191, [512, 1, 1, 1, 1024]);  bmm_191 = None
    permute_1008: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(view_906, [0, 3, 4, 1, 2]);  view_906 = None
    view_907: "f32[512, 1, 1024]" = torch.ops.aten.view.default(permute_1008, [512, 1, 1024]);  permute_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    clone_143: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_907);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    add_259: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_143, add_254);  clone_143 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_259, [2], correction = 0, keepdim = True)
    getitem_92: "f32[512, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[512, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_260: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_46: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
    sub_70: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_259, getitem_93);  add_259 = getitem_93 = None
    mul_189: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = rsqrt_46 = None
    mul_190: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_189, arg353_1);  mul_189 = arg353_1 = None
    add_261: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_190, arg354_1);  mul_190 = arg354_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_908: "f32[512, 1024]" = torch.ops.aten.view.default(add_261, [512, 1024])
    permute_1009: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
    addmm_46: "f32[512, 4096]" = torch.ops.aten.addmm.default(arg356_1, view_908, permute_1009);  arg356_1 = view_908 = permute_1009 = None
    view_909: "f32[512, 1, 4096]" = torch.ops.aten.view.default(addmm_46, [512, 1, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_191: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_909, 0.5)
    mul_192: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_909, 0.7071067811865476);  view_909 = None
    erf_23: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_192);  mul_192 = None
    add_262: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_193: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_191, add_262);  mul_191 = add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    clone_144: "f32[512, 1, 4096]" = torch.ops.aten.clone.default(mul_193);  mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_910: "f32[512, 4096]" = torch.ops.aten.view.default(clone_144, [512, 4096]);  clone_144 = None
    permute_1010: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
    addmm_47: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg358_1, view_910, permute_1010);  arg358_1 = view_910 = permute_1010 = None
    view_911: "f32[512, 1, 1024]" = torch.ops.aten.view.default(addmm_47, [512, 1, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    clone_145: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(view_911);  view_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_263: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(clone_145, add_261);  clone_145 = add_261 = None
    var_mean_47 = torch.ops.aten.var_mean.correction(add_263, [2], correction = 0, keepdim = True)
    getitem_94: "f32[512, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[512, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_264: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
    rsqrt_47: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
    sub_71: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(add_263, getitem_95);  add_263 = getitem_95 = None
    mul_194: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
    mul_195: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_194, arg359_1);  mul_194 = arg359_1 = None
    add_265: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_195, arg360_1);  mul_195 = arg360_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1257, code: output = self.dropout(output_g if output_g is not None else output_h)
    clone_146: "f32[512, 1, 1024]" = torch.ops.aten.clone.default(add_265);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1260, code: output = output.permute(1, 0, 2).contiguous()
    permute_1011: "f32[1, 512, 1024]" = torch.ops.aten.permute.default(clone_146, [1, 0, 2]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1463, code: logits = self.lm_loss(transformer_outputs[0])
    view_912: "f32[512, 1024]" = torch.ops.aten.view.default(permute_1011, [512, 1024]);  permute_1011 = None
    permute_1012: "f32[1024, 32000]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
    addmm_48: "f32[512, 32000]" = torch.ops.aten.addmm.default(arg362_1, view_912, permute_1012);  arg362_1 = view_912 = permute_1012 = None
    view_913: "f32[1, 512, 32000]" = torch.ops.aten.view.default(addmm_48, [1, 512, 32000]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1469, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    view_914: "f32[512, 32000]" = torch.ops.aten.view.default(view_913, [-1, 32000])
    view_915: "i64[512]" = torch.ops.aten.view.default(arg364_1, [-1]);  arg364_1 = None
    amax_24: "f32[512, 1]" = torch.ops.aten.amax.default(view_914, [1], True)
    sub_72: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(view_914, amax_24);  view_914 = amax_24 = None
    exp_24: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_72)
    sum_25: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_73: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(sub_72, log);  sub_72 = log = None
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_915, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[512]" = torch.ops.aten.where.self(ne, view_915, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_603: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_73, 1, unsqueeze_603);  sub_73 = unsqueeze_603 = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[512]" = torch.ops.aten.ne.Scalar(view_915, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[512]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[512]" = torch.ops.aten.ne.Scalar(view_915, -100);  view_915 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_4: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_25: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_4);  sum_27 = convert_element_type_4 = None
    return (div_25, view_913, alias_1, alias_3, alias_5, alias_7, alias_9, alias_11, alias_13, alias_15, alias_17, alias_19, alias_21, alias_23, alias_25, alias_27, alias_29, alias_31, alias_33, alias_35, alias_37, alias_39, alias_41, alias_43, alias_45, alias_47)
    