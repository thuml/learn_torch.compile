from __future__ import annotations



def forward(self, arg0_1: "f32[128100, 1536]", arg1_1: "f32[512, 1536]", arg2_1: "f32[1536]", arg3_1: "f32[1536]", arg4_1: "f32[1536, 1536]", arg5_1: "f32[1536]", arg6_1: "f32[1536, 1536]", arg7_1: "f32[1536]", arg8_1: "f32[1536, 1536]", arg9_1: "f32[1536]", arg10_1: "f32[1536, 1536]", arg11_1: "f32[1536]", arg12_1: "f32[1536]", arg13_1: "f32[1536]", arg14_1: "f32[6144, 1536]", arg15_1: "f32[6144]", arg16_1: "f32[1536, 6144]", arg17_1: "f32[1536]", arg18_1: "f32[1536]", arg19_1: "f32[1536]", arg20_1: "f32[1536, 1536]", arg21_1: "f32[1536]", arg22_1: "f32[1536, 1536]", arg23_1: "f32[1536]", arg24_1: "f32[1536, 1536]", arg25_1: "f32[1536]", arg26_1: "f32[1536, 1536]", arg27_1: "f32[1536]", arg28_1: "f32[1536]", arg29_1: "f32[1536]", arg30_1: "f32[6144, 1536]", arg31_1: "f32[6144]", arg32_1: "f32[1536, 6144]", arg33_1: "f32[1536]", arg34_1: "f32[1536]", arg35_1: "f32[1536]", arg36_1: "f32[1536, 1536]", arg37_1: "f32[1536]", arg38_1: "f32[1536, 1536]", arg39_1: "f32[1536]", arg40_1: "f32[1536, 1536]", arg41_1: "f32[1536]", arg42_1: "f32[1536, 1536]", arg43_1: "f32[1536]", arg44_1: "f32[1536]", arg45_1: "f32[1536]", arg46_1: "f32[6144, 1536]", arg47_1: "f32[6144]", arg48_1: "f32[1536, 6144]", arg49_1: "f32[1536]", arg50_1: "f32[1536]", arg51_1: "f32[1536]", arg52_1: "f32[1536, 1536]", arg53_1: "f32[1536]", arg54_1: "f32[1536, 1536]", arg55_1: "f32[1536]", arg56_1: "f32[1536, 1536]", arg57_1: "f32[1536]", arg58_1: "f32[1536, 1536]", arg59_1: "f32[1536]", arg60_1: "f32[1536]", arg61_1: "f32[1536]", arg62_1: "f32[6144, 1536]", arg63_1: "f32[6144]", arg64_1: "f32[1536, 6144]", arg65_1: "f32[1536]", arg66_1: "f32[1536]", arg67_1: "f32[1536]", arg68_1: "f32[1536, 1536]", arg69_1: "f32[1536]", arg70_1: "f32[1536, 1536]", arg71_1: "f32[1536]", arg72_1: "f32[1536, 1536]", arg73_1: "f32[1536]", arg74_1: "f32[1536, 1536]", arg75_1: "f32[1536]", arg76_1: "f32[1536]", arg77_1: "f32[1536]", arg78_1: "f32[6144, 1536]", arg79_1: "f32[6144]", arg80_1: "f32[1536, 6144]", arg81_1: "f32[1536]", arg82_1: "f32[1536]", arg83_1: "f32[1536]", arg84_1: "f32[1536, 1536]", arg85_1: "f32[1536]", arg86_1: "f32[1536, 1536]", arg87_1: "f32[1536]", arg88_1: "f32[1536, 1536]", arg89_1: "f32[1536]", arg90_1: "f32[1536, 1536]", arg91_1: "f32[1536]", arg92_1: "f32[1536]", arg93_1: "f32[1536]", arg94_1: "f32[6144, 1536]", arg95_1: "f32[6144]", arg96_1: "f32[1536, 6144]", arg97_1: "f32[1536]", arg98_1: "f32[1536]", arg99_1: "f32[1536]", arg100_1: "f32[1536, 1536]", arg101_1: "f32[1536]", arg102_1: "f32[1536, 1536]", arg103_1: "f32[1536]", arg104_1: "f32[1536, 1536]", arg105_1: "f32[1536]", arg106_1: "f32[1536, 1536]", arg107_1: "f32[1536]", arg108_1: "f32[1536]", arg109_1: "f32[1536]", arg110_1: "f32[6144, 1536]", arg111_1: "f32[6144]", arg112_1: "f32[1536, 6144]", arg113_1: "f32[1536]", arg114_1: "f32[1536]", arg115_1: "f32[1536]", arg116_1: "f32[1536, 1536]", arg117_1: "f32[1536]", arg118_1: "f32[1536, 1536]", arg119_1: "f32[1536]", arg120_1: "f32[1536, 1536]", arg121_1: "f32[1536]", arg122_1: "f32[1536, 1536]", arg123_1: "f32[1536]", arg124_1: "f32[1536]", arg125_1: "f32[1536]", arg126_1: "f32[6144, 1536]", arg127_1: "f32[6144]", arg128_1: "f32[1536, 6144]", arg129_1: "f32[1536]", arg130_1: "f32[1536]", arg131_1: "f32[1536]", arg132_1: "f32[1536, 1536]", arg133_1: "f32[1536]", arg134_1: "f32[1536, 1536]", arg135_1: "f32[1536]", arg136_1: "f32[1536, 1536]", arg137_1: "f32[1536]", arg138_1: "f32[1536, 1536]", arg139_1: "f32[1536]", arg140_1: "f32[1536]", arg141_1: "f32[1536]", arg142_1: "f32[6144, 1536]", arg143_1: "f32[6144]", arg144_1: "f32[1536, 6144]", arg145_1: "f32[1536]", arg146_1: "f32[1536]", arg147_1: "f32[1536]", arg148_1: "f32[1536, 1536]", arg149_1: "f32[1536]", arg150_1: "f32[1536, 1536]", arg151_1: "f32[1536]", arg152_1: "f32[1536, 1536]", arg153_1: "f32[1536]", arg154_1: "f32[1536, 1536]", arg155_1: "f32[1536]", arg156_1: "f32[1536]", arg157_1: "f32[1536]", arg158_1: "f32[6144, 1536]", arg159_1: "f32[6144]", arg160_1: "f32[1536, 6144]", arg161_1: "f32[1536]", arg162_1: "f32[1536]", arg163_1: "f32[1536]", arg164_1: "f32[1536, 1536]", arg165_1: "f32[1536]", arg166_1: "f32[1536, 1536]", arg167_1: "f32[1536]", arg168_1: "f32[1536, 1536]", arg169_1: "f32[1536]", arg170_1: "f32[1536, 1536]", arg171_1: "f32[1536]", arg172_1: "f32[1536]", arg173_1: "f32[1536]", arg174_1: "f32[6144, 1536]", arg175_1: "f32[6144]", arg176_1: "f32[1536, 6144]", arg177_1: "f32[1536]", arg178_1: "f32[1536]", arg179_1: "f32[1536]", arg180_1: "f32[1536, 1536]", arg181_1: "f32[1536]", arg182_1: "f32[1536, 1536]", arg183_1: "f32[1536]", arg184_1: "f32[1536, 1536]", arg185_1: "f32[1536]", arg186_1: "f32[1536, 1536]", arg187_1: "f32[1536]", arg188_1: "f32[1536]", arg189_1: "f32[1536]", arg190_1: "f32[6144, 1536]", arg191_1: "f32[6144]", arg192_1: "f32[1536, 6144]", arg193_1: "f32[1536]", arg194_1: "f32[1536]", arg195_1: "f32[1536]", arg196_1: "f32[1536, 1536]", arg197_1: "f32[1536]", arg198_1: "f32[1536, 1536]", arg199_1: "f32[1536]", arg200_1: "f32[1536, 1536]", arg201_1: "f32[1536]", arg202_1: "f32[1536, 1536]", arg203_1: "f32[1536]", arg204_1: "f32[1536]", arg205_1: "f32[1536]", arg206_1: "f32[6144, 1536]", arg207_1: "f32[6144]", arg208_1: "f32[1536, 6144]", arg209_1: "f32[1536]", arg210_1: "f32[1536]", arg211_1: "f32[1536]", arg212_1: "f32[1536, 1536]", arg213_1: "f32[1536]", arg214_1: "f32[1536, 1536]", arg215_1: "f32[1536]", arg216_1: "f32[1536, 1536]", arg217_1: "f32[1536]", arg218_1: "f32[1536, 1536]", arg219_1: "f32[1536]", arg220_1: "f32[1536]", arg221_1: "f32[1536]", arg222_1: "f32[6144, 1536]", arg223_1: "f32[6144]", arg224_1: "f32[1536, 6144]", arg225_1: "f32[1536]", arg226_1: "f32[1536]", arg227_1: "f32[1536]", arg228_1: "f32[1536, 1536]", arg229_1: "f32[1536]", arg230_1: "f32[1536, 1536]", arg231_1: "f32[1536]", arg232_1: "f32[1536, 1536]", arg233_1: "f32[1536]", arg234_1: "f32[1536, 1536]", arg235_1: "f32[1536]", arg236_1: "f32[1536]", arg237_1: "f32[1536]", arg238_1: "f32[6144, 1536]", arg239_1: "f32[6144]", arg240_1: "f32[1536, 6144]", arg241_1: "f32[1536]", arg242_1: "f32[1536]", arg243_1: "f32[1536]", arg244_1: "f32[1536, 1536]", arg245_1: "f32[1536]", arg246_1: "f32[1536, 1536]", arg247_1: "f32[1536]", arg248_1: "f32[1536, 1536]", arg249_1: "f32[1536]", arg250_1: "f32[1536, 1536]", arg251_1: "f32[1536]", arg252_1: "f32[1536]", arg253_1: "f32[1536]", arg254_1: "f32[6144, 1536]", arg255_1: "f32[6144]", arg256_1: "f32[1536, 6144]", arg257_1: "f32[1536]", arg258_1: "f32[1536]", arg259_1: "f32[1536]", arg260_1: "f32[1536, 1536]", arg261_1: "f32[1536]", arg262_1: "f32[1536, 1536]", arg263_1: "f32[1536]", arg264_1: "f32[1536, 1536]", arg265_1: "f32[1536]", arg266_1: "f32[1536, 1536]", arg267_1: "f32[1536]", arg268_1: "f32[1536]", arg269_1: "f32[1536]", arg270_1: "f32[6144, 1536]", arg271_1: "f32[6144]", arg272_1: "f32[1536, 6144]", arg273_1: "f32[1536]", arg274_1: "f32[1536]", arg275_1: "f32[1536]", arg276_1: "f32[1536, 1536]", arg277_1: "f32[1536]", arg278_1: "f32[1536, 1536]", arg279_1: "f32[1536]", arg280_1: "f32[1536, 1536]", arg281_1: "f32[1536]", arg282_1: "f32[1536, 1536]", arg283_1: "f32[1536]", arg284_1: "f32[1536]", arg285_1: "f32[1536]", arg286_1: "f32[6144, 1536]", arg287_1: "f32[6144]", arg288_1: "f32[1536, 6144]", arg289_1: "f32[1536]", arg290_1: "f32[1536]", arg291_1: "f32[1536]", arg292_1: "f32[1536, 1536]", arg293_1: "f32[1536]", arg294_1: "f32[1536, 1536]", arg295_1: "f32[1536]", arg296_1: "f32[1536, 1536]", arg297_1: "f32[1536]", arg298_1: "f32[1536, 1536]", arg299_1: "f32[1536]", arg300_1: "f32[1536]", arg301_1: "f32[1536]", arg302_1: "f32[6144, 1536]", arg303_1: "f32[6144]", arg304_1: "f32[1536, 6144]", arg305_1: "f32[1536]", arg306_1: "f32[1536]", arg307_1: "f32[1536]", arg308_1: "f32[1536, 1536]", arg309_1: "f32[1536]", arg310_1: "f32[1536, 1536]", arg311_1: "f32[1536]", arg312_1: "f32[1536, 1536]", arg313_1: "f32[1536]", arg314_1: "f32[1536, 1536]", arg315_1: "f32[1536]", arg316_1: "f32[1536]", arg317_1: "f32[1536]", arg318_1: "f32[6144, 1536]", arg319_1: "f32[6144]", arg320_1: "f32[1536, 6144]", arg321_1: "f32[1536]", arg322_1: "f32[1536]", arg323_1: "f32[1536]", arg324_1: "f32[1536, 1536]", arg325_1: "f32[1536]", arg326_1: "f32[1536, 1536]", arg327_1: "f32[1536]", arg328_1: "f32[1536, 1536]", arg329_1: "f32[1536]", arg330_1: "f32[1536, 1536]", arg331_1: "f32[1536]", arg332_1: "f32[1536]", arg333_1: "f32[1536]", arg334_1: "f32[6144, 1536]", arg335_1: "f32[6144]", arg336_1: "f32[1536, 6144]", arg337_1: "f32[1536]", arg338_1: "f32[1536]", arg339_1: "f32[1536]", arg340_1: "f32[1536, 1536]", arg341_1: "f32[1536]", arg342_1: "f32[1536, 1536]", arg343_1: "f32[1536]", arg344_1: "f32[1536, 1536]", arg345_1: "f32[1536]", arg346_1: "f32[1536, 1536]", arg347_1: "f32[1536]", arg348_1: "f32[1536]", arg349_1: "f32[1536]", arg350_1: "f32[6144, 1536]", arg351_1: "f32[6144]", arg352_1: "f32[1536, 6144]", arg353_1: "f32[1536]", arg354_1: "f32[1536]", arg355_1: "f32[1536]", arg356_1: "f32[1536, 1536]", arg357_1: "f32[1536]", arg358_1: "f32[1536, 1536]", arg359_1: "f32[1536]", arg360_1: "f32[1536, 1536]", arg361_1: "f32[1536]", arg362_1: "f32[1536, 1536]", arg363_1: "f32[1536]", arg364_1: "f32[1536]", arg365_1: "f32[1536]", arg366_1: "f32[6144, 1536]", arg367_1: "f32[6144]", arg368_1: "f32[1536, 6144]", arg369_1: "f32[1536]", arg370_1: "f32[1536]", arg371_1: "f32[1536]", arg372_1: "f32[1536, 1536]", arg373_1: "f32[1536]", arg374_1: "f32[1536, 1536]", arg375_1: "f32[1536]", arg376_1: "f32[1536, 1536]", arg377_1: "f32[1536]", arg378_1: "f32[1536, 1536]", arg379_1: "f32[1536]", arg380_1: "f32[1536]", arg381_1: "f32[1536]", arg382_1: "f32[6144, 1536]", arg383_1: "f32[6144]", arg384_1: "f32[1536, 6144]", arg385_1: "f32[1536]", arg386_1: "f32[1536]", arg387_1: "f32[1536]", arg388_1: "f32[2, 1536]", arg389_1: "f32[2]", arg390_1: "i64[1, 512]", arg391_1: "i64[1, 512]", arg392_1: "i64[1]", arg393_1: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1072, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:878, code: position_ids = self.position_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(arg390_1, 0, 0, 9223372036854775807);  arg390_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:884, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 1536]" = torch.ops.aten.embedding.default(arg0_1, arg391_1, 0);  arg0_1 = arg391_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:887, code: position_embeddings = self.position_embeddings(position_ids.long())
    embedding_1: "f32[1, 512, 1536]" = torch.ops.aten.embedding.default(arg1_1, slice_1);  arg1_1 = slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:893, code: embeddings += position_embeddings
    add: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:901, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
    mul: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_2: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:907, code: mask = mask.unsqueeze(2)
    unsqueeze: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(full, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:910, code: embeddings = embeddings * mask
    mul_2: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_2, unsqueeze);  add_2 = unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:454, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    unsqueeze_1: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_2: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 1, 512]" = torch.ops.aten.squeeze.dim(unsqueeze_2, -2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:455, code: attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    unsqueeze_3: "f32[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
    mul_3: "f32[1, 1, 512, 512]" = torch.ops.aten.mul.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view: "f32[512, 1536]" = torch.ops.aten.view.default(mul_2, [512, 1536])
    permute: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg5_1, view, permute);  arg5_1 = view = permute = None
    view_1: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm, [1, 512, 1536]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_2: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_1, [1, 512, 24, -1]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_1: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    clone: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_3: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone, [-1, 512, 64]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_4: "f32[512, 1536]" = torch.ops.aten.view.default(mul_2, [512, 1536])
    permute_2: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm_1: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg7_1, view_4, permute_2);  arg7_1 = view_4 = permute_2 = None
    view_5: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_1, [1, 512, 1536]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_6: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_5, [1, 512, 24, -1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_3: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    clone_1: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    view_7: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_1, [-1, 512, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_8: "f32[512, 1536]" = torch.ops.aten.view.default(mul_2, [512, 1536])
    permute_4: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    addmm_2: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg9_1, view_8, permute_4);  arg9_1 = view_8 = permute_4 = None
    view_9: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_2, [1, 512, 1536]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_10: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_9, [1, 512, 24, -1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_5: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_2: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_11: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_2, [-1, 512, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    mul_4: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy, 1);  lift_fresh_copy = None
    sqrt: "f32[]" = torch.ops.aten.sqrt.default(mul_4);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_6: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    div: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_6, sqrt);  permute_6 = sqrt = None
    bmm: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_3, div);  view_3 = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_12: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm, [-1, 24, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type);  convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant1 = self._tensor_constant1
    lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    where: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not, lift_fresh_copy_1, view_12);  lift_fresh_copy_1 = view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not, scalar_tensor, div_1);  bitwise_not = scalar_tensor = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_14: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_1, [-1, 512, 512]);  where_1 = None
    bmm_1: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_11);  view_14 = view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_15: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_1, [-1, 24, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_7: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_3: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_16: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_3, [1, 512, -1]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_17: "f32[512, 1536]" = torch.ops.aten.view.default(view_16, [512, 1536]);  view_16 = None
    permute_8: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm_3: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg11_1, view_17, permute_8);  arg11_1 = view_17 = permute_8 = None
    view_18: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_3, [1, 512, 1536]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_3: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_18, mul_2);  view_18 = mul_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-07);  getitem_2 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
    mul_5: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_6: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_5, arg12_1);  mul_5 = arg12_1 = None
    add_5: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_6, arg13_1);  mul_6 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[512, 1536]" = torch.ops.aten.view.default(add_5, [512, 1536])
    permute_9: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    addmm_4: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg15_1, view_19, permute_9);  arg15_1 = view_19 = permute_9 = None
    view_20: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_4, [1, 512, 6144]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_7: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_20, 0.5)
    mul_8: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476);  view_20 = None
    erf: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_6: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_7, add_6);  mul_7 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_21: "f32[512, 6144]" = torch.ops.aten.view.default(mul_9, [512, 6144]);  mul_9 = None
    permute_10: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_5: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg17_1, view_21, permute_10);  arg17_1 = view_21 = permute_10 = None
    view_22: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_5, [1, 512, 1536]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_7: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_22, add_5);  view_22 = add_5 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-07);  getitem_4 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  add_7 = getitem_5 = None
    mul_10: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_11: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_10, arg18_1);  mul_10 = arg18_1 = None
    add_9: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_11, arg19_1);  mul_11 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_23: "f32[512, 1536]" = torch.ops.aten.view.default(add_9, [512, 1536])
    permute_11: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_6: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg21_1, view_23, permute_11);  arg21_1 = view_23 = permute_11 = None
    view_24: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_6, [1, 512, 1536]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_25: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_24, [1, 512, 24, -1]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_12: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_4: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_26: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_4, [-1, 512, 64]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_27: "f32[512, 1536]" = torch.ops.aten.view.default(add_9, [512, 1536])
    permute_13: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    addmm_7: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg23_1, view_27, permute_13);  arg23_1 = view_27 = permute_13 = None
    view_28: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_7, [1, 512, 1536]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_29: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_28, [1, 512, 24, -1]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_14: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_5: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_30: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_5, [-1, 512, 64]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_31: "f32[512, 1536]" = torch.ops.aten.view.default(add_9, [512, 1536])
    permute_15: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_8: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg25_1, view_31, permute_15);  arg25_1 = view_31 = permute_15 = None
    view_32: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_8, [1, 512, 1536]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_33: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_32, [1, 512, 24, -1]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_16: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    clone_6: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_34: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_6, [-1, 512, 64]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant2 = self._tensor_constant2
    lift_fresh_copy_2: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
    mul_12: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_2, 1);  lift_fresh_copy_2 = None
    sqrt_1: "f32[]" = torch.ops.aten.sqrt.default(mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_17: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
    div_2: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_17, sqrt_1);  permute_17 = sqrt_1 = None
    bmm_2: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_26, div_2);  view_26 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_35: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_2, [-1, 24, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_1: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_1: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_1);  convert_element_type_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant3 = self._tensor_constant3
    lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
    where_2: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_1, lift_fresh_copy_3, view_35);  lift_fresh_copy_3 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_1: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_4: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_2, amax_1);  where_2 = amax_1 = None
    exp_1: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_1, scalar_tensor_1, div_3);  bitwise_not_1 = scalar_tensor_1 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_37: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_3, [-1, 512, 512]);  where_3 = None
    bmm_3: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_37, view_34);  view_37 = view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_38: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_3, [-1, 24, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_18: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_7: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_39: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_7, [1, 512, -1]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 1536]" = torch.ops.aten.view.default(view_39, [512, 1536]);  view_39 = None
    permute_19: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_9: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg27_1, view_40, permute_19);  arg27_1 = view_40 = permute_19 = None
    view_41: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_9, [1, 512, 1536]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_10: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_41, add_9);  view_41 = add_9 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-07);  getitem_6 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_5: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_10, getitem_7);  add_10 = getitem_7 = None
    mul_13: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_14: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_13, arg28_1);  mul_13 = arg28_1 = None
    add_12: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_14, arg29_1);  mul_14 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 1536]" = torch.ops.aten.view.default(add_12, [512, 1536])
    permute_20: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_10: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg31_1, view_42, permute_20);  arg31_1 = view_42 = permute_20 = None
    view_43: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_10, [1, 512, 6144]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_15: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_16: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_1: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
    add_13: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_17: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_15, add_13);  mul_15 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_44: "f32[512, 6144]" = torch.ops.aten.view.default(mul_17, [512, 6144]);  mul_17 = None
    permute_21: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    addmm_11: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg33_1, view_44, permute_21);  arg33_1 = view_44 = permute_21 = None
    view_45: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_11, [1, 512, 1536]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_14: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_45, add_12);  view_45 = add_12 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-07);  getitem_8 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  add_14 = getitem_9 = None
    mul_18: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_19: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_18, arg34_1);  mul_18 = arg34_1 = None
    add_16: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_19, arg35_1);  mul_19 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_46: "f32[512, 1536]" = torch.ops.aten.view.default(add_16, [512, 1536])
    permute_22: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_12: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg37_1, view_46, permute_22);  arg37_1 = view_46 = permute_22 = None
    view_47: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_12, [1, 512, 1536]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_48: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_47, [1, 512, 24, -1]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_23: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_8: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_49: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_8, [-1, 512, 64]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_50: "f32[512, 1536]" = torch.ops.aten.view.default(add_16, [512, 1536])
    permute_24: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    addmm_13: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg39_1, view_50, permute_24);  arg39_1 = view_50 = permute_24 = None
    view_51: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_13, [1, 512, 1536]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_52: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_51, [1, 512, 24, -1]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_25: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    clone_9: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_53: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_9, [-1, 512, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_54: "f32[512, 1536]" = torch.ops.aten.view.default(add_16, [512, 1536])
    permute_26: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    addmm_14: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg41_1, view_54, permute_26);  arg41_1 = view_54 = permute_26 = None
    view_55: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_14, [1, 512, 1536]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_56: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_55, [1, 512, 24, -1]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_27: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
    clone_10: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    view_57: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_10, [-1, 512, 64]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant4 = self._tensor_constant4
    lift_fresh_copy_4: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
    mul_20: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_4, 1);  lift_fresh_copy_4 = None
    sqrt_2: "f32[]" = torch.ops.aten.sqrt.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_28: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    div_4: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_28, sqrt_2);  permute_28 = sqrt_2 = None
    bmm_4: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_49, div_4);  view_49 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_58: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_4, [-1, 24, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_2: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_2: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_2);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant5 = self._tensor_constant5
    lift_fresh_copy_5: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
    where_4: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_2, lift_fresh_copy_5, view_58);  lift_fresh_copy_5 = view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_2: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_7: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_4, amax_2);  where_4 = amax_2 = None
    exp_2: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_2, scalar_tensor_2, div_5);  bitwise_not_2 = scalar_tensor_2 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_60: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_5, [-1, 512, 512]);  where_5 = None
    bmm_5: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_60, view_57);  view_60 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_61: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_5, [-1, 24, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_29: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_11: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_62: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_11, [1, 512, -1]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[512, 1536]" = torch.ops.aten.view.default(view_62, [512, 1536]);  view_62 = None
    permute_30: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_15: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg43_1, view_63, permute_30);  arg43_1 = view_63 = permute_30 = None
    view_64: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_15, [1, 512, 1536]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_64, add_16);  view_64 = add_16 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-07);  getitem_10 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_8: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  add_17 = getitem_11 = None
    mul_21: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_22: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_21, arg44_1);  mul_21 = arg44_1 = None
    add_19: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_22, arg45_1);  mul_22 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_65: "f32[512, 1536]" = torch.ops.aten.view.default(add_19, [512, 1536])
    permute_31: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_16: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg47_1, view_65, permute_31);  arg47_1 = view_65 = permute_31 = None
    view_66: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_16, [1, 512, 6144]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_23: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
    mul_24: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476);  view_66 = None
    erf_2: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_20: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_23, add_20);  mul_23 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_67: "f32[512, 6144]" = torch.ops.aten.view.default(mul_25, [512, 6144]);  mul_25 = None
    permute_32: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_17: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg49_1, view_67, permute_32);  arg49_1 = view_67 = permute_32 = None
    view_68: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_17, [1, 512, 1536]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_68, add_19);  view_68 = add_19 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-07);  getitem_12 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_9: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_21, getitem_13);  add_21 = getitem_13 = None
    mul_26: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_27: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_26, arg50_1);  mul_26 = arg50_1 = None
    add_23: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_27, arg51_1);  mul_27 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_69: "f32[512, 1536]" = torch.ops.aten.view.default(add_23, [512, 1536])
    permute_33: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_18: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg53_1, view_69, permute_33);  arg53_1 = view_69 = permute_33 = None
    view_70: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_18, [1, 512, 1536]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_71: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_70, [1, 512, 24, -1]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_34: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    clone_12: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    view_72: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_12, [-1, 512, 64]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_73: "f32[512, 1536]" = torch.ops.aten.view.default(add_23, [512, 1536])
    permute_35: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_19: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg55_1, view_73, permute_35);  arg55_1 = view_73 = permute_35 = None
    view_74: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_19, [1, 512, 1536]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_75: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_74, [1, 512, 24, -1]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_36: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    clone_13: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    view_76: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_13, [-1, 512, 64]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_77: "f32[512, 1536]" = torch.ops.aten.view.default(add_23, [512, 1536])
    permute_37: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_20: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg57_1, view_77, permute_37);  arg57_1 = view_77 = permute_37 = None
    view_78: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_20, [1, 512, 1536]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_79: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_78, [1, 512, 24, -1]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_38: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
    clone_14: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_80: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_14, [-1, 512, 64]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant6 = self._tensor_constant6
    lift_fresh_copy_6: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
    mul_28: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_6, 1);  lift_fresh_copy_6 = None
    sqrt_3: "f32[]" = torch.ops.aten.sqrt.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_39: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    div_6: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_39, sqrt_3);  permute_39 = sqrt_3 = None
    bmm_6: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_72, div_6);  view_72 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_81: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_6, [-1, 24, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_3: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_3: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_3);  convert_element_type_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant7 = self._tensor_constant7
    lift_fresh_copy_7: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
    where_6: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_3, lift_fresh_copy_7, view_81);  lift_fresh_copy_7 = view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_3: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_10: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_6, amax_3);  where_6 = amax_3 = None
    exp_3: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_3, scalar_tensor_3, div_7);  bitwise_not_3 = scalar_tensor_3 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_83: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_7, [-1, 512, 512]);  where_7 = None
    bmm_7: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_83, view_80);  view_83 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_84: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_7, [-1, 24, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_40: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_15: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_85: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_15, [1, 512, -1]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 1536]" = torch.ops.aten.view.default(view_85, [512, 1536]);  view_85 = None
    permute_41: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_21: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg59_1, view_86, permute_41);  arg59_1 = view_86 = permute_41 = None
    view_87: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_21, [1, 512, 1536]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_24: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_87, add_23);  view_87 = add_23 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-07);  getitem_14 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_11: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_24, getitem_15);  add_24 = getitem_15 = None
    mul_29: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_30: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_29, arg60_1);  mul_29 = arg60_1 = None
    add_26: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_30, arg61_1);  mul_30 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_88: "f32[512, 1536]" = torch.ops.aten.view.default(add_26, [512, 1536])
    permute_42: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    addmm_22: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg63_1, view_88, permute_42);  arg63_1 = view_88 = permute_42 = None
    view_89: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_22, [1, 512, 6144]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_31: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    mul_32: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_89, 0.7071067811865476);  view_89 = None
    erf_3: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_32);  mul_32 = None
    add_27: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_33: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_31, add_27);  mul_31 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_90: "f32[512, 6144]" = torch.ops.aten.view.default(mul_33, [512, 6144]);  mul_33 = None
    permute_43: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_23: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg65_1, view_90, permute_43);  arg65_1 = view_90 = permute_43 = None
    view_91: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_23, [1, 512, 1536]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_28: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_91, add_26);  view_91 = add_26 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_29: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-07);  getitem_16 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_12: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_28, getitem_17);  add_28 = getitem_17 = None
    mul_34: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_35: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_34, arg66_1);  mul_34 = arg66_1 = None
    add_30: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_35, arg67_1);  mul_35 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_92: "f32[512, 1536]" = torch.ops.aten.view.default(add_30, [512, 1536])
    permute_44: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm_24: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg69_1, view_92, permute_44);  arg69_1 = view_92 = permute_44 = None
    view_93: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_24, [1, 512, 1536]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_94: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_93, [1, 512, 24, -1]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_45: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    clone_16: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_95: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_16, [-1, 512, 64]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_96: "f32[512, 1536]" = torch.ops.aten.view.default(add_30, [512, 1536])
    permute_46: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_25: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg71_1, view_96, permute_46);  arg71_1 = view_96 = permute_46 = None
    view_97: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_25, [1, 512, 1536]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_98: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_97, [1, 512, 24, -1]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_47: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    clone_17: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    view_99: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_17, [-1, 512, 64]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_100: "f32[512, 1536]" = torch.ops.aten.view.default(add_30, [512, 1536])
    permute_48: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_26: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg73_1, view_100, permute_48);  arg73_1 = view_100 = permute_48 = None
    view_101: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_26, [1, 512, 1536]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_102: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_101, [1, 512, 24, -1]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_49: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_18: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_103: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_18, [-1, 512, 64]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant8 = self._tensor_constant8
    lift_fresh_copy_8: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
    mul_36: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_8, 1);  lift_fresh_copy_8 = None
    sqrt_4: "f32[]" = torch.ops.aten.sqrt.default(mul_36);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_50: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    div_8: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_50, sqrt_4);  permute_50 = sqrt_4 = None
    bmm_8: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_95, div_8);  view_95 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_104: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_8, [-1, 24, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_4: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_4: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_4);  convert_element_type_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant9 = self._tensor_constant9
    lift_fresh_copy_9: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
    where_8: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_4, lift_fresh_copy_9, view_104);  lift_fresh_copy_9 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_4: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_13: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_8, amax_4);  where_8 = amax_4 = None
    exp_4: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_4, scalar_tensor_4, div_9);  bitwise_not_4 = scalar_tensor_4 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_106: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_9, [-1, 512, 512]);  where_9 = None
    bmm_9: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_106, view_103);  view_106 = view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_107: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_9, [-1, 24, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_51: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_19: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_108: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_19, [1, 512, -1]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_109: "f32[512, 1536]" = torch.ops.aten.view.default(view_108, [512, 1536]);  view_108 = None
    permute_52: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_27: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg75_1, view_109, permute_52);  arg75_1 = view_109 = permute_52 = None
    view_110: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_27, [1, 512, 1536]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_31: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_110, add_30);  view_110 = add_30 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-07);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_14: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_31, getitem_19);  add_31 = getitem_19 = None
    mul_37: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_38: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_37, arg76_1);  mul_37 = arg76_1 = None
    add_33: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_38, arg77_1);  mul_38 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_111: "f32[512, 1536]" = torch.ops.aten.view.default(add_33, [512, 1536])
    permute_53: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_28: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg79_1, view_111, permute_53);  arg79_1 = view_111 = permute_53 = None
    view_112: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_28, [1, 512, 6144]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_39: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_112, 0.5)
    mul_40: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_112, 0.7071067811865476);  view_112 = None
    erf_4: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_34: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_41: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_39, add_34);  mul_39 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_113: "f32[512, 6144]" = torch.ops.aten.view.default(mul_41, [512, 6144]);  mul_41 = None
    permute_54: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_29: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg81_1, view_113, permute_54);  arg81_1 = view_113 = permute_54 = None
    view_114: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_29, [1, 512, 1536]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_35: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_114, add_33);  view_114 = add_33 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-07);  getitem_20 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_15: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_35, getitem_21);  add_35 = getitem_21 = None
    mul_42: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_43: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_42, arg82_1);  mul_42 = arg82_1 = None
    add_37: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_43, arg83_1);  mul_43 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_115: "f32[512, 1536]" = torch.ops.aten.view.default(add_37, [512, 1536])
    permute_55: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_30: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg85_1, view_115, permute_55);  arg85_1 = view_115 = permute_55 = None
    view_116: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_30, [1, 512, 1536]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_117: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_116, [1, 512, 24, -1]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_56: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    clone_20: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_118: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_20, [-1, 512, 64]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_119: "f32[512, 1536]" = torch.ops.aten.view.default(add_37, [512, 1536])
    permute_57: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_31: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg87_1, view_119, permute_57);  arg87_1 = view_119 = permute_57 = None
    view_120: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_31, [1, 512, 1536]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_121: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_120, [1, 512, 24, -1]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_58: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
    clone_21: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    view_122: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_21, [-1, 512, 64]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_123: "f32[512, 1536]" = torch.ops.aten.view.default(add_37, [512, 1536])
    permute_59: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_32: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg89_1, view_123, permute_59);  arg89_1 = view_123 = permute_59 = None
    view_124: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_32, [1, 512, 1536]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_125: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_124, [1, 512, 24, -1]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_60: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
    clone_22: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    view_126: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_22, [-1, 512, 64]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant10 = self._tensor_constant10
    lift_fresh_copy_10: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
    mul_44: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_10, 1);  lift_fresh_copy_10 = None
    sqrt_5: "f32[]" = torch.ops.aten.sqrt.default(mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_61: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    div_10: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_61, sqrt_5);  permute_61 = sqrt_5 = None
    bmm_10: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_118, div_10);  view_118 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_127: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_10, [-1, 24, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_5: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_5: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_5);  convert_element_type_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant11 = self._tensor_constant11
    lift_fresh_copy_11: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
    where_10: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_5, lift_fresh_copy_11, view_127);  lift_fresh_copy_11 = view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_5: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_16: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_10, amax_5);  where_10 = amax_5 = None
    exp_5: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_5, scalar_tensor_5, div_11);  bitwise_not_5 = scalar_tensor_5 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_129: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_11, [-1, 512, 512]);  where_11 = None
    bmm_11: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_129, view_126);  view_129 = view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_130: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_11, [-1, 24, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_62: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_23: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_131: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_23, [1, 512, -1]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_132: "f32[512, 1536]" = torch.ops.aten.view.default(view_131, [512, 1536]);  view_131 = None
    permute_63: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_33: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg91_1, view_132, permute_63);  arg91_1 = view_132 = permute_63 = None
    view_133: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_33, [1, 512, 1536]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_38: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_133, add_37);  view_133 = add_37 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-07);  getitem_22 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_17: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_38, getitem_23);  add_38 = getitem_23 = None
    mul_45: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_46: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_45, arg92_1);  mul_45 = arg92_1 = None
    add_40: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_46, arg93_1);  mul_46 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_134: "f32[512, 1536]" = torch.ops.aten.view.default(add_40, [512, 1536])
    permute_64: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_34: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg95_1, view_134, permute_64);  arg95_1 = view_134 = permute_64 = None
    view_135: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_34, [1, 512, 6144]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_135, 0.5)
    mul_48: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_135, 0.7071067811865476);  view_135 = None
    erf_5: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_41: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_49: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_47, add_41);  mul_47 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_136: "f32[512, 6144]" = torch.ops.aten.view.default(mul_49, [512, 6144]);  mul_49 = None
    permute_65: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_35: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg97_1, view_136, permute_65);  arg97_1 = view_136 = permute_65 = None
    view_137: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_35, [1, 512, 1536]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_42: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_137, add_40);  view_137 = add_40 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_43: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-07);  getitem_24 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_18: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_42, getitem_25);  add_42 = getitem_25 = None
    mul_50: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_51: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_50, arg98_1);  mul_50 = arg98_1 = None
    add_44: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_51, arg99_1);  mul_51 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_138: "f32[512, 1536]" = torch.ops.aten.view.default(add_44, [512, 1536])
    permute_66: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_36: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg101_1, view_138, permute_66);  arg101_1 = view_138 = permute_66 = None
    view_139: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_36, [1, 512, 1536]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_140: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_139, [1, 512, 24, -1]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_67: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    clone_24: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    view_141: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_24, [-1, 512, 64]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_142: "f32[512, 1536]" = torch.ops.aten.view.default(add_44, [512, 1536])
    permute_68: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_37: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg103_1, view_142, permute_68);  arg103_1 = view_142 = permute_68 = None
    view_143: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_37, [1, 512, 1536]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_144: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_143, [1, 512, 24, -1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_69: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
    clone_25: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_145: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_25, [-1, 512, 64]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_146: "f32[512, 1536]" = torch.ops.aten.view.default(add_44, [512, 1536])
    permute_70: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_38: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg105_1, view_146, permute_70);  arg105_1 = view_146 = permute_70 = None
    view_147: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_38, [1, 512, 1536]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_148: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_147, [1, 512, 24, -1]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_71: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_26: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    view_149: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_26, [-1, 512, 64]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant12 = self._tensor_constant12
    lift_fresh_copy_12: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
    mul_52: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_12, 1);  lift_fresh_copy_12 = None
    sqrt_6: "f32[]" = torch.ops.aten.sqrt.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_72: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    div_12: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_72, sqrt_6);  permute_72 = sqrt_6 = None
    bmm_12: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_141, div_12);  view_141 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_150: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_12, [-1, 24, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_6: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_6: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_6);  convert_element_type_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant13 = self._tensor_constant13
    lift_fresh_copy_13: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
    where_12: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_6, lift_fresh_copy_13, view_150);  lift_fresh_copy_13 = view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_6: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_12, [-1], True)
    sub_19: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_12, amax_6);  where_12 = amax_6 = None
    exp_6: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_6, scalar_tensor_6, div_13);  bitwise_not_6 = scalar_tensor_6 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_152: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_13, [-1, 512, 512]);  where_13 = None
    bmm_13: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_152, view_149);  view_152 = view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_153: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_13, [-1, 24, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_73: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_27: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_154: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_27, [1, 512, -1]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_155: "f32[512, 1536]" = torch.ops.aten.view.default(view_154, [512, 1536]);  view_154 = None
    permute_74: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_39: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg107_1, view_155, permute_74);  arg107_1 = view_155 = permute_74 = None
    view_156: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_39, [1, 512, 1536]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_156, add_44);  view_156 = add_44 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-07);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_20: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_45, getitem_27);  add_45 = getitem_27 = None
    mul_53: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_54: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_53, arg108_1);  mul_53 = arg108_1 = None
    add_47: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_54, arg109_1);  mul_54 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_157: "f32[512, 1536]" = torch.ops.aten.view.default(add_47, [512, 1536])
    permute_75: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_40: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg111_1, view_157, permute_75);  arg111_1 = view_157 = permute_75 = None
    view_158: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_40, [1, 512, 6144]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_56: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_6: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_48: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_57: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_55, add_48);  mul_55 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_159: "f32[512, 6144]" = torch.ops.aten.view.default(mul_57, [512, 6144]);  mul_57 = None
    permute_76: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_41: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg113_1, view_159, permute_76);  arg113_1 = view_159 = permute_76 = None
    view_160: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_41, [1, 512, 1536]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_160, add_47);  view_160 = add_47 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-07);  getitem_28 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_21: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_49, getitem_29);  add_49 = getitem_29 = None
    mul_58: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_59: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_58, arg114_1);  mul_58 = arg114_1 = None
    add_51: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_59, arg115_1);  mul_59 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_161: "f32[512, 1536]" = torch.ops.aten.view.default(add_51, [512, 1536])
    permute_77: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_42: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg117_1, view_161, permute_77);  arg117_1 = view_161 = permute_77 = None
    view_162: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_42, [1, 512, 1536]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_163: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_162, [1, 512, 24, -1]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_78: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    clone_28: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    view_164: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_28, [-1, 512, 64]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_165: "f32[512, 1536]" = torch.ops.aten.view.default(add_51, [512, 1536])
    permute_79: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_43: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg119_1, view_165, permute_79);  arg119_1 = view_165 = permute_79 = None
    view_166: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_43, [1, 512, 1536]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_167: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_166, [1, 512, 24, -1]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_80: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
    clone_29: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    view_168: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_29, [-1, 512, 64]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_169: "f32[512, 1536]" = torch.ops.aten.view.default(add_51, [512, 1536])
    permute_81: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_44: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg121_1, view_169, permute_81);  arg121_1 = view_169 = permute_81 = None
    view_170: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_44, [1, 512, 1536]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_171: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_170, [1, 512, 24, -1]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_82: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    clone_30: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_172: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_30, [-1, 512, 64]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant14 = self._tensor_constant14
    lift_fresh_copy_14: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
    mul_60: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_14, 1);  lift_fresh_copy_14 = None
    sqrt_7: "f32[]" = torch.ops.aten.sqrt.default(mul_60);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_83: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    div_14: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_83, sqrt_7);  permute_83 = sqrt_7 = None
    bmm_14: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_164, div_14);  view_164 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_173: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_14, [-1, 24, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_7: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_7: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_7);  convert_element_type_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant15 = self._tensor_constant15
    lift_fresh_copy_15: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
    where_14: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_7, lift_fresh_copy_15, view_173);  lift_fresh_copy_15 = view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_7: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_14, [-1], True)
    sub_22: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_14, amax_7);  where_14 = amax_7 = None
    exp_7: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_7, scalar_tensor_7, div_15);  bitwise_not_7 = scalar_tensor_7 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_175: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_15, [-1, 512, 512]);  where_15 = None
    bmm_15: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_175, view_172);  view_175 = view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_176: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_15, [-1, 24, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_84: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_31: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_177: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_31, [1, 512, -1]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_178: "f32[512, 1536]" = torch.ops.aten.view.default(view_177, [512, 1536]);  view_177 = None
    permute_85: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_45: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg123_1, view_178, permute_85);  arg123_1 = view_178 = permute_85 = None
    view_179: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_45, [1, 512, 1536]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_52: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_179, add_51);  view_179 = add_51 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_53: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-07);  getitem_30 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_23: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_52, getitem_31);  add_52 = getitem_31 = None
    mul_61: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_62: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_61, arg124_1);  mul_61 = arg124_1 = None
    add_54: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_62, arg125_1);  mul_62 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_180: "f32[512, 1536]" = torch.ops.aten.view.default(add_54, [512, 1536])
    permute_86: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_46: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg127_1, view_180, permute_86);  arg127_1 = view_180 = permute_86 = None
    view_181: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_46, [1, 512, 6144]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_63: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
    mul_64: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476);  view_181 = None
    erf_7: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_55: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_65: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_63, add_55);  mul_63 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_182: "f32[512, 6144]" = torch.ops.aten.view.default(mul_65, [512, 6144]);  mul_65 = None
    permute_87: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_47: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg129_1, view_182, permute_87);  arg129_1 = view_182 = permute_87 = None
    view_183: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_47, [1, 512, 1536]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_56: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_183, add_54);  view_183 = add_54 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_57: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-07);  getitem_32 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_24: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_56, getitem_33);  add_56 = getitem_33 = None
    mul_66: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_67: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_66, arg130_1);  mul_66 = arg130_1 = None
    add_58: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_67, arg131_1);  mul_67 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_184: "f32[512, 1536]" = torch.ops.aten.view.default(add_58, [512, 1536])
    permute_88: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_48: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg133_1, view_184, permute_88);  arg133_1 = view_184 = permute_88 = None
    view_185: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_48, [1, 512, 1536]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_186: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_185, [1, 512, 24, -1]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_89: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_32: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
    view_187: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_32, [-1, 512, 64]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_188: "f32[512, 1536]" = torch.ops.aten.view.default(add_58, [512, 1536])
    permute_90: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_49: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg135_1, view_188, permute_90);  arg135_1 = view_188 = permute_90 = None
    view_189: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_49, [1, 512, 1536]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_190: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_189, [1, 512, 24, -1]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_91: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_33: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    view_191: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_33, [-1, 512, 64]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_192: "f32[512, 1536]" = torch.ops.aten.view.default(add_58, [512, 1536])
    permute_92: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_50: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg137_1, view_192, permute_92);  arg137_1 = view_192 = permute_92 = None
    view_193: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_50, [1, 512, 1536]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_194: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_193, [1, 512, 24, -1]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_93: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
    clone_34: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    view_195: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_34, [-1, 512, 64]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant16 = self._tensor_constant16
    lift_fresh_copy_16: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
    mul_68: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_16, 1);  lift_fresh_copy_16 = None
    sqrt_8: "f32[]" = torch.ops.aten.sqrt.default(mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_94: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    div_16: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_94, sqrt_8);  permute_94 = sqrt_8 = None
    bmm_16: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_187, div_16);  view_187 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_196: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_16, [-1, 24, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_8: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_8: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_8);  convert_element_type_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant17 = self._tensor_constant17
    lift_fresh_copy_17: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
    where_16: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_8, lift_fresh_copy_17, view_196);  lift_fresh_copy_17 = view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_8: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_16, [-1], True)
    sub_25: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_16, amax_8);  where_16 = amax_8 = None
    exp_8: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_8, scalar_tensor_8, div_17);  bitwise_not_8 = scalar_tensor_8 = div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_198: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_17, [-1, 512, 512]);  where_17 = None
    bmm_17: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_198, view_195);  view_198 = view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_199: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_17, [-1, 24, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_95: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_35: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_200: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_35, [1, 512, -1]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_201: "f32[512, 1536]" = torch.ops.aten.view.default(view_200, [512, 1536]);  view_200 = None
    permute_96: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_51: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg139_1, view_201, permute_96);  arg139_1 = view_201 = permute_96 = None
    view_202: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_51, [1, 512, 1536]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_202, add_58);  view_202 = add_58 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-07);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_26: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_59, getitem_35);  add_59 = getitem_35 = None
    mul_69: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_70: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_69, arg140_1);  mul_69 = arg140_1 = None
    add_61: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_70, arg141_1);  mul_70 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_203: "f32[512, 1536]" = torch.ops.aten.view.default(add_61, [512, 1536])
    permute_97: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_52: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg143_1, view_203, permute_97);  arg143_1 = view_203 = permute_97 = None
    view_204: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_52, [1, 512, 6144]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_204, 0.5)
    mul_72: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_204, 0.7071067811865476);  view_204 = None
    erf_8: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_62: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_73: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_71, add_62);  mul_71 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_205: "f32[512, 6144]" = torch.ops.aten.view.default(mul_73, [512, 6144]);  mul_73 = None
    permute_98: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_53: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg145_1, view_205, permute_98);  arg145_1 = view_205 = permute_98 = None
    view_206: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_53, [1, 512, 1536]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_206, add_61);  view_206 = add_61 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-07);  getitem_36 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_27: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_63, getitem_37);  add_63 = getitem_37 = None
    mul_74: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_75: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_74, arg146_1);  mul_74 = arg146_1 = None
    add_65: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_75, arg147_1);  mul_75 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_207: "f32[512, 1536]" = torch.ops.aten.view.default(add_65, [512, 1536])
    permute_99: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_54: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg149_1, view_207, permute_99);  arg149_1 = view_207 = permute_99 = None
    view_208: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_54, [1, 512, 1536]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_209: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_208, [1, 512, 24, -1]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_100: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
    clone_36: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
    view_210: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_36, [-1, 512, 64]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_211: "f32[512, 1536]" = torch.ops.aten.view.default(add_65, [512, 1536])
    permute_101: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_55: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg151_1, view_211, permute_101);  arg151_1 = view_211 = permute_101 = None
    view_212: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_55, [1, 512, 1536]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_213: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_212, [1, 512, 24, -1]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_102: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    clone_37: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_214: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_37, [-1, 512, 64]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_215: "f32[512, 1536]" = torch.ops.aten.view.default(add_65, [512, 1536])
    permute_103: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_56: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg153_1, view_215, permute_103);  arg153_1 = view_215 = permute_103 = None
    view_216: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_56, [1, 512, 1536]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_217: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_216, [1, 512, 24, -1]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_104: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
    clone_38: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_218: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_38, [-1, 512, 64]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant18 = self._tensor_constant18
    lift_fresh_copy_18: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
    mul_76: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_18, 1);  lift_fresh_copy_18 = None
    sqrt_9: "f32[]" = torch.ops.aten.sqrt.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_105: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
    div_18: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_105, sqrt_9);  permute_105 = sqrt_9 = None
    bmm_18: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_210, div_18);  view_210 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_219: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_18, [-1, 24, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_9: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_9: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_9);  convert_element_type_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant19 = self._tensor_constant19
    lift_fresh_copy_19: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant19);  _tensor_constant19 = None
    where_18: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_9, lift_fresh_copy_19, view_219);  lift_fresh_copy_19 = view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_9: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_18, [-1], True)
    sub_28: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_18, amax_9);  where_18 = amax_9 = None
    exp_9: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_9, scalar_tensor_9, div_19);  bitwise_not_9 = scalar_tensor_9 = div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_221: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_19, [-1, 512, 512]);  where_19 = None
    bmm_19: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_221, view_218);  view_221 = view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_222: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_19, [-1, 24, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_106: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_39: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_223: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_39, [1, 512, -1]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_224: "f32[512, 1536]" = torch.ops.aten.view.default(view_223, [512, 1536]);  view_223 = None
    permute_107: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_57: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg155_1, view_224, permute_107);  arg155_1 = view_224 = permute_107 = None
    view_225: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_57, [1, 512, 1536]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_66: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_225, add_65);  view_225 = add_65 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_67: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-07);  getitem_38 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_29: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_66, getitem_39);  add_66 = getitem_39 = None
    mul_77: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_78: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_77, arg156_1);  mul_77 = arg156_1 = None
    add_68: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_78, arg157_1);  mul_78 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_226: "f32[512, 1536]" = torch.ops.aten.view.default(add_68, [512, 1536])
    permute_108: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_58: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg159_1, view_226, permute_108);  arg159_1 = view_226 = permute_108 = None
    view_227: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_58, [1, 512, 6144]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_79: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_227, 0.5)
    mul_80: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_227, 0.7071067811865476);  view_227 = None
    erf_9: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_80);  mul_80 = None
    add_69: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_81: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_79, add_69);  mul_79 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_228: "f32[512, 6144]" = torch.ops.aten.view.default(mul_81, [512, 6144]);  mul_81 = None
    permute_109: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_59: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg161_1, view_228, permute_109);  arg161_1 = view_228 = permute_109 = None
    view_229: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_59, [1, 512, 1536]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_70: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_229, add_68);  view_229 = add_68 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_71: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-07);  getitem_40 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_30: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_70, getitem_41);  add_70 = getitem_41 = None
    mul_82: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_83: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_82, arg162_1);  mul_82 = arg162_1 = None
    add_72: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_83, arg163_1);  mul_83 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_230: "f32[512, 1536]" = torch.ops.aten.view.default(add_72, [512, 1536])
    permute_110: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_60: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg165_1, view_230, permute_110);  arg165_1 = view_230 = permute_110 = None
    view_231: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_60, [1, 512, 1536]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_232: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_231, [1, 512, 24, -1]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_111: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
    clone_40: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_233: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_40, [-1, 512, 64]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_234: "f32[512, 1536]" = torch.ops.aten.view.default(add_72, [512, 1536])
    permute_112: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_61: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg167_1, view_234, permute_112);  arg167_1 = view_234 = permute_112 = None
    view_235: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_61, [1, 512, 1536]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_236: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_235, [1, 512, 24, -1]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_113: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_41: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    view_237: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_41, [-1, 512, 64]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_238: "f32[512, 1536]" = torch.ops.aten.view.default(add_72, [512, 1536])
    permute_114: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_62: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg169_1, view_238, permute_114);  arg169_1 = view_238 = permute_114 = None
    view_239: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_62, [1, 512, 1536]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_240: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_239, [1, 512, 24, -1]);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_115: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    clone_42: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_241: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_42, [-1, 512, 64]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant20 = self._tensor_constant20
    lift_fresh_copy_20: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
    mul_84: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_20, 1);  lift_fresh_copy_20 = None
    sqrt_10: "f32[]" = torch.ops.aten.sqrt.default(mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_116: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
    div_20: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_116, sqrt_10);  permute_116 = sqrt_10 = None
    bmm_20: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_233, div_20);  view_233 = div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_242: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_20, [-1, 24, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_10: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_10: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_10);  convert_element_type_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant21 = self._tensor_constant21
    lift_fresh_copy_21: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant21);  _tensor_constant21 = None
    where_20: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_10, lift_fresh_copy_21, view_242);  lift_fresh_copy_21 = view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_10: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_20, [-1], True)
    sub_31: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_20, amax_10);  where_20 = amax_10 = None
    exp_10: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_10, scalar_tensor_10, div_21);  bitwise_not_10 = scalar_tensor_10 = div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_244: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_21, [-1, 512, 512]);  where_21 = None
    bmm_21: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_244, view_241);  view_244 = view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_245: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_21, [-1, 24, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_117: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_43: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_246: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_43, [1, 512, -1]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_247: "f32[512, 1536]" = torch.ops.aten.view.default(view_246, [512, 1536]);  view_246 = None
    permute_118: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_63: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg171_1, view_247, permute_118);  arg171_1 = view_247 = permute_118 = None
    view_248: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_63, [1, 512, 1536]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_248, add_72);  view_248 = add_72 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-07);  getitem_42 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_32: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_73, getitem_43);  add_73 = getitem_43 = None
    mul_85: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_86: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_85, arg172_1);  mul_85 = arg172_1 = None
    add_75: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_86, arg173_1);  mul_86 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_249: "f32[512, 1536]" = torch.ops.aten.view.default(add_75, [512, 1536])
    permute_119: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_64: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg175_1, view_249, permute_119);  arg175_1 = view_249 = permute_119 = None
    view_250: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_64, [1, 512, 6144]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_250, 0.5)
    mul_88: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_250, 0.7071067811865476);  view_250 = None
    erf_10: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_76: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_89: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_87, add_76);  mul_87 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_251: "f32[512, 6144]" = torch.ops.aten.view.default(mul_89, [512, 6144]);  mul_89 = None
    permute_120: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_65: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg177_1, view_251, permute_120);  arg177_1 = view_251 = permute_120 = None
    view_252: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_65, [1, 512, 1536]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_252, add_75);  view_252 = add_75 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-07);  getitem_44 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_33: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_77, getitem_45);  add_77 = getitem_45 = None
    mul_90: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_91: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_90, arg178_1);  mul_90 = arg178_1 = None
    add_79: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_91, arg179_1);  mul_91 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_253: "f32[512, 1536]" = torch.ops.aten.view.default(add_79, [512, 1536])
    permute_121: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_66: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg181_1, view_253, permute_121);  arg181_1 = view_253 = permute_121 = None
    view_254: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_66, [1, 512, 1536]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_255: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_254, [1, 512, 24, -1]);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_122: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    clone_44: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_256: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_44, [-1, 512, 64]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_257: "f32[512, 1536]" = torch.ops.aten.view.default(add_79, [512, 1536])
    permute_123: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_67: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg183_1, view_257, permute_123);  arg183_1 = view_257 = permute_123 = None
    view_258: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_67, [1, 512, 1536]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_259: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_258, [1, 512, 24, -1]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_124: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    clone_45: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    view_260: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_45, [-1, 512, 64]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_261: "f32[512, 1536]" = torch.ops.aten.view.default(add_79, [512, 1536])
    permute_125: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    addmm_68: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg185_1, view_261, permute_125);  arg185_1 = view_261 = permute_125 = None
    view_262: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_68, [1, 512, 1536]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_263: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_262, [1, 512, 24, -1]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_126: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
    clone_46: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    view_264: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_46, [-1, 512, 64]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant22 = self._tensor_constant22
    lift_fresh_copy_22: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
    mul_92: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_22, 1);  lift_fresh_copy_22 = None
    sqrt_11: "f32[]" = torch.ops.aten.sqrt.default(mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_127: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
    div_22: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_127, sqrt_11);  permute_127 = sqrt_11 = None
    bmm_22: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_256, div_22);  view_256 = div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_265: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_22, [-1, 24, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_11: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_11: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_11);  convert_element_type_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant23 = self._tensor_constant23
    lift_fresh_copy_23: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
    where_22: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_11, lift_fresh_copy_23, view_265);  lift_fresh_copy_23 = view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_11: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_22, [-1], True)
    sub_34: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_22, amax_11);  where_22 = amax_11 = None
    exp_11: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_11, scalar_tensor_11, div_23);  bitwise_not_11 = scalar_tensor_11 = div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_267: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_23, [-1, 512, 512]);  where_23 = None
    bmm_23: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_267, view_264);  view_267 = view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_268: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_23, [-1, 24, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_128: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_47: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_269: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_47, [1, 512, -1]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_270: "f32[512, 1536]" = torch.ops.aten.view.default(view_269, [512, 1536]);  view_269 = None
    permute_129: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_69: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg187_1, view_270, permute_129);  arg187_1 = view_270 = permute_129 = None
    view_271: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_69, [1, 512, 1536]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_80: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_271, add_79);  view_271 = add_79 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_81: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-07);  getitem_46 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_35: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_80, getitem_47);  add_80 = getitem_47 = None
    mul_93: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_94: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_93, arg188_1);  mul_93 = arg188_1 = None
    add_82: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_94, arg189_1);  mul_94 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_272: "f32[512, 1536]" = torch.ops.aten.view.default(add_82, [512, 1536])
    permute_130: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_70: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg191_1, view_272, permute_130);  arg191_1 = view_272 = permute_130 = None
    view_273: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_70, [1, 512, 6144]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_95: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_273, 0.5)
    mul_96: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_273, 0.7071067811865476);  view_273 = None
    erf_11: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_83: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_97: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_95, add_83);  mul_95 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_274: "f32[512, 6144]" = torch.ops.aten.view.default(mul_97, [512, 6144]);  mul_97 = None
    permute_131: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_71: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg193_1, view_274, permute_131);  arg193_1 = view_274 = permute_131 = None
    view_275: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_71, [1, 512, 1536]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_84: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_275, add_82);  view_275 = add_82 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_85: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-07);  getitem_48 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_36: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_84, getitem_49);  add_84 = getitem_49 = None
    mul_98: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_99: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_98, arg194_1);  mul_98 = arg194_1 = None
    add_86: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_99, arg195_1);  mul_99 = arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_276: "f32[512, 1536]" = torch.ops.aten.view.default(add_86, [512, 1536])
    permute_132: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    addmm_72: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg197_1, view_276, permute_132);  arg197_1 = view_276 = permute_132 = None
    view_277: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_72, [1, 512, 1536]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_278: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_277, [1, 512, 24, -1]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_133: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    clone_48: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_279: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_48, [-1, 512, 64]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_280: "f32[512, 1536]" = torch.ops.aten.view.default(add_86, [512, 1536])
    permute_134: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    addmm_73: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg199_1, view_280, permute_134);  arg199_1 = view_280 = permute_134 = None
    view_281: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_73, [1, 512, 1536]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_282: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_281, [1, 512, 24, -1]);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_135: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    clone_49: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    view_283: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_49, [-1, 512, 64]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_284: "f32[512, 1536]" = torch.ops.aten.view.default(add_86, [512, 1536])
    permute_136: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
    addmm_74: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg201_1, view_284, permute_136);  arg201_1 = view_284 = permute_136 = None
    view_285: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_74, [1, 512, 1536]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_286: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_285, [1, 512, 24, -1]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_137: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    clone_50: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    view_287: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_50, [-1, 512, 64]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant24 = self._tensor_constant24
    lift_fresh_copy_24: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant24);  _tensor_constant24 = None
    mul_100: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_24, 1);  lift_fresh_copy_24 = None
    sqrt_12: "f32[]" = torch.ops.aten.sqrt.default(mul_100);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_138: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
    div_24: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_138, sqrt_12);  permute_138 = sqrt_12 = None
    bmm_24: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_279, div_24);  view_279 = div_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_288: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_24, [-1, 24, 512, 512]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_12: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_12: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_12);  convert_element_type_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant25 = self._tensor_constant25
    lift_fresh_copy_25: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant25);  _tensor_constant25 = None
    where_24: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_12, lift_fresh_copy_25, view_288);  lift_fresh_copy_25 = view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_12: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_24, [-1], True)
    sub_37: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_24, amax_12);  where_24 = amax_12 = None
    exp_12: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_13: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_25: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_12, scalar_tensor_12, div_25);  bitwise_not_12 = scalar_tensor_12 = div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_290: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_25, [-1, 512, 512]);  where_25 = None
    bmm_25: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_290, view_287);  view_290 = view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_291: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_25, [-1, 24, 512, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_139: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_51: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_292: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_51, [1, 512, -1]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_293: "f32[512, 1536]" = torch.ops.aten.view.default(view_292, [512, 1536]);  view_292 = None
    permute_140: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
    addmm_75: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg203_1, view_293, permute_140);  arg203_1 = view_293 = permute_140 = None
    view_294: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_75, [1, 512, 1536]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_87: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_294, add_86);  view_294 = add_86 = None
    var_mean_25 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_88: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-07);  getitem_50 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_38: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_87, getitem_51);  add_87 = getitem_51 = None
    mul_101: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
    mul_102: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_101, arg204_1);  mul_101 = arg204_1 = None
    add_89: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_102, arg205_1);  mul_102 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_295: "f32[512, 1536]" = torch.ops.aten.view.default(add_89, [512, 1536])
    permute_141: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    addmm_76: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg207_1, view_295, permute_141);  arg207_1 = view_295 = permute_141 = None
    view_296: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_76, [1, 512, 6144]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_103: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_296, 0.5)
    mul_104: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_296, 0.7071067811865476);  view_296 = None
    erf_12: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
    add_90: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_105: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_103, add_90);  mul_103 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_297: "f32[512, 6144]" = torch.ops.aten.view.default(mul_105, [512, 6144]);  mul_105 = None
    permute_142: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    addmm_77: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg209_1, view_297, permute_142);  arg209_1 = view_297 = permute_142 = None
    view_298: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_77, [1, 512, 1536]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_91: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_298, add_89);  view_298 = add_89 = None
    var_mean_26 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_26[1];  var_mean_26 = None
    add_92: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-07);  getitem_52 = None
    rsqrt_26: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_39: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_91, getitem_53);  add_91 = getitem_53 = None
    mul_106: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
    mul_107: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_106, arg210_1);  mul_106 = arg210_1 = None
    add_93: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_107, arg211_1);  mul_107 = arg211_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_299: "f32[512, 1536]" = torch.ops.aten.view.default(add_93, [512, 1536])
    permute_143: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
    addmm_78: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg213_1, view_299, permute_143);  arg213_1 = view_299 = permute_143 = None
    view_300: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_78, [1, 512, 1536]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_301: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_300, [1, 512, 24, -1]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_144: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    clone_52: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    view_302: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_52, [-1, 512, 64]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_303: "f32[512, 1536]" = torch.ops.aten.view.default(add_93, [512, 1536])
    permute_145: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    addmm_79: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg215_1, view_303, permute_145);  arg215_1 = view_303 = permute_145 = None
    view_304: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_79, [1, 512, 1536]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_305: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_304, [1, 512, 24, -1]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_146: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
    clone_53: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    view_306: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_53, [-1, 512, 64]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_307: "f32[512, 1536]" = torch.ops.aten.view.default(add_93, [512, 1536])
    permute_147: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
    addmm_80: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg217_1, view_307, permute_147);  arg217_1 = view_307 = permute_147 = None
    view_308: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_80, [1, 512, 1536]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_309: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_308, [1, 512, 24, -1]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_148: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
    clone_54: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    view_310: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_54, [-1, 512, 64]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant26 = self._tensor_constant26
    lift_fresh_copy_26: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant26);  _tensor_constant26 = None
    mul_108: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_26, 1);  lift_fresh_copy_26 = None
    sqrt_13: "f32[]" = torch.ops.aten.sqrt.default(mul_108);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_149: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None
    div_26: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_149, sqrt_13);  permute_149 = sqrt_13 = None
    bmm_26: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_302, div_26);  view_302 = div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_311: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_26, [-1, 24, 512, 512]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_13: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_13: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_13);  convert_element_type_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant27 = self._tensor_constant27
    lift_fresh_copy_27: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant27);  _tensor_constant27 = None
    where_26: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_13, lift_fresh_copy_27, view_311);  lift_fresh_copy_27 = view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_13: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_26, [-1], True)
    sub_40: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_26, amax_13);  where_26 = amax_13 = None
    exp_13: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_14: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_27: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_13, scalar_tensor_13, div_27);  bitwise_not_13 = scalar_tensor_13 = div_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_313: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_27, [-1, 512, 512]);  where_27 = None
    bmm_27: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_313, view_310);  view_313 = view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_314: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_27, [-1, 24, 512, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_150: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_55: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_315: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_55, [1, 512, -1]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_316: "f32[512, 1536]" = torch.ops.aten.view.default(view_315, [512, 1536]);  view_315 = None
    permute_151: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
    addmm_81: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg219_1, view_316, permute_151);  arg219_1 = view_316 = permute_151 = None
    view_317: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_81, [1, 512, 1536]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_94: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_317, add_93);  view_317 = add_93 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 512, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 512, 1]" = var_mean_27[1];  var_mean_27 = None
    add_95: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-07);  getitem_54 = None
    rsqrt_27: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_41: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_94, getitem_55);  add_94 = getitem_55 = None
    mul_109: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
    mul_110: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_109, arg220_1);  mul_109 = arg220_1 = None
    add_96: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_110, arg221_1);  mul_110 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_318: "f32[512, 1536]" = torch.ops.aten.view.default(add_96, [512, 1536])
    permute_152: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
    addmm_82: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg223_1, view_318, permute_152);  arg223_1 = view_318 = permute_152 = None
    view_319: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_82, [1, 512, 6144]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_111: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_319, 0.5)
    mul_112: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_319, 0.7071067811865476);  view_319 = None
    erf_13: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
    add_97: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_113: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_111, add_97);  mul_111 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_320: "f32[512, 6144]" = torch.ops.aten.view.default(mul_113, [512, 6144]);  mul_113 = None
    permute_153: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    addmm_83: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg225_1, view_320, permute_153);  arg225_1 = view_320 = permute_153 = None
    view_321: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_83, [1, 512, 1536]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_98: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_321, add_96);  view_321 = add_96 = None
    var_mean_28 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 512, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 512, 1]" = var_mean_28[1];  var_mean_28 = None
    add_99: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-07);  getitem_56 = None
    rsqrt_28: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_42: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_98, getitem_57);  add_98 = getitem_57 = None
    mul_114: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
    mul_115: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_114, arg226_1);  mul_114 = arg226_1 = None
    add_100: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_115, arg227_1);  mul_115 = arg227_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_322: "f32[512, 1536]" = torch.ops.aten.view.default(add_100, [512, 1536])
    permute_154: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    addmm_84: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg229_1, view_322, permute_154);  arg229_1 = view_322 = permute_154 = None
    view_323: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_84, [1, 512, 1536]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_324: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_323, [1, 512, 24, -1]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_155: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    clone_56: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    view_325: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_56, [-1, 512, 64]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_326: "f32[512, 1536]" = torch.ops.aten.view.default(add_100, [512, 1536])
    permute_156: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    addmm_85: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg231_1, view_326, permute_156);  arg231_1 = view_326 = permute_156 = None
    view_327: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_85, [1, 512, 1536]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_328: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_327, [1, 512, 24, -1]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_157: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
    clone_57: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    view_329: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_57, [-1, 512, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_330: "f32[512, 1536]" = torch.ops.aten.view.default(add_100, [512, 1536])
    permute_158: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
    addmm_86: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg233_1, view_330, permute_158);  arg233_1 = view_330 = permute_158 = None
    view_331: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_86, [1, 512, 1536]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_332: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_331, [1, 512, 24, -1]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_159: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
    clone_58: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    view_333: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_58, [-1, 512, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant28 = self._tensor_constant28
    lift_fresh_copy_28: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant28);  _tensor_constant28 = None
    mul_116: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_28, 1);  lift_fresh_copy_28 = None
    sqrt_14: "f32[]" = torch.ops.aten.sqrt.default(mul_116);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_160: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_329, [0, 2, 1]);  view_329 = None
    div_28: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_160, sqrt_14);  permute_160 = sqrt_14 = None
    bmm_28: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_325, div_28);  view_325 = div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_334: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_28, [-1, 24, 512, 512]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_14: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_14: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_14);  convert_element_type_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant29 = self._tensor_constant29
    lift_fresh_copy_29: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant29);  _tensor_constant29 = None
    where_28: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_14, lift_fresh_copy_29, view_334);  lift_fresh_copy_29 = view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_14: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_28, [-1], True)
    sub_43: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_28, amax_14);  where_28 = amax_14 = None
    exp_14: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_29: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_14, scalar_tensor_14, div_29);  bitwise_not_14 = scalar_tensor_14 = div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_336: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_29, [-1, 512, 512]);  where_29 = None
    bmm_29: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_336, view_333);  view_336 = view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_337: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_29, [-1, 24, 512, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_161: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_59: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_338: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_59, [1, 512, -1]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_339: "f32[512, 1536]" = torch.ops.aten.view.default(view_338, [512, 1536]);  view_338 = None
    permute_162: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    addmm_87: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg235_1, view_339, permute_162);  arg235_1 = view_339 = permute_162 = None
    view_340: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_87, [1, 512, 1536]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_101: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_340, add_100);  view_340 = add_100 = None
    var_mean_29 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_29[1];  var_mean_29 = None
    add_102: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-07);  getitem_58 = None
    rsqrt_29: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_44: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_101, getitem_59);  add_101 = getitem_59 = None
    mul_117: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
    mul_118: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_117, arg236_1);  mul_117 = arg236_1 = None
    add_103: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_118, arg237_1);  mul_118 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_341: "f32[512, 1536]" = torch.ops.aten.view.default(add_103, [512, 1536])
    permute_163: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    addmm_88: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg239_1, view_341, permute_163);  arg239_1 = view_341 = permute_163 = None
    view_342: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_88, [1, 512, 6144]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_119: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_342, 0.5)
    mul_120: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_342, 0.7071067811865476);  view_342 = None
    erf_14: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
    add_104: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_121: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_119, add_104);  mul_119 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_343: "f32[512, 6144]" = torch.ops.aten.view.default(mul_121, [512, 6144]);  mul_121 = None
    permute_164: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    addmm_89: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg241_1, view_343, permute_164);  arg241_1 = view_343 = permute_164 = None
    view_344: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_89, [1, 512, 1536]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_105: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_344, add_103);  view_344 = add_103 = None
    var_mean_30 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 512, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 512, 1]" = var_mean_30[1];  var_mean_30 = None
    add_106: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-07);  getitem_60 = None
    rsqrt_30: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_45: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_105, getitem_61);  add_105 = getitem_61 = None
    mul_122: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
    mul_123: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_122, arg242_1);  mul_122 = arg242_1 = None
    add_107: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_123, arg243_1);  mul_123 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_345: "f32[512, 1536]" = torch.ops.aten.view.default(add_107, [512, 1536])
    permute_165: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    addmm_90: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg245_1, view_345, permute_165);  arg245_1 = view_345 = permute_165 = None
    view_346: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_90, [1, 512, 1536]);  addmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_347: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_346, [1, 512, 24, -1]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_166: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
    clone_60: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    view_348: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_60, [-1, 512, 64]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_349: "f32[512, 1536]" = torch.ops.aten.view.default(add_107, [512, 1536])
    permute_167: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
    addmm_91: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg247_1, view_349, permute_167);  arg247_1 = view_349 = permute_167 = None
    view_350: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_91, [1, 512, 1536]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_351: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_350, [1, 512, 24, -1]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_168: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
    clone_61: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    view_352: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_61, [-1, 512, 64]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_353: "f32[512, 1536]" = torch.ops.aten.view.default(add_107, [512, 1536])
    permute_169: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
    addmm_92: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg249_1, view_353, permute_169);  arg249_1 = view_353 = permute_169 = None
    view_354: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_92, [1, 512, 1536]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_355: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_354, [1, 512, 24, -1]);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_170: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_355, [0, 2, 1, 3]);  view_355 = None
    clone_62: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    view_356: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_62, [-1, 512, 64]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant30 = self._tensor_constant30
    lift_fresh_copy_30: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant30);  _tensor_constant30 = None
    mul_124: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_30, 1);  lift_fresh_copy_30 = None
    sqrt_15: "f32[]" = torch.ops.aten.sqrt.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_171: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
    div_30: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_171, sqrt_15);  permute_171 = sqrt_15 = None
    bmm_30: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_348, div_30);  view_348 = div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_357: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_30, [-1, 24, 512, 512]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_15: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_15: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_15);  convert_element_type_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant31 = self._tensor_constant31
    lift_fresh_copy_31: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant31);  _tensor_constant31 = None
    where_30: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_15, lift_fresh_copy_31, view_357);  lift_fresh_copy_31 = view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_15: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_30, [-1], True)
    sub_46: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_30, amax_15);  where_30 = amax_15 = None
    exp_15: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_16: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_31: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_15, scalar_tensor_15, div_31);  bitwise_not_15 = scalar_tensor_15 = div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_359: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_31, [-1, 512, 512]);  where_31 = None
    bmm_31: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_359, view_356);  view_359 = view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_360: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_31, [-1, 24, 512, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_172: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_63: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_361: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_63, [1, 512, -1]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_362: "f32[512, 1536]" = torch.ops.aten.view.default(view_361, [512, 1536]);  view_361 = None
    permute_173: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    addmm_93: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg251_1, view_362, permute_173);  arg251_1 = view_362 = permute_173 = None
    view_363: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_93, [1, 512, 1536]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_108: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_363, add_107);  view_363 = add_107 = None
    var_mean_31 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_31[1];  var_mean_31 = None
    add_109: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-07);  getitem_62 = None
    rsqrt_31: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_47: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_108, getitem_63);  add_108 = getitem_63 = None
    mul_125: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
    mul_126: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_125, arg252_1);  mul_125 = arg252_1 = None
    add_110: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_126, arg253_1);  mul_126 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_364: "f32[512, 1536]" = torch.ops.aten.view.default(add_110, [512, 1536])
    permute_174: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_94: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg255_1, view_364, permute_174);  arg255_1 = view_364 = permute_174 = None
    view_365: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_94, [1, 512, 6144]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_127: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_365, 0.5)
    mul_128: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_365, 0.7071067811865476);  view_365 = None
    erf_15: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_111: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_129: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_127, add_111);  mul_127 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_366: "f32[512, 6144]" = torch.ops.aten.view.default(mul_129, [512, 6144]);  mul_129 = None
    permute_175: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
    addmm_95: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg257_1, view_366, permute_175);  arg257_1 = view_366 = permute_175 = None
    view_367: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_95, [1, 512, 1536]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_112: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_367, add_110);  view_367 = add_110 = None
    var_mean_32 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 512, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 512, 1]" = var_mean_32[1];  var_mean_32 = None
    add_113: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-07);  getitem_64 = None
    rsqrt_32: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_48: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_112, getitem_65);  add_112 = getitem_65 = None
    mul_130: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
    mul_131: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_130, arg258_1);  mul_130 = arg258_1 = None
    add_114: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_131, arg259_1);  mul_131 = arg259_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_368: "f32[512, 1536]" = torch.ops.aten.view.default(add_114, [512, 1536])
    permute_176: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    addmm_96: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg261_1, view_368, permute_176);  arg261_1 = view_368 = permute_176 = None
    view_369: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_96, [1, 512, 1536]);  addmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_370: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_369, [1, 512, 24, -1]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_177: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    clone_64: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    view_371: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_64, [-1, 512, 64]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_372: "f32[512, 1536]" = torch.ops.aten.view.default(add_114, [512, 1536])
    permute_178: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
    addmm_97: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg263_1, view_372, permute_178);  arg263_1 = view_372 = permute_178 = None
    view_373: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_97, [1, 512, 1536]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_374: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_373, [1, 512, 24, -1]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_179: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    clone_65: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_375: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_65, [-1, 512, 64]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_376: "f32[512, 1536]" = torch.ops.aten.view.default(add_114, [512, 1536])
    permute_180: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
    addmm_98: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg265_1, view_376, permute_180);  arg265_1 = view_376 = permute_180 = None
    view_377: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_98, [1, 512, 1536]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_378: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_377, [1, 512, 24, -1]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_181: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
    clone_66: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    view_379: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_66, [-1, 512, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant32 = self._tensor_constant32
    lift_fresh_copy_32: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant32);  _tensor_constant32 = None
    mul_132: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_32, 1);  lift_fresh_copy_32 = None
    sqrt_16: "f32[]" = torch.ops.aten.sqrt.default(mul_132);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_182: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
    div_32: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_182, sqrt_16);  permute_182 = sqrt_16 = None
    bmm_32: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_371, div_32);  view_371 = div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_380: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_32, [-1, 24, 512, 512]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_16: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_16: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_16);  convert_element_type_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant33 = self._tensor_constant33
    lift_fresh_copy_33: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant33);  _tensor_constant33 = None
    where_32: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_16, lift_fresh_copy_33, view_380);  lift_fresh_copy_33 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_16: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_32, [-1], True)
    sub_49: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_32, amax_16);  where_32 = amax_16 = None
    exp_16: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_17: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_33: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_16, scalar_tensor_16, div_33);  bitwise_not_16 = scalar_tensor_16 = div_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_382: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_33, [-1, 512, 512]);  where_33 = None
    bmm_33: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_382, view_379);  view_382 = view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_383: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_33, [-1, 24, 512, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_183: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_67: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_384: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_67, [1, 512, -1]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_385: "f32[512, 1536]" = torch.ops.aten.view.default(view_384, [512, 1536]);  view_384 = None
    permute_184: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
    addmm_99: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg267_1, view_385, permute_184);  arg267_1 = view_385 = permute_184 = None
    view_386: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_99, [1, 512, 1536]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_115: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_386, add_114);  view_386 = add_114 = None
    var_mean_33 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 512, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 512, 1]" = var_mean_33[1];  var_mean_33 = None
    add_116: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-07);  getitem_66 = None
    rsqrt_33: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_50: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_115, getitem_67);  add_115 = getitem_67 = None
    mul_133: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
    mul_134: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_133, arg268_1);  mul_133 = arg268_1 = None
    add_117: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_134, arg269_1);  mul_134 = arg269_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_387: "f32[512, 1536]" = torch.ops.aten.view.default(add_117, [512, 1536])
    permute_185: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg270_1, [1, 0]);  arg270_1 = None
    addmm_100: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg271_1, view_387, permute_185);  arg271_1 = view_387 = permute_185 = None
    view_388: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_100, [1, 512, 6144]);  addmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_135: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_388, 0.5)
    mul_136: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_388, 0.7071067811865476);  view_388 = None
    erf_16: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_136);  mul_136 = None
    add_118: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_137: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_135, add_118);  mul_135 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[512, 6144]" = torch.ops.aten.view.default(mul_137, [512, 6144]);  mul_137 = None
    permute_186: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
    addmm_101: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg273_1, view_389, permute_186);  arg273_1 = view_389 = permute_186 = None
    view_390: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_101, [1, 512, 1536]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_119: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_390, add_117);  view_390 = add_117 = None
    var_mean_34 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_34[1];  var_mean_34 = None
    add_120: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-07);  getitem_68 = None
    rsqrt_34: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_51: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_119, getitem_69);  add_119 = getitem_69 = None
    mul_138: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
    mul_139: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_138, arg274_1);  mul_138 = arg274_1 = None
    add_121: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_139, arg275_1);  mul_139 = arg275_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_391: "f32[512, 1536]" = torch.ops.aten.view.default(add_121, [512, 1536])
    permute_187: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
    addmm_102: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg277_1, view_391, permute_187);  arg277_1 = view_391 = permute_187 = None
    view_392: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_102, [1, 512, 1536]);  addmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_393: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_392, [1, 512, 24, -1]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_188: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    clone_68: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_394: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_68, [-1, 512, 64]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_395: "f32[512, 1536]" = torch.ops.aten.view.default(add_121, [512, 1536])
    permute_189: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
    addmm_103: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg279_1, view_395, permute_189);  arg279_1 = view_395 = permute_189 = None
    view_396: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_103, [1, 512, 1536]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_397: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_396, [1, 512, 24, -1]);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_190: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    clone_69: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
    view_398: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_69, [-1, 512, 64]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_399: "f32[512, 1536]" = torch.ops.aten.view.default(add_121, [512, 1536])
    permute_191: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
    addmm_104: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg281_1, view_399, permute_191);  arg281_1 = view_399 = permute_191 = None
    view_400: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_104, [1, 512, 1536]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_401: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_400, [1, 512, 24, -1]);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_192: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None
    clone_70: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
    view_402: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_70, [-1, 512, 64]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant34 = self._tensor_constant34
    lift_fresh_copy_34: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant34);  _tensor_constant34 = None
    mul_140: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_34, 1);  lift_fresh_copy_34 = None
    sqrt_17: "f32[]" = torch.ops.aten.sqrt.default(mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_193: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None
    div_34: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_193, sqrt_17);  permute_193 = sqrt_17 = None
    bmm_34: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_394, div_34);  view_394 = div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_403: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_34, [-1, 24, 512, 512]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_17: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_17: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_17);  convert_element_type_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant35 = self._tensor_constant35
    lift_fresh_copy_35: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant35);  _tensor_constant35 = None
    where_34: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_17, lift_fresh_copy_35, view_403);  lift_fresh_copy_35 = view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_17: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_34, [-1], True)
    sub_52: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_34, amax_17);  where_34 = amax_17 = None
    exp_17: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_18: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_35: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_17, scalar_tensor_17, div_35);  bitwise_not_17 = scalar_tensor_17 = div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_405: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_35, [-1, 512, 512]);  where_35 = None
    bmm_35: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_405, view_402);  view_405 = view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_406: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_35, [-1, 24, 512, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_194: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_71: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_407: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_71, [1, 512, -1]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_408: "f32[512, 1536]" = torch.ops.aten.view.default(view_407, [512, 1536]);  view_407 = None
    permute_195: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
    addmm_105: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg283_1, view_408, permute_195);  arg283_1 = view_408 = permute_195 = None
    view_409: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_105, [1, 512, 1536]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_122: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_409, add_121);  view_409 = add_121 = None
    var_mean_35 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 512, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 512, 1]" = var_mean_35[1];  var_mean_35 = None
    add_123: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-07);  getitem_70 = None
    rsqrt_35: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_53: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_122, getitem_71);  add_122 = getitem_71 = None
    mul_141: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
    mul_142: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_141, arg284_1);  mul_141 = arg284_1 = None
    add_124: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_142, arg285_1);  mul_142 = arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_410: "f32[512, 1536]" = torch.ops.aten.view.default(add_124, [512, 1536])
    permute_196: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
    addmm_106: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg287_1, view_410, permute_196);  arg287_1 = view_410 = permute_196 = None
    view_411: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_106, [1, 512, 6144]);  addmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_143: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_411, 0.5)
    mul_144: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_411, 0.7071067811865476);  view_411 = None
    erf_17: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
    add_125: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_145: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_143, add_125);  mul_143 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_412: "f32[512, 6144]" = torch.ops.aten.view.default(mul_145, [512, 6144]);  mul_145 = None
    permute_197: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
    addmm_107: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg289_1, view_412, permute_197);  arg289_1 = view_412 = permute_197 = None
    view_413: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_107, [1, 512, 1536]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_126: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_413, add_124);  view_413 = add_124 = None
    var_mean_36 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_36[1];  var_mean_36 = None
    add_127: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-07);  getitem_72 = None
    rsqrt_36: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_54: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_126, getitem_73);  add_126 = getitem_73 = None
    mul_146: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
    mul_147: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_146, arg290_1);  mul_146 = arg290_1 = None
    add_128: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_147, arg291_1);  mul_147 = arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_414: "f32[512, 1536]" = torch.ops.aten.view.default(add_128, [512, 1536])
    permute_198: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
    addmm_108: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg293_1, view_414, permute_198);  arg293_1 = view_414 = permute_198 = None
    view_415: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_108, [1, 512, 1536]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_416: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_415, [1, 512, 24, -1]);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_199: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
    clone_72: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_417: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_72, [-1, 512, 64]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_418: "f32[512, 1536]" = torch.ops.aten.view.default(add_128, [512, 1536])
    permute_200: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
    addmm_109: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg295_1, view_418, permute_200);  arg295_1 = view_418 = permute_200 = None
    view_419: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_109, [1, 512, 1536]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_420: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_419, [1, 512, 24, -1]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_201: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    clone_73: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
    view_421: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_73, [-1, 512, 64]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_422: "f32[512, 1536]" = torch.ops.aten.view.default(add_128, [512, 1536])
    permute_202: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
    addmm_110: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg297_1, view_422, permute_202);  arg297_1 = view_422 = permute_202 = None
    view_423: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_110, [1, 512, 1536]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_424: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_423, [1, 512, 24, -1]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_203: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
    clone_74: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    view_425: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_74, [-1, 512, 64]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant36 = self._tensor_constant36
    lift_fresh_copy_36: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant36);  _tensor_constant36 = None
    mul_148: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_36, 1);  lift_fresh_copy_36 = None
    sqrt_18: "f32[]" = torch.ops.aten.sqrt.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_204: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
    div_36: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_204, sqrt_18);  permute_204 = sqrt_18 = None
    bmm_36: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_417, div_36);  view_417 = div_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_426: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_36, [-1, 24, 512, 512]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_18: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_18: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_18);  convert_element_type_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant37 = self._tensor_constant37
    lift_fresh_copy_37: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant37);  _tensor_constant37 = None
    where_36: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_18, lift_fresh_copy_37, view_426);  lift_fresh_copy_37 = view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_18: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_36, [-1], True)
    sub_55: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_36, amax_18);  where_36 = amax_18 = None
    exp_18: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_19: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_37: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_18, scalar_tensor_18, div_37);  bitwise_not_18 = scalar_tensor_18 = div_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_428: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_37, [-1, 512, 512]);  where_37 = None
    bmm_37: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_428, view_425);  view_428 = view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_429: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_37, [-1, 24, 512, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_205: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_429, [0, 2, 1, 3]);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_75: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_430: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_75, [1, 512, -1]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_431: "f32[512, 1536]" = torch.ops.aten.view.default(view_430, [512, 1536]);  view_430 = None
    permute_206: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
    addmm_111: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg299_1, view_431, permute_206);  arg299_1 = view_431 = permute_206 = None
    view_432: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_111, [1, 512, 1536]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_129: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_432, add_128);  view_432 = add_128 = None
    var_mean_37 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 512, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 512, 1]" = var_mean_37[1];  var_mean_37 = None
    add_130: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-07);  getitem_74 = None
    rsqrt_37: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_56: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_129, getitem_75);  add_129 = getitem_75 = None
    mul_149: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
    mul_150: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_149, arg300_1);  mul_149 = arg300_1 = None
    add_131: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_150, arg301_1);  mul_150 = arg301_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_433: "f32[512, 1536]" = torch.ops.aten.view.default(add_131, [512, 1536])
    permute_207: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
    addmm_112: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg303_1, view_433, permute_207);  arg303_1 = view_433 = permute_207 = None
    view_434: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_112, [1, 512, 6144]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_151: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_434, 0.5)
    mul_152: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_434, 0.7071067811865476);  view_434 = None
    erf_18: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
    add_132: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_153: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_151, add_132);  mul_151 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_435: "f32[512, 6144]" = torch.ops.aten.view.default(mul_153, [512, 6144]);  mul_153 = None
    permute_208: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
    addmm_113: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg305_1, view_435, permute_208);  arg305_1 = view_435 = permute_208 = None
    view_436: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_113, [1, 512, 1536]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_133: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_436, add_131);  view_436 = add_131 = None
    var_mean_38 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 512, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 512, 1]" = var_mean_38[1];  var_mean_38 = None
    add_134: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-07);  getitem_76 = None
    rsqrt_38: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_57: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_133, getitem_77);  add_133 = getitem_77 = None
    mul_154: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
    mul_155: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_154, arg306_1);  mul_154 = arg306_1 = None
    add_135: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_155, arg307_1);  mul_155 = arg307_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_437: "f32[512, 1536]" = torch.ops.aten.view.default(add_135, [512, 1536])
    permute_209: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
    addmm_114: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg309_1, view_437, permute_209);  arg309_1 = view_437 = permute_209 = None
    view_438: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_114, [1, 512, 1536]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_439: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_438, [1, 512, 24, -1]);  view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_210: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_439, [0, 2, 1, 3]);  view_439 = None
    clone_76: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    view_440: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_76, [-1, 512, 64]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_441: "f32[512, 1536]" = torch.ops.aten.view.default(add_135, [512, 1536])
    permute_211: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
    addmm_115: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg311_1, view_441, permute_211);  arg311_1 = view_441 = permute_211 = None
    view_442: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_115, [1, 512, 1536]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_443: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_442, [1, 512, 24, -1]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_212: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    clone_77: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
    view_444: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_77, [-1, 512, 64]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_445: "f32[512, 1536]" = torch.ops.aten.view.default(add_135, [512, 1536])
    permute_213: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
    addmm_116: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg313_1, view_445, permute_213);  arg313_1 = view_445 = permute_213 = None
    view_446: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_116, [1, 512, 1536]);  addmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_447: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_446, [1, 512, 24, -1]);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_214: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
    clone_78: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    view_448: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_78, [-1, 512, 64]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant38 = self._tensor_constant38
    lift_fresh_copy_38: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant38);  _tensor_constant38 = None
    mul_156: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_38, 1);  lift_fresh_copy_38 = None
    sqrt_19: "f32[]" = torch.ops.aten.sqrt.default(mul_156);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_215: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
    div_38: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_215, sqrt_19);  permute_215 = sqrt_19 = None
    bmm_38: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_440, div_38);  view_440 = div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_449: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_38, [-1, 24, 512, 512]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_19: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_19: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_19);  convert_element_type_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant39 = self._tensor_constant39
    lift_fresh_copy_39: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant39);  _tensor_constant39 = None
    where_38: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_19, lift_fresh_copy_39, view_449);  lift_fresh_copy_39 = view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_19: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_38, [-1], True)
    sub_58: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_38, amax_19);  where_38 = amax_19 = None
    exp_19: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_20: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_39: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_19, scalar_tensor_19, div_39);  bitwise_not_19 = scalar_tensor_19 = div_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_451: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_39, [-1, 512, 512]);  where_39 = None
    bmm_39: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_451, view_448);  view_451 = view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_452: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_39, [-1, 24, 512, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_216: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_79: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_453: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_79, [1, 512, -1]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_454: "f32[512, 1536]" = torch.ops.aten.view.default(view_453, [512, 1536]);  view_453 = None
    permute_217: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
    addmm_117: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg315_1, view_454, permute_217);  arg315_1 = view_454 = permute_217 = None
    view_455: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_117, [1, 512, 1536]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_136: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_455, add_135);  view_455 = add_135 = None
    var_mean_39 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_39[1];  var_mean_39 = None
    add_137: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-07);  getitem_78 = None
    rsqrt_39: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_59: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_136, getitem_79);  add_136 = getitem_79 = None
    mul_157: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
    mul_158: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_157, arg316_1);  mul_157 = arg316_1 = None
    add_138: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_158, arg317_1);  mul_158 = arg317_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_456: "f32[512, 1536]" = torch.ops.aten.view.default(add_138, [512, 1536])
    permute_218: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
    addmm_118: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg319_1, view_456, permute_218);  arg319_1 = view_456 = permute_218 = None
    view_457: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_118, [1, 512, 6144]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_159: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_457, 0.5)
    mul_160: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_457, 0.7071067811865476);  view_457 = None
    erf_19: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
    add_139: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_161: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_159, add_139);  mul_159 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_458: "f32[512, 6144]" = torch.ops.aten.view.default(mul_161, [512, 6144]);  mul_161 = None
    permute_219: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
    addmm_119: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg321_1, view_458, permute_219);  arg321_1 = view_458 = permute_219 = None
    view_459: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_119, [1, 512, 1536]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_140: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_459, add_138);  view_459 = add_138 = None
    var_mean_40 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 512, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 512, 1]" = var_mean_40[1];  var_mean_40 = None
    add_141: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-07);  getitem_80 = None
    rsqrt_40: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_60: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_140, getitem_81);  add_140 = getitem_81 = None
    mul_162: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
    mul_163: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_162, arg322_1);  mul_162 = arg322_1 = None
    add_142: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_163, arg323_1);  mul_163 = arg323_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_460: "f32[512, 1536]" = torch.ops.aten.view.default(add_142, [512, 1536])
    permute_220: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
    addmm_120: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg325_1, view_460, permute_220);  arg325_1 = view_460 = permute_220 = None
    view_461: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_120, [1, 512, 1536]);  addmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_462: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_461, [1, 512, 24, -1]);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_221: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
    clone_80: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    view_463: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_80, [-1, 512, 64]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_464: "f32[512, 1536]" = torch.ops.aten.view.default(add_142, [512, 1536])
    permute_222: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
    addmm_121: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg327_1, view_464, permute_222);  arg327_1 = view_464 = permute_222 = None
    view_465: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_121, [1, 512, 1536]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_466: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_465, [1, 512, 24, -1]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_223: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
    clone_81: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
    view_467: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_81, [-1, 512, 64]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_468: "f32[512, 1536]" = torch.ops.aten.view.default(add_142, [512, 1536])
    permute_224: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
    addmm_122: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg329_1, view_468, permute_224);  arg329_1 = view_468 = permute_224 = None
    view_469: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_122, [1, 512, 1536]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_470: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_469, [1, 512, 24, -1]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_225: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    clone_82: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    view_471: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_82, [-1, 512, 64]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant40 = self._tensor_constant40
    lift_fresh_copy_40: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant40);  _tensor_constant40 = None
    mul_164: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_40, 1);  lift_fresh_copy_40 = None
    sqrt_20: "f32[]" = torch.ops.aten.sqrt.default(mul_164);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_226: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None
    div_40: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_226, sqrt_20);  permute_226 = sqrt_20 = None
    bmm_40: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_463, div_40);  view_463 = div_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_472: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_40, [-1, 24, 512, 512]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_20: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_20: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_20);  convert_element_type_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant41 = self._tensor_constant41
    lift_fresh_copy_41: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant41);  _tensor_constant41 = None
    where_40: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_20, lift_fresh_copy_41, view_472);  lift_fresh_copy_41 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_20: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_40, [-1], True)
    sub_61: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_40, amax_20);  where_40 = amax_20 = None
    exp_20: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_21: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_41: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_41: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_20, scalar_tensor_20, div_41);  bitwise_not_20 = scalar_tensor_20 = div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_474: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_41, [-1, 512, 512]);  where_41 = None
    bmm_41: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_474, view_471);  view_474 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_475: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_41, [-1, 24, 512, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_227: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_83: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_476: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_83, [1, 512, -1]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_477: "f32[512, 1536]" = torch.ops.aten.view.default(view_476, [512, 1536]);  view_476 = None
    permute_228: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
    addmm_123: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg331_1, view_477, permute_228);  arg331_1 = view_477 = permute_228 = None
    view_478: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_123, [1, 512, 1536]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_143: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_478, add_142);  view_478 = add_142 = None
    var_mean_41 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_41[1];  var_mean_41 = None
    add_144: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-07);  getitem_82 = None
    rsqrt_41: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_62: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_143, getitem_83);  add_143 = getitem_83 = None
    mul_165: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
    mul_166: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_165, arg332_1);  mul_165 = arg332_1 = None
    add_145: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_166, arg333_1);  mul_166 = arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_479: "f32[512, 1536]" = torch.ops.aten.view.default(add_145, [512, 1536])
    permute_229: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg334_1, [1, 0]);  arg334_1 = None
    addmm_124: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg335_1, view_479, permute_229);  arg335_1 = view_479 = permute_229 = None
    view_480: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_124, [1, 512, 6144]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_167: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_480, 0.5)
    mul_168: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_480, 0.7071067811865476);  view_480 = None
    erf_20: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
    add_146: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_169: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_167, add_146);  mul_167 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_481: "f32[512, 6144]" = torch.ops.aten.view.default(mul_169, [512, 6144]);  mul_169 = None
    permute_230: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg336_1, [1, 0]);  arg336_1 = None
    addmm_125: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg337_1, view_481, permute_230);  arg337_1 = view_481 = permute_230 = None
    view_482: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_125, [1, 512, 1536]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_147: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_482, add_145);  view_482 = add_145 = None
    var_mean_42 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 512, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 512, 1]" = var_mean_42[1];  var_mean_42 = None
    add_148: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-07);  getitem_84 = None
    rsqrt_42: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_63: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_147, getitem_85);  add_147 = getitem_85 = None
    mul_170: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
    mul_171: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_170, arg338_1);  mul_170 = arg338_1 = None
    add_149: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_171, arg339_1);  mul_171 = arg339_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_483: "f32[512, 1536]" = torch.ops.aten.view.default(add_149, [512, 1536])
    permute_231: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg340_1, [1, 0]);  arg340_1 = None
    addmm_126: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg341_1, view_483, permute_231);  arg341_1 = view_483 = permute_231 = None
    view_484: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_126, [1, 512, 1536]);  addmm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_485: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_484, [1, 512, 24, -1]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_232: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_485, [0, 2, 1, 3]);  view_485 = None
    clone_84: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
    view_486: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_84, [-1, 512, 64]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_487: "f32[512, 1536]" = torch.ops.aten.view.default(add_149, [512, 1536])
    permute_233: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg342_1, [1, 0]);  arg342_1 = None
    addmm_127: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg343_1, view_487, permute_233);  arg343_1 = view_487 = permute_233 = None
    view_488: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_127, [1, 512, 1536]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_489: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_488, [1, 512, 24, -1]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_234: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
    clone_85: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
    view_490: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_85, [-1, 512, 64]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_491: "f32[512, 1536]" = torch.ops.aten.view.default(add_149, [512, 1536])
    permute_235: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
    addmm_128: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg345_1, view_491, permute_235);  arg345_1 = view_491 = permute_235 = None
    view_492: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_128, [1, 512, 1536]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_493: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_492, [1, 512, 24, -1]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_236: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    clone_86: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    view_494: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_86, [-1, 512, 64]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant42 = self._tensor_constant42
    lift_fresh_copy_42: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant42);  _tensor_constant42 = None
    mul_172: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_42, 1);  lift_fresh_copy_42 = None
    sqrt_21: "f32[]" = torch.ops.aten.sqrt.default(mul_172);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_237: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_490, [0, 2, 1]);  view_490 = None
    div_42: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_237, sqrt_21);  permute_237 = sqrt_21 = None
    bmm_42: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_486, div_42);  view_486 = div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_495: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_42, [-1, 24, 512, 512]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_21: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_21: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_21);  convert_element_type_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant43 = self._tensor_constant43
    lift_fresh_copy_43: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant43);  _tensor_constant43 = None
    where_42: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_21, lift_fresh_copy_43, view_495);  lift_fresh_copy_43 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_21: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_42, [-1], True)
    sub_64: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_42, amax_21);  where_42 = amax_21 = None
    exp_21: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_22: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_43: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_43: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_21, scalar_tensor_21, div_43);  bitwise_not_21 = scalar_tensor_21 = div_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_497: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_43, [-1, 512, 512]);  where_43 = None
    bmm_43: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_497, view_494);  view_497 = view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_498: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_43, [-1, 24, 512, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_238: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_87: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_499: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_87, [1, 512, -1]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_500: "f32[512, 1536]" = torch.ops.aten.view.default(view_499, [512, 1536]);  view_499 = None
    permute_239: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg346_1, [1, 0]);  arg346_1 = None
    addmm_129: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg347_1, view_500, permute_239);  arg347_1 = view_500 = permute_239 = None
    view_501: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_129, [1, 512, 1536]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_150: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_501, add_149);  view_501 = add_149 = None
    var_mean_43 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 512, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 512, 1]" = var_mean_43[1];  var_mean_43 = None
    add_151: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-07);  getitem_86 = None
    rsqrt_43: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_65: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_150, getitem_87);  add_150 = getitem_87 = None
    mul_173: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
    mul_174: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_173, arg348_1);  mul_173 = arg348_1 = None
    add_152: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_174, arg349_1);  mul_174 = arg349_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_502: "f32[512, 1536]" = torch.ops.aten.view.default(add_152, [512, 1536])
    permute_240: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
    addmm_130: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg351_1, view_502, permute_240);  arg351_1 = view_502 = permute_240 = None
    view_503: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_130, [1, 512, 6144]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_175: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
    mul_176: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476);  view_503 = None
    erf_21: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_176);  mul_176 = None
    add_153: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_177: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_175, add_153);  mul_175 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[512, 6144]" = torch.ops.aten.view.default(mul_177, [512, 6144]);  mul_177 = None
    permute_241: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg352_1, [1, 0]);  arg352_1 = None
    addmm_131: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg353_1, view_504, permute_241);  arg353_1 = view_504 = permute_241 = None
    view_505: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_131, [1, 512, 1536]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_154: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_505, add_152);  view_505 = add_152 = None
    var_mean_44 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_44[1];  var_mean_44 = None
    add_155: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-07);  getitem_88 = None
    rsqrt_44: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_66: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_154, getitem_89);  add_154 = getitem_89 = None
    mul_178: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
    mul_179: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_178, arg354_1);  mul_178 = arg354_1 = None
    add_156: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_179, arg355_1);  mul_179 = arg355_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_506: "f32[512, 1536]" = torch.ops.aten.view.default(add_156, [512, 1536])
    permute_242: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
    addmm_132: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg357_1, view_506, permute_242);  arg357_1 = view_506 = permute_242 = None
    view_507: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_132, [1, 512, 1536]);  addmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_508: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_507, [1, 512, 24, -1]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_243: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
    clone_88: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
    view_509: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_88, [-1, 512, 64]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_510: "f32[512, 1536]" = torch.ops.aten.view.default(add_156, [512, 1536])
    permute_244: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
    addmm_133: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg359_1, view_510, permute_244);  arg359_1 = view_510 = permute_244 = None
    view_511: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_133, [1, 512, 1536]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_512: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_511, [1, 512, 24, -1]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_245: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
    clone_89: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
    view_513: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_89, [-1, 512, 64]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_514: "f32[512, 1536]" = torch.ops.aten.view.default(add_156, [512, 1536])
    permute_246: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg360_1, [1, 0]);  arg360_1 = None
    addmm_134: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg361_1, view_514, permute_246);  arg361_1 = view_514 = permute_246 = None
    view_515: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_134, [1, 512, 1536]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_516: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_515, [1, 512, 24, -1]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_247: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
    clone_90: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    view_517: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_90, [-1, 512, 64]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant44 = self._tensor_constant44
    lift_fresh_copy_44: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant44);  _tensor_constant44 = None
    mul_180: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_44, 1);  lift_fresh_copy_44 = None
    sqrt_22: "f32[]" = torch.ops.aten.sqrt.default(mul_180);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_248: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_513, [0, 2, 1]);  view_513 = None
    div_44: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_248, sqrt_22);  permute_248 = sqrt_22 = None
    bmm_44: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_509, div_44);  view_509 = div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_518: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_44, [-1, 24, 512, 512]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_22: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_22: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_22);  convert_element_type_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant45 = self._tensor_constant45
    lift_fresh_copy_45: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant45);  _tensor_constant45 = None
    where_44: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_22, lift_fresh_copy_45, view_518);  lift_fresh_copy_45 = view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_22: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_44, [-1], True)
    sub_67: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_44, amax_22);  where_44 = amax_22 = None
    exp_22: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_23: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_45: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_45: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_22, scalar_tensor_22, div_45);  bitwise_not_22 = scalar_tensor_22 = div_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_520: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_45, [-1, 512, 512]);  where_45 = None
    bmm_45: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_520, view_517);  view_520 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_521: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_45, [-1, 24, 512, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_249: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_521, [0, 2, 1, 3]);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_91: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_522: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_91, [1, 512, -1]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_523: "f32[512, 1536]" = torch.ops.aten.view.default(view_522, [512, 1536]);  view_522 = None
    permute_250: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg362_1, [1, 0]);  arg362_1 = None
    addmm_135: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg363_1, view_523, permute_250);  arg363_1 = view_523 = permute_250 = None
    view_524: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_135, [1, 512, 1536]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_157: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_524, add_156);  view_524 = add_156 = None
    var_mean_45 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 512, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 512, 1]" = var_mean_45[1];  var_mean_45 = None
    add_158: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-07);  getitem_90 = None
    rsqrt_45: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_68: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_157, getitem_91);  add_157 = getitem_91 = None
    mul_181: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
    mul_182: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_181, arg364_1);  mul_181 = arg364_1 = None
    add_159: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_182, arg365_1);  mul_182 = arg365_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_525: "f32[512, 1536]" = torch.ops.aten.view.default(add_159, [512, 1536])
    permute_251: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg366_1, [1, 0]);  arg366_1 = None
    addmm_136: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg367_1, view_525, permute_251);  arg367_1 = view_525 = permute_251 = None
    view_526: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_136, [1, 512, 6144]);  addmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_183: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_526, 0.5)
    mul_184: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_526, 0.7071067811865476);  view_526 = None
    erf_22: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_184);  mul_184 = None
    add_160: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_185: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_183, add_160);  mul_183 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_527: "f32[512, 6144]" = torch.ops.aten.view.default(mul_185, [512, 6144]);  mul_185 = None
    permute_252: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg368_1, [1, 0]);  arg368_1 = None
    addmm_137: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg369_1, view_527, permute_252);  arg369_1 = view_527 = permute_252 = None
    view_528: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_137, [1, 512, 1536]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_161: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_528, add_159);  view_528 = add_159 = None
    var_mean_46 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_46[1];  var_mean_46 = None
    add_162: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-07);  getitem_92 = None
    rsqrt_46: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_69: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_161, getitem_93);  add_161 = getitem_93 = None
    mul_186: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
    mul_187: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_186, arg370_1);  mul_186 = arg370_1 = None
    add_163: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_187, arg371_1);  mul_187 = arg371_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_529: "f32[512, 1536]" = torch.ops.aten.view.default(add_163, [512, 1536])
    permute_253: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg372_1, [1, 0]);  arg372_1 = None
    addmm_138: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg373_1, view_529, permute_253);  arg373_1 = view_529 = permute_253 = None
    view_530: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_138, [1, 512, 1536]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_531: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_530, [1, 512, 24, -1]);  view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_254: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None
    clone_92: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    view_532: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_92, [-1, 512, 64]);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_533: "f32[512, 1536]" = torch.ops.aten.view.default(add_163, [512, 1536])
    permute_255: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg374_1, [1, 0]);  arg374_1 = None
    addmm_139: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg375_1, view_533, permute_255);  arg375_1 = view_533 = permute_255 = None
    view_534: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_139, [1, 512, 1536]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_535: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_534, [1, 512, 24, -1]);  view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_256: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
    clone_93: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    view_536: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_93, [-1, 512, 64]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_537: "f32[512, 1536]" = torch.ops.aten.view.default(add_163, [512, 1536])
    permute_257: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg376_1, [1, 0]);  arg376_1 = None
    addmm_140: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg377_1, view_537, permute_257);  arg377_1 = view_537 = permute_257 = None
    view_538: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_140, [1, 512, 1536]);  addmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_539: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_538, [1, 512, 24, -1]);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    permute_258: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
    clone_94: "f32[1, 24, 512, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    view_540: "f32[24, 512, 64]" = torch.ops.aten.view.default(clone_94, [-1, 512, 64]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant46 = self._tensor_constant46
    lift_fresh_copy_46: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant46);  _tensor_constant46 = None
    mul_188: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_46, 1);  lift_fresh_copy_46 = None
    sqrt_23: "f32[]" = torch.ops.aten.sqrt.default(mul_188);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    permute_259: "f32[24, 64, 512]" = torch.ops.aten.permute.default(view_536, [0, 2, 1]);  view_536 = None
    div_46: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(permute_259, sqrt_23);  permute_259 = sqrt_23 = None
    bmm_46: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_532, div_46);  view_532 = div_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_541: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_46, [-1, 24, 512, 512]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:110, code: rmask = ~(mask.to(torch.bool))
    convert_element_type_23: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  mul_3 = None
    bitwise_not_23: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_23);  convert_element_type_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:112, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
    _tensor_constant47 = self._tensor_constant47
    lift_fresh_copy_47: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant47);  _tensor_constant47 = None
    where_46: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_23, lift_fresh_copy_47, view_541);  lift_fresh_copy_47 = view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:113, code: output = torch.softmax(output, self.dim)
    amax_23: "f32[1, 24, 512, 1]" = torch.ops.aten.amax.default(where_46, [-1], True)
    sub_70: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where_46, amax_23);  where_46 = amax_23 = None
    exp_23: "f32[1, 24, 512, 512]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_24: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_47: "f32[1, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:114, code: output.masked_fill_(rmask, 0)
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_47: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(bitwise_not_23, scalar_tensor_23, div_47);  bitwise_not_23 = scalar_tensor_23 = div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    view_543: "f32[24, 512, 512]" = torch.ops.aten.view.default(where_47, [-1, 512, 512]);  where_47 = None
    bmm_47: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_543, view_540);  view_543 = view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_544: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_47, [-1, 24, 512, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_260: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:747, code: .contiguous()
    clone_95: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_545: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_95, [1, 512, -1]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_546: "f32[512, 1536]" = torch.ops.aten.view.default(view_545, [512, 1536]);  view_545 = None
    permute_261: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
    addmm_141: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg379_1, view_546, permute_261);  arg379_1 = view_546 = permute_261 = None
    view_547: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_141, [1, 512, 1536]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_164: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_547, add_163);  view_547 = add_163 = None
    var_mean_47 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 512, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 512, 1]" = var_mean_47[1];  var_mean_47 = None
    add_165: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-07);  getitem_94 = None
    rsqrt_47: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_71: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_164, getitem_95);  add_164 = getitem_95 = None
    mul_189: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
    mul_190: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_189, arg380_1);  mul_189 = arg380_1 = None
    add_166: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_190, arg381_1);  mul_190 = arg381_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_548: "f32[512, 1536]" = torch.ops.aten.view.default(add_166, [512, 1536])
    permute_262: "f32[1536, 6144]" = torch.ops.aten.permute.default(arg382_1, [1, 0]);  arg382_1 = None
    addmm_142: "f32[512, 6144]" = torch.ops.aten.addmm.default(arg383_1, view_548, permute_262);  arg383_1 = view_548 = permute_262 = None
    view_549: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_142, [1, 512, 6144]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_191: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_549, 0.5)
    mul_192: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_549, 0.7071067811865476);  view_549 = None
    erf_23: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_192);  mul_192 = None
    add_167: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_193: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_191, add_167);  mul_191 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_550: "f32[512, 6144]" = torch.ops.aten.view.default(mul_193, [512, 6144]);  mul_193 = None
    permute_263: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg384_1, [1, 0]);  arg384_1 = None
    addmm_143: "f32[512, 1536]" = torch.ops.aten.addmm.default(arg385_1, view_550, permute_263);  arg385_1 = view_550 = permute_263 = None
    view_551: "f32[1, 512, 1536]" = torch.ops.aten.view.default(addmm_143, [1, 512, 1536]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_168: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(view_551, add_166);  view_551 = add_166 = None
    var_mean_48 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 512, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 512, 1]" = var_mean_48[1];  var_mean_48 = None
    add_169: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-07);  getitem_96 = None
    rsqrt_48: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    sub_72: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(add_168, getitem_97);  add_168 = getitem_97 = None
    mul_194: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
    mul_195: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_194, arg386_1);  mul_194 = arg386_1 = None
    add_170: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_195, arg387_1);  mul_195 = arg387_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1513, code: logits = self.qa_outputs(sequence_output)
    view_552: "f32[512, 1536]" = torch.ops.aten.view.default(add_170, [512, 1536]);  add_170 = None
    permute_264: "f32[1536, 2]" = torch.ops.aten.permute.default(arg388_1, [1, 0]);  arg388_1 = None
    addmm_144: "f32[512, 2]" = torch.ops.aten.addmm.default(arg389_1, view_552, permute_264);  arg389_1 = view_552 = permute_264 = None
    view_553: "f32[1, 512, 2]" = torch.ops.aten.view.default(addmm_144, [1, 512, 2]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1514, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_553, [1, 1], 2);  view_553 = None
    getitem_98: "f32[1, 512, 1]" = split_with_sizes[0]
    getitem_99: "f32[1, 512, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_98, -1);  getitem_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1515, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_96: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # No stacktrace found for following nodes
    squeeze_2: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_99, -1);  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1516, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_97: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1527, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(arg392_1, 0);  arg392_1 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1528, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg393_1, 0);  arg393_1 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1531, code: start_loss = loss_fct(start_logits, start_positions)
    amax_24: "f32[1, 1]" = torch.ops.aten.amax.default(clone_96, [1], True)
    sub_73: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_96, amax_24);  amax_24 = None
    exp_24: "f32[1, 512]" = torch.ops.aten.exp.default(sub_73)
    sum_25: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_74: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_73, log);  sub_73 = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor_24: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_48: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, scalar_tensor_24);  ne = scalar_tensor_24 = None
    unsqueeze_4: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_48, 1);  where_48 = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_74, 1, unsqueeze_4);  sub_74 = unsqueeze_4 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_49: "f32[1]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_25);  ne_1 = neg = scalar_tensor_25 = None
    ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512);  clamp_max = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_24: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_49);  where_49 = None
    div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_24);  sum_27 = convert_element_type_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1532, code: end_loss = loss_fct(end_logits, end_positions)
    amax_25: "f32[1, 1]" = torch.ops.aten.amax.default(clone_97, [1], True)
    sub_75: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_97, amax_25);  amax_25 = None
    exp_25: "f32[1, 512]" = torch.ops.aten.exp.default(sub_75)
    sum_28: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [1], True);  exp_25 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_28);  sum_28 = None
    sub_76: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_75, log_1);  sub_75 = log_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_26: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_50: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, scalar_tensor_26);  ne_3 = scalar_tensor_26 = None
    unsqueeze_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_50, 1);  where_50 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_76, 1, unsqueeze_5);  sub_76 = unsqueeze_5 = None
    squeeze_4: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
    ne_4: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_51: "f32[1]" = torch.ops.aten.where.self(ne_4, neg_1, scalar_tensor_27);  ne_4 = neg_1 = scalar_tensor_27 = None
    ne_5: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512);  clamp_max_1 = None
    sum_29: "i64[]" = torch.ops.aten.sum.default(ne_5);  ne_5 = None
    convert_element_type_25: "f32[]" = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None
    sum_30: "f32[]" = torch.ops.aten.sum.default(where_51);  where_51 = None
    div_49: "f32[]" = torch.ops.aten.div.Tensor(sum_30, convert_element_type_25);  sum_30 = convert_element_type_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1533, code: total_loss = (start_loss + end_loss) / 2
    add_171: "f32[]" = torch.ops.aten.add.Tensor(div_48, div_49);  div_48 = div_49 = None
    div_50: "f32[]" = torch.ops.aten.div.Tensor(add_171, 2);  add_171 = None
    return (div_50, clone_96, clone_97)
    